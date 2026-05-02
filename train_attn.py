import json
import os
from typing import List, Optional

import fire
import torch
import transformers
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator
import torch.optim as optim

from datasets import Dataset

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
)

from src.utils import generate_prompt, generate_prompt_before_items
from src.dataset import get_dataset
from src.model import get_base_model
from src.attn_model import LayerLearnableItemAttnLlamaForCausalLM
from src.item_data import ItemDataProcessor

class CustomDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    default_labels = ["input_ids","attention_mask","labels"]
    
    def __call__(self, features):
        default_features = []
        for data in features:
            default_features.append({
                label: data[label] for label in self.default_labels
            })

        batch = super().__call__(default_features)

        # padding item_len and item_mask
        max_len = batch['input_ids'].shape[1]
        
        item_len = []
        item_mask = []
        for data in features:
            item_len.append([0] * (max_len - len(data["item_len"])) + data["item_len"])
            item_mask.append([False] * (max_len - len(data["item_mask"])) + data["item_mask"])

        batch['item_mask'] = torch.tensor(item_mask)
        batch['item_len'] = torch.tensor(item_len)
        
        return batch

class CustomTrainer(transformers.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.model.save_pretrained(output_dir, save_embedding_layers=False)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        # save extra pos embeddings
        if self.accelerator.is_main_process:
            self.model.model.save_meta(output_dir)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")

        outputs = model(**inputs)
        logits = outputs.logits # [bs, seq_len, vocab]

        loss = None
        # Shift so that tokens < n predict n
        _shift_logits = logits[..., :-1, :].contiguous()
        _shift_labels = labels[..., 1:].contiguous()

        # Flatten the tokens
        shift_logits = _shift_logits.view(-1, self.model.config.vocab_size)
        shift_labels = _shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)

        mask = shift_labels != -100
        shift_labels = shift_labels[mask]
        shift_logits = shift_logits[mask]

        pos_logits = torch.exp(shift_logits.gather(1, shift_labels.unsqueeze(1)).squeeze(1))
        pos_loss = -torch.log(pos_logits)

        neg_logits = torch.exp(shift_logits)
        neg_loss = torch.log(neg_logits.sum(dim=-1))

        loss = (pos_loss + neg_loss).mean()

        return (loss, outputs) if return_outputs else loss



def train(
    dataset_name: str,
    base_model: str = "Llama-3.2-3B",
    sample: int = -1,
    seed: int = 42,
    # Length Bias reduce hyperparams
    k: float = 0.,
    b: float = 1.,
    kb_learning_rate: float = 1e-3,
    # training hyperparams
    batch_size: int = 128,
    num_epochs: int = 20,
    learning_rate: float = 1e-4,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    train_on_inputs: int = 0,
):
    params = locals()
    transformers.set_seed(seed)
    accelerator = Accelerator()

    train_data, _ = get_dataset(dataset_name, prefix="train", sample=sample)

    dir_name = f"sample{sample}_epoch{num_epochs}_k{k}_b{b}"
    dir_name += f"_kblr{kb_learning_rate}"
    dir_name += f"_log(k*mx+b)-log(k*x+b)"

    father_path = os.path.join(
        f"./save_attn_lora_model/",
        dataset_name,
        base_model.replace('/','_'),
        dir_name
    )
    i = 0
    output_dir = os.path.join(father_path, str(i))
    while os.path.exists(output_dir):
        i += 1
        output_dir = os.path.join(father_path, str(i))
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "params.json"), "w") as f:
            json.dump(params, f, indent=4)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    micro_batch_size = batch_size // world_size
    gradient_accumulation_steps = batch_size // micro_batch_size // world_size

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model, tokenizer = get_base_model(base_model, bnb_config, LayerLearnableItemAttnLlamaForCausalLM)

    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    itemDataProcessor = ItemDataProcessor(tokenizer)

    train_data = Dataset.from_list(train_data)
    train_data = train_data.shuffle(seed=seed)
    train_data = train_data.map(lambda x: itemDataProcessor.generate_and_tokenize_prompt(x, train_on_inputs))
    
    model.init_meta(k, b)

    trainer = CustomTrainer(
        model=model,
        train_dataset=train_data,
        data_collator=CustomDataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
        args=transformers.TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=20,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True,
            tf32=True,
            optim="adamw_torch",
            logging_strategy="steps",
            logging_steps=0.1,
            save_strategy="steps",
            save_steps=(1 / (num_epochs)),
            save_on_each_node=False,
            log_on_each_node=False,
            ddp_find_unused_parameters=False if (world_size != 1) else None,
            report_to="tensorboard",
            local_rank=int(os.environ.get("LOCAL_RANK", -1)),
            seed=seed,
            data_seed=seed,
            label_names=['item_len','var_input','var_output_label'] # extra key for var calc
        ),
    )

    ############## add k,b training ##############

    def inject_trainer_optimizer(trainer, model, step_fn, zero_grad_fn):
        super_create_optimizer_and_scheduler = trainer.create_optimizer_and_scheduler
        super_zero_grad = model.zero_grad

        def inject_optimizer(self):
            super_step = self.step
            
            def step(*pargs, **args):
                super_step(*pargs, **args)
                step_fn()
            self.step = step

        def create_optimizer_and_scheduler(num_training_steps: int):
            super_create_optimizer_and_scheduler(num_training_steps)
            inject_optimizer(trainer.optimizer)

        trainer.create_optimizer_and_scheduler = create_optimizer_and_scheduler

        def zero_grad():
            super_zero_grad()
            zero_grad_fn()
        model.zero_grad = zero_grad


    optimizer = optim.AdamW([model.k, model.b], lr=kb_learning_rate)
    inject_trainer_optimizer(
        trainer,
        model,
        step_fn=optimizer.step,
        zero_grad_fn=optimizer.zero_grad
    )

    ##############################################
    
    model.config.use_cache = False
    trainer.train()
    model.save_pretrained(output_dir, save_embedding_layers=False)
    model.model.save_meta(output_dir)

if __name__ == "__main__":
    fire.Fire(train)
