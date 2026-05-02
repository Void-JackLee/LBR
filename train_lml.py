import json
import os
from typing import List, Optional

import fire
import torch
import transformers
from accelerate import Accelerator

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

from src.utils import generate_prompt
from src.dataset import get_dataset
from src.model import get_base_model

class CustomTrainer(transformers.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.model.save_pretrained(output_dir, save_embedding_layers=False)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


def train(
    dataset_name: str,
    base_model: str = "Llama-3.2-3B",
    sample: int = -1,
    seed: int = 42,
    # training hyperparams
    batch_size: int = 128,
    num_epochs: int = 10,
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

    father_path = os.path.join(
        f"./save_lora_model/",
        dataset_name,
        base_model.replace('/','_'),
        f"sample{sample}_epoch{num_epochs}",
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
    model, tokenizer = get_base_model(base_model, bnb_config)

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

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(prompt, padding=False, return_tensors=None)
        if result["input_ids"][-1] != tokenizer.eos_token_id and add_eos_token:
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)

        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]

        return tokenized_full_prompt

    train_data = Dataset.from_list(train_data)
    train_data = train_data.shuffle(seed=seed)
    train_data = train_data.map(lambda x: generate_and_tokenize_prompt(x))

    trainer = CustomTrainer(
        model=model,
        train_dataset=train_data,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
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
        ),
    )
    model.config.use_cache = False
    trainer.train()
    model.save_pretrained(output_dir, save_embedding_layers=False)


if __name__ == "__main__":
    fire.Fire(train)
