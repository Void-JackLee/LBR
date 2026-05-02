import os
import torch

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_base_model(
    model_name: str = "./Llama-3.2-3B/",
    bnb_config = None,
    model_class = AutoModelForCausalLM
):
    model = model_class.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "left"

    return model, tokenizer

def get_model(
    lora_weights_path: str,
    base_model: str = "./Llama-3.2-3B/",
    compile = True,
    model_class = AutoModelForCausalLM
):
    model, tokenizer = get_base_model(base_model, model_class=model_class)

    model = PeftModel.from_pretrained(model, lora_weights_path, torch_dtype=torch.bfloat16)
    model.merge_and_unload()
    model.generation_config.cache_implementation = "static"
    if compile:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
    model.eval()

    return model, tokenizer