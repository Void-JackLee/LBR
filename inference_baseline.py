import ast
import pandas as pd
import fire
import torch

import json
from tqdm import tqdm
import os

from accelerate import Accelerator
from accelerate.utils import gather_object
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from genre.trie import MarisaTrie
import transformers
from src.utils import get_prompt, generate_prompt
from src.dataset import generate_list_from_csv, get_dataset
from src.model import get_model

def main(
    lora_weights_path: str,
    dataset_name: str,
    sample: int = 5000,
    batch_size: int = 8,
    base_model: str = "Llama-3.2-3B",
    num_beams: int = 10
):
    transformers.set_seed(42)
    accelerator = Accelerator()
    accelerator.print("Dataset: ", dataset_name)
    accelerator.print("LoRA Weights Path: ", lora_weights_path)

    test_data, id2title_dict = get_dataset(dataset_name, sample=sample)

    result_json_data = f"predict_{dataset_name}_{sample}_CBS"
    result_json_data = os.path.join(lora_weights_path, result_json_data + ".json")

    if os.path.exists(result_json_data):
        accelerator.print(f"The {result_json_data} has existed.")
        return
    accelerator.wait_for_everyone()

    model, tokenizer = get_model(lora_weights_path, base_model, compile=True)

    sep = tokenizer.encode("### Response:\n", add_special_tokens=False)  # [14711, 6075, 512]
    titles_list = list(id2title_dict.values())
    tokens_list = [
        tokenizer.encode("### Response:\n" + f'"{title}"', add_special_tokens=False) for title in titles_list
    ]
    trie = MarisaTrie(tokens_list)

    def prefix_allowed_tokens_fn(batch_id: int, input_ids: torch.Tensor) -> list:
        input_ids = input_ids.tolist()
        for i in range(len(input_ids)):
            if input_ids[i : i + len(sep)] == sep:
                break

        prefix = input_ids[i:]
        allowed_tokens = trie.get(prefix)
        allowed_tokens = [tokenizer.eos_token_id] if allowed_tokens == [] else allowed_tokens

        return allowed_tokens

    def evaluate(instructions, inputs, num_beams=num_beams, max_new_tokens=128):
        prompt = [generate_prompt({"instruction": instruction, "input" :input, "output": ""}) for instruction, input in zip(instructions, inputs)]
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(accelerator.device)

        with torch.no_grad():
            generation_config = GenerationConfig(
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
            )
            generation_output = model.generate(
                **inputs,
                generation_config=generation_config,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            )

            sequences_scores = generation_output.sequences_scores.tolist()
            sequences_scores = [
                sequences_scores[i * num_beams : (i + 1) * num_beams] for i in range(len(sequences_scores) // num_beams)
            ]

            output_seq = generation_output.sequences
            output = tokenizer.batch_decode(output_seq, skip_special_tokens=True)
            output = [_.split("Response:\n")[-1] for _ in output]
            real_outputs = [output[i * num_beams : (i + 1) * num_beams] for i in range(len(output) // num_beams)]

        return real_outputs, sequences_scores

    def batch(list, batch_size=batch_size):
        chunk_size = (len(list) - 1) // batch_size + 1
        for i in range(chunk_size):
            yield list[batch_size * i : batch_size * (i + 1)]

    instructions = [_["instruction"] for _ in test_data]
    inputs = [_["input"] for _ in test_data]
    input_dict = {"instructions": instructions, "inputs": inputs}

    with accelerator.split_between_processes(input_dict) as input_temp:
        outputs = []
        sequences_scores = []

        for batch1 in tqdm(
            zip(batch(input_temp["instructions"]), batch(input_temp["inputs"])),
            total=(len(input_temp["instructions"]) + batch_size - 1) // batch_size,
        ):
            instructions, inputs = batch1
            output, sequences_score = evaluate(instructions, inputs)
            outputs.extend(output)
            sequences_scores.extend(sequences_score)

    outputs = gather_object(outputs)
    sequences_scores = gather_object(sequences_scores)
    assert len(outputs) == len(test_data)
    assert len(sequences_scores) == len(test_data)

    if accelerator.is_main_process:
        for i, _ in enumerate(test_data):
            test_data[i]["predict"] = outputs[i]
            test_data[i]["scores"] = sequences_scores[i]

        with open(result_json_data, "w") as f:
            json.dump(test_data, f, indent=4)

if __name__ == "__main__":
    fire.Fire(main)
