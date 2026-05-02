import ast
import pandas as pd
import fire
import torch
import math
import json
import os
from tqdm import tqdm
import torch.nn as nn

from typing import Callable, List, Optional, Union, Dict, Tuple
from collections import UserDict
from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.generation.logits_process import PrefixConstrainedLogitsProcessor,add_start_docstrings,LOGITS_PROCESSOR_INPUTS_DOCSTRING
from transformers.generation.utils import LogitsProcessorList
from transformers.generation.beam_search import BeamHypotheses, BeamScorer

from genre.trie import MarisaTrie
import transformers
from src.utils import get_prompt, generate_prompt, generate_prompt_before_items
from src.dataset import generate_list_from_csv, get_dataset
from src.attn_model import get_attn_model, LayerLearnableItemAttnLlamaForCausalLM
from src.item_data import ItemDataProcessor
from src.customCBS import dlp_beam_search

### u_t = alpha * log(k) + beta
def u_t(k, alpha, beta, vocab_size):
    return alpha * math.log(k) + beta

class WeightedLogitsProcessor(PrefixConstrainedLogitsProcessor):
    def __init__(self, prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]], num_beams: int, alpha: float = 0.0, beta: float = 0.0):
        super().__init__(prefix_allowed_tokens_fn,num_beams)
        self._alpha = alpha
        self._beta = beta

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        vocab_size = scores.shape[1] # [batch * num_beams, vocab_size]
        for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent):
                prefix_allowed_tokens = self._prefix_allowed_tokens_fn(batch_id, sent)
                _len = len(prefix_allowed_tokens)
                if _len == 0:
                    raise ValueError(
                        f"`prefix_allowed_tokens_fn` returned an empty list for batch ID {batch_id}."
                        f"This means that the constraint is unsatisfiable. Please check your implementation"
                        f"of `prefix_allowed_tokens_fn` "
                    )
                ### \sum u_t * log(p_t)
                scores[batch_id * self._num_beams + beam_id, prefix_allowed_tokens] *= u_t(_len, self._alpha, self._beta, vocab_size)

        return scores

class DynamicLengthPenaltyBeamHypotheses(BeamHypotheses):
    def calc_dynamic_length_penalty(
        self,
        input_ids: torch.LongTensor, # [seq_len]
        decoder_prompt_len: Optional[int] = 0
    ) -> float:
        raise NotImplementedError("This method should be patched.")

    def add(
        self,
        hyp: torch.LongTensor,
        sum_logprobs: float,
        beam_indices: Optional[torch.LongTensor] = None,
        generated_len: Optional[int] = None,
    ):
        """
        Add a new hypothesis to the list.
        """
        if generated_len is not None:
            decoder_prompt_len = hyp.shape[-1] - generated_len
            score = sum_logprobs / (self.calc_dynamic_length_penalty(hyp, decoder_prompt_len)**self.length_penalty)
        else:
            raise ValueError("generated_len must be provided for DynamicLengthPenaltyBeamHypotheses.")

        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp, beam_indices))
            if len(self) > self.num_beams:
                sorted_next_scores = sorted([(s, idx) for idx, (s, _, _) in enumerate(self.beams)])
                del self.beams[sorted_next_scores[0][1]]
                self.worst_score = sorted_next_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(
        self, 
        best_sum_logprobs: float, 
        best_input_ids: torch.LongTensor,
        decoder_prompt_len: Optional[int] = 0
    ) -> bool:
        """
        If there are enough hypotheses and that none of the hypotheses being generated can become better than the worst
        one in the heap, then we are done with this sentence.
        """

        if len(self) < self.num_beams:
            return False

        # `True`: stop as soon as at least `num_beams` hypotheses are finished
        if self.early_stopping is True:
            return True
        # `False`: heuristic -- compute best possible score from `cur_len`, even though it is not entirely accurate
        #  when `length_penalty` is positive. See the discussion below for more details.
        # https://github.com/huggingface/transformers/pull/20901#issuecomment-1369845565
        elif self.early_stopping is False:
            highest_attainable_score = best_sum_logprobs / (self.calc_dynamic_length_penalty(best_input_ids, decoder_prompt_len) ** self.length_penalty)
            ret = self.worst_score >= highest_attainable_score
            return ret
        # `"never"`: compute the best possible score, depending on the signal of `length_penalty`
        else:
            raise ValueError("Unsupported early_stopping value.")

def patch_dlp(
    prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
    vocab_size: int,
    alpha: float = 0, 
    beta: float = 0,
):
    ### u_t = alpha * log_n(k) + beta
    def calc_single_dynamic_length_penalty(
        input_id: torch.LongTensor, # [seq_len]
    ) -> float:
        prefix_allowed_tokens = prefix_allowed_tokens_fn(0, input_id)
        Lr = len(prefix_allowed_tokens)
        if Lr == 0:
            return 0
        return u_t(Lr, alpha, beta, vocab_size)

    ### U_t = \sum u_t
    def calc_dynamic_length_penalty(
        input_ids: torch.LongTensor, # [seq_len]
        decoder_prompt_len: int = 0
    ) -> float:
        # return input_ids.shape[-1] + 1 - decoder_prompt_len # default
        Leff = 0
        for r in range(decoder_prompt_len - 1, input_ids.shape[-1]):
            Leff += calc_single_dynamic_length_penalty(input_ids[:r + 1])

        return Leff

    def beam_scorer_patch_fn(
        self: BeamScorer
    ):
        def process(
            input_ids: torch.LongTensor,
            next_scores: torch.FloatTensor, # [batch_size, num_beams * 2]
            next_tokens: torch.LongTensor,
            next_indices: torch.LongTensor,
            pad_token_id: Optional[Union[int, torch.Tensor]] = None,
            eos_token_id: Optional[Union[int, List[int], torch.Tensor]] = None,
            beam_indices: Optional[torch.LongTensor] = None,
            group_index: Optional[int] = 0,
            decoder_prompt_len: Optional[int] = 0,
        ) -> Dict[str, torch.Tensor]:
            # add up to the length which the next_scores is calculated on (including decoder prompt)
            ### cur_len = input_ids.shape[-1] + 1
            batch_size = len(self._beam_hyps) // self.num_beam_groups

            if not (batch_size == (input_ids.shape[0] // self.group_size)):
                if self.num_beam_groups > 1:
                    raise ValueError(
                        f"A group beam size of {input_ids.shape[0]} is used as the input, but a group beam "
                        f"size of {self.group_size} is expected by the beam scorer."
                    )
                else:
                    raise ValueError(
                        f"A beam size of {input_ids.shape[0]} is used as the input, but a beam size of "
                        f"{self.group_size} is expected by the beam scorer."
                    )

            device = input_ids.device
            next_beam_scores = torch.zeros((batch_size, self.group_size), dtype=next_scores.dtype, device=device)
            next_beam_tokens = torch.zeros((batch_size, self.group_size), dtype=next_tokens.dtype, device=device)
            next_beam_indices = torch.zeros((batch_size, self.group_size), dtype=next_indices.dtype, device=device)

            if eos_token_id is not None and not isinstance(eos_token_id, torch.Tensor):
                if isinstance(eos_token_id, int):
                    eos_token_id = [eos_token_id]
                eos_token_id = torch.tensor(eos_token_id)

            for batch_idx in range(batch_size):
                batch_group_idx = batch_idx * self.num_beam_groups + group_index
                if self._done[batch_group_idx]:
                    if self.num_beams < len(self._beam_hyps[batch_group_idx]):
                        raise ValueError(f"Batch can only be done if at least {self.num_beams} beams have been generated")
                    if eos_token_id is None or pad_token_id is None:
                        raise ValueError("Generated beams >= num_beams -> eos_token_id and pad_token have to be defined")
                    # pad the batch
                    next_beam_scores[batch_idx, :] = 0
                    next_beam_tokens[batch_idx, :] = pad_token_id
                    next_beam_indices[batch_idx, :] = 0
                    continue

                # next tokens for this sentence
                beam_idx = 0
                for beam_token_rank, (next_token, next_score, next_index) in enumerate(
                    zip(next_tokens[batch_idx], next_scores[batch_idx], next_indices[batch_idx])
                ):
                    batch_beam_idx = batch_idx * self.group_size + next_index
                    # add to generated hypotheses if end of sentence
                    if (eos_token_id is not None) and (next_token.item() in eos_token_id):
                        # if beam_token does not belong to top num_beams tokens, it should not be added
                        is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.group_size
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        if beam_indices is not None:
                            beam_index = beam_indices[batch_beam_idx]
                            beam_index = beam_index + (batch_beam_idx,)
                        else:
                            beam_index = None

                        self._beam_hyps[batch_group_idx].add(
                            input_ids[batch_beam_idx].clone(),
                            next_score.item(),
                            beam_indices=beam_index,
                            generated_len=input_ids.shape[-1] - decoder_prompt_len, ### modified
                        )
                    else:
                        # add next predicted token since it is not eos_token
                        next_beam_scores[batch_idx, beam_idx] = next_score
                        next_beam_tokens[batch_idx, beam_idx] = next_token
                        next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                        beam_idx += 1

                    # once the beam for next step is full, don't add more tokens to it.
                    if beam_idx == self.group_size:
                        break

                if beam_idx < self.group_size:
                    raise ValueError(
                        f"At most {self.group_size} tokens in {next_tokens[batch_idx]} can be equal to `eos_token_id:"
                        f" {eos_token_id}`. Make sure {next_tokens[batch_idx]} are corrected."
                    )

                # Check if we are done so that we can save a pad step if all(done)
                ### Modified, input best_input_ids instead of cur_len
                best_idx = next_scores[batch_idx].argmax().item()
                best_batch_beam_idx = batch_idx * self.group_size + next_indices[batch_idx][best_idx].item()
                self._done[batch_group_idx] = self._done[batch_group_idx] or self._beam_hyps[batch_group_idx].is_done(
                    next_scores[batch_idx].max().item(), input_ids[best_batch_beam_idx], decoder_prompt_len
                )

            return UserDict(
                {
                    "next_beam_scores": next_beam_scores.view(-1),
                    "next_beam_tokens": next_beam_tokens.view(-1),
                    "next_beam_indices": next_beam_indices.view(-1),
                }
            )

        # replace class
        for hyp in self._beam_hyps:
            hyp.__class__ = DynamicLengthPenaltyBeamHypotheses
            hyp.calc_dynamic_length_penalty = calc_dynamic_length_penalty
        # patch method
        self.process = process
    
    def dlp_calc_fn(
        input_ids: torch.LongTensor, # [batch_size * num_beams, seq_len]
    ) -> torch.Tensor: # [batch_size * num_beams]
        dlp = []
        for beam_idx, input_id in enumerate(input_ids): # [seq_len]
            Leff = calc_single_dynamic_length_penalty(input_id)
            dlp.append(Leff)
        return torch.tensor(dlp).to(input_ids.device)

    return {
        "beam_scorer_patch_fn": beam_scorer_patch_fn,
        "dlp_calc_fn": dlp_calc_fn
    }

def apply_dlp_model(
    model, 
    prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]], 
    num_beams: int, 
    alpha: float,
    beta: float
):  
    weighted_logits_processor = WeightedLogitsProcessor(prefix_allowed_tokens_fn, num_beams, alpha, beta)

    def constrain_after_softmax(
        input_ids: torch.LongTensor,
        next_token_logits: torch.Tensor,
        logits_processor: LogitsProcessorList,
    ):
        next_token_scores = nn.functional.log_softmax(
            next_token_logits, dim=-1
        )  # (batch_size * num_beams, vocab_size)
        next_token_scores_processed = logits_processor(input_ids, next_token_scores)
        next_token_scores_processed = weighted_logits_processor(input_ids, next_token_scores_processed)
        return next_token_scores_processed
    
    dlp_beam_search(model.model, 
                    **patch_dlp(prefix_allowed_tokens_fn, model.model.config.vocab_size, alpha, beta),
                    logits_fn=constrain_after_softmax)


def main(
    lora_weights_path: str,
    dataset_name: str,
    sample: int = 5000,
    batch_size: int = 8,
    base_model: str = "Llama-3.2-3B",
    num_beams: int = 10,
    length_penalty: float = 1.0,
    alpha: float = 1.0,
    beta: float = 0.0,
    name: str = '"Ut^-lp*sum(ut*log(softmax(z))),ut=a*log(k)+b,Ut=sum(ut)"'
):
    transformers.set_seed(42)
    accelerator = Accelerator()
    accelerator.print("Dataset: ", dataset_name)
    accelerator.print("LoRA Weights Path: ", lora_weights_path)

    test_data, id2title_dict = get_dataset(dataset_name, sample=sample)

    os.makedirs(os.path.join(lora_weights_path, name), exist_ok=True)
    result_json_data = os.path.join(name,f"predict_{dataset_name}_{sample}_lp{length_penalty}_alpha{alpha}_beta{beta}")
    result_json_data += "_CBS"
    result_json_data = os.path.join(lora_weights_path, result_json_data + ".json")

    if os.path.exists(result_json_data):
        accelerator.print(f"The {result_json_data} has existed.")
        return
    accelerator.wait_for_everyone()

    model, tokenizer = get_attn_model(lora_weights_path, base_model, True, LayerLearnableItemAttnLlamaForCausalLM)

    sep = tokenizer.encode("### Response:\n", add_special_tokens=False)  # [14711, 6075, 512]
    titles_list = list(id2title_dict.values())
    tokens_list = [
        tokenizer.encode("### Response:\n" + f'"{title}"', add_special_tokens=False) for title in titles_list
    ]
    trie = MarisaTrie(tokens_list)

    itemDataProcessor = ItemDataProcessor(tokenizer)

    def extend_data_item(data_point):
        prompt = generate_prompt({"instruction": data_point["instruction"], "input": data_point["input"], "output": ""})
        item_meta = itemDataProcessor.get_item_mask(data_point, tokenizer(prompt, padding=False, return_tensors=None))
        return {
            **data_point,
            **item_meta,
            "prompt": prompt
        }

    def prefix_allowed_tokens_fn(batch_id: int, input_ids: torch.Tensor) -> list:
        input_ids = input_ids.tolist()
        for i in range(len(input_ids)):
            if input_ids[i : i + len(sep)] == sep:
                break

        prefix = input_ids[i:]
        allowed_tokens = trie.get(prefix)
        allowed_tokens = [tokenizer.eos_token_id] if allowed_tokens == [] else allowed_tokens

        return allowed_tokens

    def evaluate(prompts, item_lens, item_masks, num_beams=num_beams, max_new_tokens=128):
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        # padding
        max_len = inputs["input_ids"].shape[1]
        item_lens = list(map(lambda item_len: [0] * (max_len - len(item_len)) + item_len, item_lens))
        item_masks = list(map(lambda item_mask: [False] * (max_len - len(item_mask)) + item_mask, item_masks))

        inputs = {
            **inputs,
            "item_len": torch.tensor(item_lens).to(model.device),
            "item_mask": torch.tensor(item_masks).to(model.device)
        }

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

    apply_dlp_model(model,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                    num_beams=num_beams,
                    alpha=alpha, beta=beta)

    test_data = list(map(extend_data_item, tqdm(test_data)))
    prompts = [_["prompt"] for _ in test_data]
    item_lens = [_["item_len"] for _ in test_data]
    item_masks = [_["item_mask"] for _ in test_data]
    input_dict = {"prompts": prompts, "item_lens": item_lens, "item_masks": item_masks}

    with accelerator.split_between_processes(input_dict) as input_temp:
        outputs = []
        sequences_scores = []

        for batch1 in tqdm(
            zip(batch(input_temp["prompts"]), batch(input_temp["item_lens"]), batch(input_temp["item_masks"])),
            total=(len(input_temp["prompts"]) + batch_size - 1) // batch_size,
        ):
            prompts, item_lens, item_masks = batch1
            output, sequences_score = evaluate(prompts, item_lens, item_masks)
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
