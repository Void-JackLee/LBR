import os
import json
import math
import bisect
import torch
import torch.nn as nn
from peft import PeftModel
from typing import Optional, Tuple, Union, Unpack, Callable

from transformers.utils import add_start_docstrings_to_model_forward
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.models.llama.modeling_llama import (
    LlamaModel, LlamaForCausalLM,
    LLAMA_INPUTS_DOCSTRING, logger,
    Cache, DynamicCache
)

from src.model import get_base_model

class LayerItemAttnLlamaModel(LlamaModel):
    def add_attn_bias(
        self,
        causal_mask: torch.Tensor,
        dtype: torch.dtype,
        item_mask: torch.BoolTensor,
        item_len: torch.IntTensor,
        k: float = 0.,
        b: float = 1.
    ):

        batch_size = causal_mask.shape[0]
        seq_len = causal_mask.shape[-1]
        
        def s1():
            mask_bias = torch.zeros((batch_size,*causal_mask.shape[-2:]), dtype=dtype, device=causal_mask.device)

            for bi in range(batch_size):
                _item_len = item_len[bi]
                _item_len = _item_len[_item_len != 0]
                mean_len = _item_len.float().mean()
                imask = torch.where(item_mask[bi])[0]
                st = 0
                for _len in _item_len:
                    mask_bias[bi, :, imask[st:st + _len]] += (torch.log(torch.clamp(k * mean_len + b,min=1e-8)) - torch.log(torch.clamp(k * _len + b,min=1e-8)))
                    
                    st += _len
                if causal_mask.shape[-2] != 1: mask_bias[bi] = torch.tril(mask_bias[bi], diagonal=0)

            return mask_bias.unsqueeze(1) + causal_mask
        
        return s1()

    # copy from transformer 4.49.0
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        ### add extra param ###
        item_mask: torch.BoolTensor = None,
        item_len: torch.IntTensor = None,
        k: Union[float, torch.Tensor] = 0.,
        b: Union[float, torch.Tensor] = 1.,
        #######################
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        ## modify causal mask ##
        if causal_mask is None:
            dtype = inputs_embeds.dtype
            if attention_mask.shape[-1] == inputs_embeds.shape[1]:
                min_dtype = torch.finfo(dtype).min
                causal_mask = torch.full(
                    (inputs_embeds.shape[1], attention_mask.shape[-1]), 
                    fill_value=min_dtype, dtype=dtype, device=inputs_embeds.device
                )
                causal_mask = torch.triu(causal_mask, diagonal=1)
                causal_mask = causal_mask.expand(attention_mask.shape[0], 1, -1, -1)
            else:
                causal_mask = torch.zeros(attention_mask.shape[0], 1, inputs_embeds.shape[1], attention_mask.shape[-1], dtype=dtype).to(inputs_embeds.device)
        
        causal_mask = self.add_attn_bias(
            causal_mask=causal_mask,
            dtype=inputs_embeds.dtype,
            item_mask=item_mask,
            item_len=item_len,
            k = k,
            b = b
        )
        ########################

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()

class LayerLearnableItemAttnLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = LayerItemAttnLlamaModel(config)

    def init_meta(self, k = 0., b = 1.):
        self.k = torch.tensor(k, requires_grad=True, device=self.device)
        self.b = torch.tensor(b, requires_grad=True, device=self.device)

    def save_meta(self, output_dir):
        config = {
            "k": self.k.detach().cpu().item(),
            "b": self.b.detach().cpu().item(),
        }
        with open(os.path.join(output_dir, "attn_config.json"), "w") as f:
            json.dump(config, f, indent=4)
        torch.save({ "k": self.k, "b": self.b }, os.path.join(output_dir, "kb.pt"))

    def load_meta(self, output_dir):
        with open(os.path.join(output_dir, "attn_config.json"), "r") as f:
            config = json.load(f)
        self.init_meta(**config)
        kb = torch.load(os.path.join(output_dir, "kb.pt"), self.k.device)
        self.k = kb["k"]
        self.b = kb["b"]

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        item_mask: torch.BoolTensor = None,
        item_len: torch.IntTensor = None,
        **kwargs
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids) # [bs, token_len, dim]
        
        inputs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        pop_keys = []
        for k in inputs:
            if inputs[k] is None:
                pop_keys.append(k)
        for k in pop_keys: inputs.pop(k)
        return super().forward(
            item_mask=item_mask,
            item_len=item_len,
            k=self.k,
            b=self.b,
            **inputs,
            **kwargs
        )
    
    # inject extra input to model
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        item_mask: torch.BoolTensor = None, # extra input
        item_len: torch.IntTensor = None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs
        )
        # extra_inputs

        model_inputs["item_len"] = item_len
        model_inputs["item_mask"] = item_mask

        return model_inputs
    
def get_attn_model(
    lora_weights_path: str,
    base_model: str = "./Llama-3.2-3B/",
    compile = True,
    model_class=LayerLearnableItemAttnLlamaForCausalLM
):
    model, tokenizer = get_base_model(base_model, model_class=model_class)
    
    model = PeftModel.from_pretrained(model, lora_weights_path, torch_dtype=torch.bfloat16)
    model.merge_and_unload()
    model.generation_config.cache_implementation = "static"
    model.load_meta(lora_weights_path)
    if compile:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
    model.eval()

    return model, tokenizer