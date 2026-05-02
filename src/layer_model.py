import os
import json
import bisect
import torch

from torch import nn
from peft import PeftModel
from typing import Optional, Tuple, Union, Unpack
from transformers import LlamaModel, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LLAMA_INPUTS_DOCSTRING, logger
from transformers.utils import add_start_docstrings_to_model_forward
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.models.llama.modeling_llama import Cache, DynamicCache

from model import get_base_model

### 每层layer都输入position embedding
class LayerPositionLlamaModel(LlamaModel):
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
        ext_pos_embedding: Optional[Union[nn.Embedding, nn.ModuleList]] = None,
        share: bool = True,
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

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            ######## add extra position embeddings ########
            if ext_pos_embedding is not None:
                if share:
                    hidden_states = hidden_states + ext_pos_embedding
                else:
                    hidden_states = hidden_states + ext_pos_embedding[i]
            ###############################################

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
                    position_embeddings,
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

class LayerPositionLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = LayerPositionLlamaModel(config)

    def init_meta(self, item_group, alpha = 1., share = True):
        self.item_group = item_group
        self.share = share
        self.num_item_group = len(self.item_group)
        self.len_pos_embedding = nn.Embedding(self.num_item_group,self.model.config.hidden_size).to(self.device) if share else nn.ModuleList(
            [nn.Embedding(self.num_item_group,self.model.config.hidden_size) for layer_idx in range(self.model.config.num_hidden_layers)]
        ).to(self.device)
        if share:
            nn.init.constant_(self.len_pos_embedding.weight, 0)
        else:
            for layer_idx in range(self.model.config.num_hidden_layers):
                nn.init.constant_(self.len_pos_embedding[layer_idx].weight, 0)
        self.alpha = alpha

    def save_meta(self, output_dir):
        config = {
            "item_group": self.item_group,
            "alpha": self.alpha,
            "share": self.share
        }
        with open(os.path.join(output_dir, "len_pos_config.json"), "w") as f:
            json.dump(config, f, indent=4)
        torch.save(self.len_pos_embedding.state_dict(), os.path.join(output_dir, "len_pos_embedding.pt"))

    def load_meta(self, output_dir):
        with open(os.path.join(output_dir, "len_pos_config.json"), "r") as f:
            config = json.load(f)
        self.init_meta(**config)
        self.len_pos_embedding.load_state_dict(torch.load(os.path.join(output_dir, "len_pos_embedding.pt")))

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
        
        if self.share:
            ext_pos_embedding = torch.zeros_like(inputs_embeds).to(inputs_embeds.device)
        else:
            ext_pos_embedding = [torch.zeros_like(inputs_embeds).to(inputs_embeds.device) for _ in range(self.model.config.num_hidden_layers)]

        # add extra position embd to inputs_embeds
        if not("cache_position" in kwargs) or kwargs["cache_position"] is None or kwargs["cache_position"].shape[0] == item_mask.shape[1]:
            # training or generate at the first time
            
            batch_size = inputs_embeds.shape[0]
            for bi in range(batch_size):
                _item_len = item_len[bi].to(inputs_embeds.device)
                _item_len = _item_len[_item_len != 0]
                mask = torch.where(item_mask[bi])[0].to(inputs_embeds.device)
                st = 0
                for _len in _item_len:
                    group_idx = max(0, bisect.bisect_right(self.item_group,_len) - 1) # get group index of corresponding _len
                    group_idx = torch.tensor(group_idx).to(inputs_embeds.device)
                    if self.share:
                        ext_pos_embedding[bi, mask[st:st + _len]] += self.alpha * self.len_pos_embedding(group_idx) # add pos embeddings to item embeddings
                    else:
                        for i, pos_embed in enumerate(ext_pos_embedding):
                            pos_embed[bi, mask[st:st + _len]] += self.alpha * self.len_pos_embedding[i](group_idx) # add pos embeddings to item embeddings
                    st += _len
        
        inputs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "labels": labels
        }
        pop_keys = []
        for k in inputs:
            if inputs[k] is None:
                pop_keys.append(k)
        for k in pop_keys: inputs.pop(k)
        return super().forward(
            ext_pos_embedding=ext_pos_embedding,
            share=self.share,
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
    
def get_pos_model(
    lora_weights_path: str,
    base_model: str = "./Llama-3.2-3B/",
    model_class=LayerPositionLlamaForCausalLM,
    compile=True
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