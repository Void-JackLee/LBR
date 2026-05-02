import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import (
    GenerationMixin,
    logger, Cache, FlashAttentionKwargs,
    apply_rotary_pos_emb, ALL_ATTENTION_FUNCTIONS, repeat_kv
)
from typing import (
    Tuple, Optional, Unpack, Callable,
    List, Literal, Union
)

## 改模型，使其可以获得softmax之前的weight
def inject_model_attn_score(model: GenerationMixin):
    def eager_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float = 0.0,
        **kwargs,
    ):
        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)

        _attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            _attn_weights = _attn_weights + causal_mask

        attn_weights = nn.functional.softmax(_attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output, _attn_weights
    def put_forward(self):
        def forward(
            hidden_states: torch.Tensor,
            position_embeddings: Tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor],
            past_key_value: Optional[Cache] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs: Unpack[FlashAttentionKwargs],
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)

            query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            if past_key_value is not None:
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

            attention_interface: Callable = eager_attention_forward
            if self.config._attn_implementation != "eager":
                if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                    logger.warning_once(
                        "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                        'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                    )
                else:
                    attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **kwargs,
            )

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            return attn_output, attn_weights
        
        self.forward = forward
    
    for layer in model.model.layers:
        put_forward(layer.self_attn)

def run_get_attn(model, item):
    inputs = {
        "input_ids": torch.tensor([item["input_ids"]]).to(model.device),
        "attention_mask": torch.tensor([item["attention_mask"]]).to(model.device)
    }
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        attentions = outputs.attentions
    # print(attentions[0].shape) # [bs, head, seq_len, seq_len]
    # print(outputs.logits.shape) # [bs, seq_len, vocab_size(probs of every token)]
    return attentions

def run_get_attn_with_item_mask(model, item):
    inputs = {
        "input_ids": torch.tensor([item["input_ids"]]).to(model.device),
        "attention_mask": torch.tensor([item["attention_mask"]]).to(model.device),
        "item_len": torch.tensor([item["item_len"]]).to(model.device),
        "item_mask": torch.tensor([item["item_mask"]]).to(model.device)
    }
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        attentions = outputs.attentions
    # print(attentions[0].shape) # [bs, head, seq_len, seq_len]
    # print(outputs.logits.shape) # [bs, seq_len, vocab_size(probs of every token)]
    return attentions

def aggr_attn(
    attentions: List[torch.Tensor], 
    layer: Union[Literal["mean"], int] = -1, 
    head: Union[Literal["mean"], int] = "mean", 
    batch: Union[Literal[None], int] = 0, 
    value_type: Literal["score", "weight"] = "weight"
): # [(bs,) seq_len, seq_len]
    if layer == "mean":
        layers = []
        for multi_head_attn in attentions:
            if head == "mean":
                attn = multi_head_attn.mean(1)
            else:
                attn = multi_head_attn[:, head, :, :]  # [bs, seq_len, seq_len]
            if value_type == "weight":
                attn_value = nn.functional.softmax(attn, dim=-1, dtype=torch.float32).to(attn.dtype)
            else:
                attn_value = attn
            layers.append(attn_value if batch is None else attn_value[batch])
        layers = torch.stack(layers) # [layers,(bs), seq_len, seq_len]
        return layers.mean(0)
    else:
        multi_head_attn = attentions[layer] # [bs, head, seq_len, seq_len]
        if head == "mean":
            attn = multi_head_attn.mean(1)
        else:
            attn = multi_head_attn[:, head, :, :]  # [bs, seq_len, seq_len]
        if value_type == "weight":
            attn_value = nn.functional.softmax(attn, dim=-1, dtype=torch.float32).to(attn.dtype)
        else:
            attn_value = attn
        if batch is None:
            return attn_value
        return attn_value[batch]
    