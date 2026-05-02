from typing import Literal
from captum.attr import (
    FeatureAblation, 
    LayerIntegratedGradients, 
    LLMAttribution, 
    LLMGradientAttribution, 
    TextTemplateInput,
)

from .utils import generate_prompt

def get_text(data):
    template = generate_prompt({**data,"output": ""},template=True)
    values = [f'"{item}"' for item in data["input_arr"]]
    target = data["output"]
    return template, values, target

def get_llm_attr(model, tokenizer, type: Literal['fa', 'grad'] = 'fa'):
    if type == 'grad':
        lig = LayerIntegratedGradients(model, model.model.model.embed_tokens)
        llm_attr = LLMGradientAttribution(lig, tokenizer)
    elif type == 'fa':
        fa = FeatureAblation(model)
        llm_attr = LLMAttribution(fa, tokenizer)
    else:
        assert 1 != 1, "not valid type in ['fa', 'grad']"
    return llm_attr

def run_single_score(llm_attr, data, debug = False):
    template, values, target = get_text(data)
    if debug:
        print(template)
        print(values)

    skip_tokens = [128000]

    inp = TextTemplateInput(
        template=template,
        values=values
    )

    attr_res = llm_attr.attribute(inp, target=target, use_cached_outputs=False, skip_tokens=skip_tokens)

    if debug:
        print("attr to the output tokens:", attr_res.token_attr.shape)  # shape(n_output_token, n_input_token)
        attr_res.plot_token_attr(show=True)
        print("attr to the output sequence:", attr_res.seq_attr.shape)  # shape(n_input_token)
        attr_res.plot_seq_attr(show=True)
    return attr_res, values