import argparse

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
    set_seed,
)

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_path', default=None, help='ckpt path')
parser.add_argument('--base_model', default='meta-llama/Llama-2-7b-hf', help='base model')

args = parser.parse_args()

set_seed(42)
BASE_MODEL = args.base_model
ckpt_path = args.ckpt_path
assert (
    BASE_MODEL
), "Please specify a value for BASE_MODEL environment variable, e.g. `export BASE_MODEL=huggyllama/llama-7b`"

if 'gemma' in BASE_MODEL or 'Llama-3.1' in BASE_MODEL:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map={"": "cuda"},
        # gemma-2 soft-caps the attention logits, which sdpa/flash cannot express
        attn_implementation="eager" if "gemma-2" in BASE_MODEL else None,
    )
else:
    tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)

    base_model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map={"": "cuda"},
    )

first_weight = base_model.model.layers[0].self_attn.q_proj.weight
first_weight_old = first_weight.clone()

model = PeftModel.from_pretrained(
    base_model,
    ckpt_path,
    device_map={"": "cuda"},
    torch_dtype=torch.float16,
)

model = model.merge_and_unload()

merged_weight = model.model.layers[0].self_attn.q_proj.weight
assert not torch.allclose(first_weight_old, merged_weight)

save_path = f'{ckpt_path}/merged_model'
model.save_pretrained(save_path, max_shard_size="20GB")
tokenizer.save_pretrained(save_path)
