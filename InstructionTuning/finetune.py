import os
import sys
from os.path import join
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
    set_seed,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from peft import (
    FrameConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

from utils.prompter import Prompter


class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        print('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)

        touch(join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)


def train(
    base_model: str = "",
    data_path: str = "yahma/alpaca-cleaned",
    output_dir: str = "./frame-alpaca",
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 4e-3,
    cutoff_len: int = 256,
    val_set_size: int = 2000,
    seed: int = 42,
    target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    n_ff_coeffs: int = 1000,
    frame_scale: float = 256.0,
    init_std: float = 1.0,
    tff_l: int = 2,
    tff_l_out: int = None,
    tff_block_size: int = 4096,
    tff_block_size_out: int = None,
    tff_num_blocks: int = None,
    share_entry: bool = False,
    entry_seed: int = 2024,
    train_on_inputs: bool = True,
    add_eos_token: bool = False,
    group_by_length: bool = False,
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",
    wandb_log_model: str = "",
    resume_from_checkpoint: str = None,
    prompt_template_name: str = "alpaca",
):
    set_seed(seed)

    _frac = os.environ.get("FRAMEFT_MEM_FRACTION", "")
    if _frac and torch.cuda.is_available():
        for _d in range(torch.cuda.device_count()):
            torch.cuda.set_per_process_memory_fraction(float(_frac), _d)
        print(f"[frameft] capped GPU memory to {float(_frac):.0%} per device")

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-FrameFT model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"target_modules: {target_modules}\n"
            f"n_ff_coeffs: {n_ff_coeffs}\n"
            f"frame_scale: {frame_scale}\n"
            f"init_std: {init_std}\n"
            f"tff_l: {tff_l}\n"
            f"tff_l_out: {tff_l_out}\n"
            f"tff_block_size: {tff_block_size}\n"
            f"tff_block_size_out: {tff_block_size_out}\n"
            f"tff_num_blocks: {tff_num_blocks}\n"
            f"share_entry: {share_entry}\n"
            f"entry_seed: {entry_seed}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    if 'gemma' in base_model or 'Llama-3.1' in base_model:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map=device_map,
            attn_implementation="eager" if "gemma-2" in base_model else None,
        )

        tokenizer = AutoTokenizer.from_pretrained(base_model)
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map=device_map,
        )

        tokenizer = LlamaTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = (
        0
    )
    tokenizer.padding_side = "left"

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]
        return tokenized_full_prompt

    config = FrameConfig(
        n_ff_coeffs=n_ff_coeffs,
        scale=frame_scale,
        init_std=init_std,
        tff_l=tff_l,
        tff_l_out=tff_l_out,
        tff_block_size=tff_block_size,
        tff_block_size_out=tff_block_size_out,
        tff_num_blocks=tff_num_blocks,
        share_entry=share_entry,
        entry_seed=entry_seed,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)

    for _, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()

    num_train_parameters = 0
    total_num_parameters = 0
    for n, p in model.named_parameters():
        total_num_parameters += p.numel()
        if p.requires_grad:
            print(n, p.shape, p.device, p.requires_grad)
            num_train_parameters += p.numel()

    print(f"trainable params: {num_train_parameters} || all params: {total_num_parameters} || trainable%: {100 * num_train_parameters / total_num_parameters:.2f}")

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if resume_from_checkpoint:
        checkpoint_name = None
        for fname in (
            "pytorch_model.bin",
            "adapter_model.safetensors",
            "adapter_model.bin",
        ):
            candidate = os.path.join(resume_from_checkpoint, fname)
            if os.path.exists(candidate):
                checkpoint_name = candidate
                break

        if checkpoint_name is None:
            print(f"Checkpoint {resume_from_checkpoint} has no adapter weights")
            resume_from_checkpoint = False
        else:
            print(f"Restarting from {checkpoint_name}")
            if checkpoint_name.endswith(".safetensors"):
                from safetensors.torch import load_file

                adapters_weights = load_file(checkpoint_name)
            else:
                adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)

    model.print_trainable_parameters()

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=seed
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    _grad_ckpt = os.environ.get("FRAMEFT_GRAD_CKPT", "") == "1"
    if _grad_ckpt:
        model.enable_input_require_grads()
        print("[frameft] gradient checkpointing enabled")

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_checkpointing=_grad_ckpt,
            gradient_checkpointing_kwargs={"use_reentrant": False} if _grad_ckpt else None,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=0.03,
            weight_decay=0.001,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    trainer.add_callback(SavePeftModelCallback)

    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    fire.Fire(train)
