from args import *
import os
import time
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    FrameConfig,
    PeftType,
    FrameModel,
)
from datasets import load_dataset, load_metric
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from tqdm import tqdm
import csv

import wandb

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        if 'classifier' in name:
            print(name, param.numel(), param.requires_grad)
            continue
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )
    return trainable_params, all_param


args = get_args()
print(args)

torch.manual_seed(args.seed)
task = args.task
device = "cuda"
num_labels = 2
if task == "stsb":
    num_labels = 1

# setup wandb
wandb.init(
    project="FrameFT",
    name=args.exp_name,
    group=task,
    config=args.__dict__,
)

peft_type = PeftType.FRAME
peft_config = FrameConfig(task_type="SEQ_CLS", inference_mode=False, n_ff_coeffs = args.n_ff_coeffs, scale = args.scale, init_std=args.init_std, tff_l=args.tff_l, coeff_block_size=args.coeff_block_size, share_entry=args.share_entry, entry_seed=args.entry_seed)

def log(*pargs):
    log_dir = './logs_glue/' + task + '/' + args.model_name_or_path.split("-")[1]
    os.makedirs(log_dir, exist_ok=True)
    path_log = log_dir + '/bs' + str(args.bs) + 'maxlen' + str(args.max_length) + 'f_lr' + str(args.fft_lr)+ 'h_lr' + str(args.head_lr) + \
          'num' + str(args.n_ff_coeffs) + 'scale' + str(args.scale) + 'seed' + str(args.seed) + 'init_std' + str(args.init_std) + '.txt'
    print(path_log)
    with open(path_log, mode = 'a+') as w:
        w.write(" ".join(["{}".format(t) for t in pargs]))
        w.write("\n")

if any(k in args.model_name_or_path for k in ("gpt", "opt", "bloom")):
    padding_side = "left"
else:
    padding_side = "right"

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side=padding_side)
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

datasets = load_dataset("glue", task)
metric = load_metric("glue", task)

def tokenize_function(examples):
    # max_length=None => use the model max length (it's actually the default)
    if task == 'sst2' or task == 'cola':
        outputs = tokenizer(examples["sentence"], truncation=True, max_length=args.max_length)
    elif task == 'qnli':
        outputs = tokenizer(examples["question"], examples["sentence"], truncation=True, max_length=args.max_length)
    elif task == 'qqp':
        outputs = tokenizer(examples["question1"], examples["question2"], truncation=True, max_length=args.max_length)
    else:
        outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=args.max_length)
    return outputs

if task == 'sst2' or task == 'cola':
    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["idx", "sentence"],
    )
elif task == 'qnli':
    tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=["idx", "question", "sentence"],
    )
elif task == 'qqp':
    tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=["idx", "question1", "question2"],
    )
else:
    tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=["idx", "sentence1", "sentence2"],
    )

tokenized_datasets = tokenized_datasets.rename_column("label", "labels")


def collate_fn(examples):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")


# Instantiate dataloaders.
train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=args.bs)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], shuffle=False, collate_fn=collate_fn, batch_size=args.bs
)

model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path,num_labels=num_labels,return_dict=True)
model = get_peft_model(model, peft_config)

model.print_trainable_parameters()
print_trainable_parameters(model)
model

head_param = list(map(id, model.classifier.parameters()))

others_param = filter(lambda p: id(p) not in head_param, model.parameters()) 

optimizer = AdamW([
    {"params": model.classifier.parameters(), "lr": args.head_lr},
    {"params": others_param, "lr": args.fft_lr}
],weight_decay=args.weight_decay)

# Instantiate scheduler
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=args.warm_step * (len(train_dataloader) * args.num_epochs),
    num_training_steps=(len(train_dataloader) * args.num_epochs),
)

acc_list = []
model.to(device)
for epoch in range(args.num_epochs):
    
    model.train()
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch.to(device)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    model.eval()
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch.to(device)
        with torch.no_grad():
            outputs = model(**batch)
        if task == "stsb":
            predictions = outputs.logits
        else:
            predictions = outputs.logits.argmax(dim=-1)
        predictions, references = predictions, batch["labels"]
        # print(outputs.logits)
        metric.add_batch(
            predictions=predictions,
            references=references,
        )

    eval_metric = metric.compute()
    if task == "stsb":
        acc_list.append(eval_metric['pearson'])
        log(f"epoch {epoch}:", eval_metric, ', current_best_pearson:',max(acc_list),'train_loss:',loss)
        print(f"epoch {epoch}:", eval_metric, '\033[32m, current_best_pearson:\033[0m',max(acc_list),'train_loss:',loss)
    elif task == 'cola':
        acc_list.append(eval_metric['matthews_correlation'])
        print(f"epoch {epoch}:", eval_metric, '\033[32m, current_best_corr:\033[0m',max(acc_list),'train_loss:',loss)
        log(f"epoch {epoch}:", eval_metric, ', current_best_corr:',max(acc_list),'train_loss:',loss)
    else:
        acc_list.append(eval_metric['accuracy'])
        print(f"epoch {epoch}:", eval_metric, '\033[32m, current_best_acc:\033[0m',max(acc_list),'train_loss:',loss)
        log(f"epoch {epoch}:", eval_metric, ', current_best_acc:',max(acc_list),'train_loss:',loss)

    # wandb log 
    wandb.log({
        "epoch": epoch,
        f"{task}-n{args.n_ff_coeffs}_inst": acc_list[-1],
        f"{task}-n{args.n_ff_coeffs}_max": max(acc_list),
    })



results = [args.exp_name, max(acc_list)]
# write results to csv
with open('results.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(results)

wandb.log({
    f"{task}-final_acc": max(acc_list),
    "n_ff_coeffs": args.n_ff_coeffs,
    "task": task,
})

# save model
os.makedirs(args.output_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(args.output_dir, f"model_ckpt.pt"))
