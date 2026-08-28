## FrameFT

This repository contains the code for **FrameFT**, a parameter-efficient fine-tuning (PEFT) method
for Transformer models. FrameFT represents the weight update `δW` in the basis of a **Tight Fusion
Frame (TFF)** and trains only a small set of **sparse coefficients**. Because the frame
basis is generated algorithmically and shared across all layers of the same dimension, only the
sparse coefficient matrices need to be stored — giving FrameFT a very small parameter and storage
footprint while matching or beating LoRA and full fine-tuning. Please find our paper,
[*Fine-Tuning of Transformer models with Frames*](https://openreview.net/pdf?id=KCfbSVIheg)
(ICML 2026), for theoretical analysis and algorithm details.

![FrameFT](figures/svd_vs_FF_FrameFT_merge.png)

## Installation

To install the environment, start from the official
PyTorch image and install the dependencies inside it.

```bash
# From the FrameFT directory, launch a container with the repo mounted
docker run --gpus all -it --ipc=host -v "$PWD:/workspace" -w /workspace \
  pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel bash
```

Then, inside the container, install the dependencies:

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install the FrameFT adapter (registers `frame` as a PEFT tuner)
bash install_peft.sh

# Install the lm-evaluation harness (for evaluating instruction tuned LLMs)
bash install_lm_eval.sh
```


## Running FrameFT

### GLUE benchmark

The GLUE experiments live under `GLUE/`. We provide ready-to-run launch scripts for the base and
large RoBERTa models across all tasks in `GLUE/scripts/`. For example, to fine-tune **RoBERTa-base**
on **CoLA**:

```bash
cd GLUE
bash scripts/RoBERTa-base-cola.sh
```

Use the corresponding script in
`GLUE/scripts/` to launch any other model/task combination, e.g. `RoBERTa-large-mrpc.sh`,
`RoBERTa-base-sst2.sh`, `RoBERTa-large-rte.sh`, etc. (models: `base`/`large`; tasks: `cola`, `mrpc`,
`qnli`, `rte`, `sst2`, `stsb`). Update `--output_dir` in the script to your preferred results path.

we use `wandb` for logging . Either run wandb login once inside the container, or disable logging with:


export WANDB_MODE=disabled

### Instruction Tuning experiments

The LLM Instruction Tuning experiments live under `InstructionTuning/`. The scripts for launching these experiments for different models are in `InstructionTuning/launch_scripts/`. For example, to fine-tune **Llama 2 7B** model on the `Alpaca` dataset, run:

```bash
cd InstructionTuning
bash launch_scripts/Llama-2-7b-frameft.sh
```

## Experimental Results

We compare FrameFT against full fine-tuning (FF), LoRA, and FourierFT. FrameFT matches or exceeds
these baselines on average while using far fewer trainable parameters.

<details open>
<summary>Fine-tuning RoBERTa Base and Large on the GLUE benchmark</summary>

**RoBERTa Base** (SST-2/QNLI/RTE: Acc., MRPC: Acc., CoLA: MCC, STS-B: PCC)

| Method | Params | SST-2 | MRPC | CoLA | QNLI | RTE | STS-B | Avg. |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| FF | 125M | 94.8 | 90.2 | 63.6 | **92.8** | 78.7 | 91.2 | 85.2 |
| LoRA | 0.3M | **95.1** | 89.7 | 63.4 | **93.3** | 78.4 | **91.5** | 85.2 |
| FourierFT | 24K | 94.2 | 90.0 | 63.8 | 92.2 | 79.1 | 90.8 | 85.0 |
| **FrameFT** | **24K** | 94.3 | **92.3** | **66.8** | 92.4 | 79.8 | 90.9 | **86.1** |

**RoBERTa Large**

| Method | Params | SST-2 | MRPC | CoLA | QNLI | RTE | STS-B | Avg. |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| FF | 356M | **96.4** | 90.9 | 68.0 | 94.7 | 86.6 | **92.4** | 88.2 |
| LoRA | 0.8M | 96.2 | 90.2 | 68.2 | **94.8** | 85.2 | 92.3 | 88.2 |
| FourierFT | 48K | 96.0 | 90.9 | 67.1 | 94.4 | 87.4 | 91.9 | 88.0 |
| **FrameFT** | **48K** | 96.2 | **92.6** | 69.8 | 93.4 | **88.1** | 91.9 | **88.7** |

</details>

<details open>
<summary>Instruction tuning (Llama-2-7B on Alpaca, evaluated on the LM-eval-harness)</summary>

| Method | Params | ARC-c | ARC-e | BoolQ | HellaSwag | OBQA | PIQA | RTE | WinoGrande | Avg. |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| FF | 6.7B | 47.52 | 77.73 | 78.96 | 58.99 | 33.6 | 78.61 | 62.09 | 69.61 | 63.39 |
| LoRA | 16.7M | 45.82 | 77.02 | 78.81 | 58.08 | 35.2 | 78.83 | 61.01 | 70.72 | 63.18 |
| FourierFT | 320K | 44.96 | 77.14 | 79.05 | 58.21 | 34.6 | 78.89 | 62.45 | 70.48 | 63.22 |
| **FrameFT** | **320K** | 45.22 | 76.93 | 78.62 | 58.08 | 34.2 | 78.62 | **66.06** | **71.19** | **63.62** |

</details>

<details open>
<summary>Fine-tuning Vision Transformers (ViT-L) on 8 image classification tasks</summary>

**ViT-L**

| Method | Params | Pets | Cars | CIFAR10 | DTD | EuroSAT | FGVC | RESISC45 | CIFAR100 | Avg. |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Full finetune | 303.3M | 94.43 | 88.90 | 99.15 | 81.79 | 99.04 | 68.25 | 96.43 | 93.58 | 90.20 |
| LoRA | 1.57M | 94.82 | 73.25 | **99.13** | 81.79 | 98.63 | 42.32 | 94.71 | **94.87** | 84.94 |
| FourierFT | 480K | 94.84 | 79.14 | 99.08 | **81.88** | 98.66 | 51.28 | **95.20** | 93.37 | 86.68 |
| **FrameFT** | **240K** | 94.20 | **82.83** | 98.90 | 81.54 | **98.87** | **59.63** | 94.96 | 92.71 | **87.95** |

</details>

## Cite

Please cite our work if you find it useful!

```bibtex
@InProceedings{adepuFrameFTIcml26,
  author    = {Harshavardhan Adepu and Li Zhang and Sanjiv Kumar and Vikas Singh},
  title     = {Fine-Tuning of Transformer models with Frames},
  booktitle = {Proceedings of the International Conference on Machine Learning (ICML)},
  year      = {2026},
  venue     = {ICML},
  month     = {July}
}
```
