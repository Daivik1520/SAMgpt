# SAMgpt — Modular GPT Training & Inference (Educational + Practical)

<div align="center">

<img src="assets/nanogpt.jpg" alt="SAMgpt" width="100%" />

<p>
  <strong>Clean, extensible, and production-aware GPT training stack</strong><br/>
  Built by <b>DAIVIK REDDY</b> for learning, research, and real-world experimentation.
</p>

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
  <img alt="Transformers" src="https://img.shields.io/badge/Transformers-🤗-FFCA28?style=for-the-badge&logo=huggingface&logoColor=black">
  <img alt="MPS" src="https://img.shields.io/badge/Apple%20MPS-Optimized-000000?style=for-the-badge&logo=apple&logoColor=white">
  <img alt="CUDA" src="https://img.shields.io/badge/CUDA-Optional-76B900?style=for-the-badge&logo=nvidia&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge&logo=opensourceinitiative&logoColor=white">
</p>

<p>
  <a href="#-overview">Overview</a> •
  <a href="#-features">Features</a> •
  <a href="#-getting-started">Quick Start</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-training">Training</a> •
  <a href="#-finetuning">Finetuning</a> •
  <a href="#-inference">Inference</a> •
  <a href="#-benchmarks">Benchmarks</a> •
  <a href="#-troubleshooting">Troubleshooting</a>
</p>

</div>

---

## 🎯 Overview

SAMgpt is a concise yet powerful re-implementation of GPT training from scratch, with a strong emphasis on:
- Educational clarity (readable ~300-line `train.py` and ~300-line `model.py`)
- Practical experimentation (config-driven runs, MPS/CUDA support)
- Modularity (swap tokenizers, datasets, schedulers, optimizers easily)

Use it to learn Transformers, reproduce small GPT baselines, or bootstrap finetuning pipelines on consumer hardware (Mac MPS, single NVIDIA GPU, or CPU-only).

---

## ✨ Features

- Minimal but production-aware codebase (logging, configs, checkpoints)
- Character-level and BPE-token-level training flows
- Apple Silicon (MPS) acceleration out-of-the-box
- PyTorch 2.x `torch.compile` toggle for speed
- Reproducible configs for small GPT (Shakespeare) and GPT-2-style runs
- Sampling utilities for quick qualitative evaluation

---

## 🚀 Getting Started

### 1) Install
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # or: pip install torch numpy transformers tiktoken wandb tqdm
```

### 2) Quick sanity run (Shakespeare, char-level)
```bash
# Prepare data
python data/shakespeare_char/prepare.py

# Train (pick your device)
python train.py config/train_shakespeare_char.py --device=mps --compile=False   # Apple Silicon
python train.py config/train_shakespeare_char.py                                 # NVIDIA GPU
python train.py config/train_shakespeare_char.py --device=cpu --compile=False   # CPU only

# Sample text
python sample.py --out_dir=out-shakespeare-char --device=mps
```

Expected: ~10–12M params model, ~1.4–1.6 val loss in minutes on MPS/GPU.

---

## 🏗 Architecture

```
SAMgpt/
├── config/                 # Training configs (small GPT, GPT-2, finetune)
├── data/                   # Dataset scripts (shakespeare_char, openwebtext)
├── src/ (optional)         # If present, helpers/utilities
├── train.py                # Training loop (~300 LOC)
├── model.py                # GPT model (~300 LOC)
├── sample.py               # Inference/sampling
├── bench.py                # Micro-benchmarks
└── requirements.txt        # Python deps
```

Key ideas:
- `model.py`: Decoder-only Transformer with GELU, LayerNorm, causal mask
- `train.py`: Config parsing, data loader, optimizer/scheduler, compile toggle, checkpointing
- Configs define tokenization, model dims, dataset paths, batch sizes, iters, eval cadence

---

## 🧪 Training

### Shakespeare (character-level)
```bash
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char.py --device=mps --compile=False
```

Common flags:
```bash
--device=mps|cuda|cpu   # select accelerator
--compile=True|False    # PyTorch 2 compile
--max_iters=5000        # training steps
--block_size=256        # context length
--batch_size=64         # global batch
--n_layer=6 --n_head=6 --n_embd=384
--eval_interval=200 --eval_iters=200
```

### GPT-2 style (OpenWebText token-level)
```bash
python data/openwebtext/prepare.py
# single GPU
python train.py config/train_gpt2.py
# 8x GPUs
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

Hardware tips:
- Multi-node DDP: set master addr/port and node ranks
- If no Infiniband: `NCCL_IB_DISABLE=1`

---

## 🪄 Finetuning

Start from OpenAI GPT-2 checkpoints and adapt to your corpus:
```bash
python train.py config/finetune_shakespeare.py
```
Tweak `init_from`, learning rate, and `block_size`. Best checkpoint is saved under `out_dir`.

---

## 🗣 Inference (Sampling)

Sample from GPT-2 XL or your trained checkpoint:
```bash
python sample.py \
  --init_from=gpt2-xl \
  --start="What is the answer to life?" \
  --num_samples=5 --max_new_tokens=100

# or your own model
python sample.py --out_dir=out-shakespeare-char --start="ROMEO:" --max_new_tokens=200
```

---

## 📊 Benchmarks

- 10.65M params (6L/6H/384d), block_size 256 → ~1.47 val loss (Shakespeare-char) on MPS in ~8–10 min
- GPT-2 (124M) reproduction on OWT with 8×A100 → ~2.85 val loss after finetune

Performance levers:
- Lower `block_size`, increase `FRAME_SKIP` equivalent via eval cadence
- Use `--compile=True` on stable PyTorch 2.x for speedups
- Prefer MPS/CUDA over CPU; reduce model dims for CPU

---

## 🔧 Configuration

Environment variables (optional):
```bash
WANDB_PROJECT=samgpt
TOKENIZER=tiktoken           # or hf_tokenizer
MIXED_PRECISION=true         # if supported
SAVE_INTERVAL=500            # steps between checkpoints
```

Example minimal config (pseudo):
```python
config = dict(
  dataset="shakespeare_char",
  block_size=256,
  batch_size=64,
  n_layer=6, n_head=6, n_embd=384,
  max_iters=5000, lr=3e-4,
  device="mps", compile=False,
)
```

---

## 🐛 Troubleshooting

- Windows / PyTorch 2 compile issues → add `--compile=False`
- Slow CPU training → shrink model: `--n_layer 4 --n_head 4 --n_embd 128`, `--block_size 64`
- CUDA NCCL hangs → set `NCCL_IB_DISABLE=1` or check driver versions
- Tokenizer mismatch → ensure same tokenizer at train and sample time

---

## 🤝 Contributing

PRs welcome! Please:
- Keep functions focused and documented
- Add small tests if changing core logic
- Use conventional commits

---

## 📄 License

MIT — see `LICENSE`.
