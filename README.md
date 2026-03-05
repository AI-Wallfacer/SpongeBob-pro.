# SpongeBob-Pro: 中文 Causal LM 训练项目

从零构建 0.1B 参数中文语言模型完整训练框架，覆盖 Tokenizer → 预训练 → SFT 全链路。融合多个开源数据集构建 7B Token 预训练语料与 2M+ 条 SFT 对话数据，手写类 Qwen3 Dense 模型结构与全部训练代码，搭配两套评测体系验证模型能力，最终 C3 准确率从 0.25 提升至 0.38，XCOPA 提升至 0.55。

## 项目概述

SpongeBob-Pro 是一个轻量级的 Transformer 架构，采用 Grouped Query Attention (GQA) 和 RoPE 位置编码，支持高效的中文文本生成。本项目包含：

- **Pretrain（预训练）**: 在大规模中文语料上进行语言建模训练
- **SFT（监督微调）**: 基于对话数据进行指令对齐
- **GRPO（强化学习）**: 使用deepseek作为 Judge 模型进行奖励优化

## 模型架构

### 核心配置（0.1B 参数）

```python
hidden_size = 768              # 隐藏层维度
num_hidden_layers = 12         # Transformer 层数
num_attention_heads = 12       # 注意力头数
num_key_value_heads = 4        # KV 头数（GQA）
intermediate_size = 2048       # FFN 中间层维度
vocab_size = 15000             # 词表大小
max_position_embeddings = 32768  # 最大序列长度
```

### 技术特性

- ✅ Grouped Query Attention (GQA) - 减少 KV Cache 内存占用
- ✅ RoPE 位置编码 - 支持长序列外推
- ✅ Flash Attention - 加速训练和推理
- ✅ 混合精度训练 (bfloat16/float16)
- ✅ 多卡 DDP 训练支持
- ✅ 断点续训功能

## 项目结构

```
.
├── model/                      # 模型架构
│   ├── config.py              # SpongeBobConfig 配置类
│   └── model_spongebob_pro.py # 模型实现
├── dataset/                    # 数据集处理
│   ├── preprocess_data.py     # Pretrain 数据预处理
│   ├── pretrain_dataset.py    # Pretrain 数据加载器
│   └── sft_dataset.py         # SFT 数据加载器
├── train/                      # 训练脚本
│   ├── pretrain.py            # 预训练（DDP）
│   ├── train_sft.py           # SFT 训练（DDP）
│   └── train_grpo.py          # GRPO 训练（DDP）
├── tokenizer_15k/              # 15k 词表 tokenizer
├── benchmark/                  # 评测工具
├── eval.py                     # 交互式推理脚本
└── README.md
```

---

## 阶段一：Pretrain（预训练）

预训练阶段在大规模中文语料上进行无监督的语言建模训练，让模型学习基础的语言知识和语法结构。

![Pretrain Training Process](assets/pretrain.png)

训练过程中 loss 稳定下降，表明模型正在有效学习语言模式。

### 数据准备

**原始数据格式（JSONL）**:

```json
{"text": "这是一段训练文本..."}
{"text": "另一段训练文本..."}
```

**数据预处理**:

将原始文本转换为二进制格式以加速训练：

```bash
python dataset/preprocess_data.py \
  --input /path/to/pretrain.jsonl \
  --output /path/to/pretrain_data/pretrain_512 \
  --tokenizer ./tokenizer_15k \
  --seq_len 512
```

这会生成 `pretrain_512.bin` 和 `pretrain_512.meta` 两个文件。

### 训练配置

**单卡训练**:
```bash
python train/pretrain.py \
  --data_path /path/to/pretrain_data/pretrain_512 \
  --save_dir ./out_pretrain/exp_1 \
  --hidden_size 768 \
  --num_hidden_layers 12 \
  --max_seq_len 512 \
  --batch_size 128 \
  --learning_rate 1e-3 \
  --epochs 2 \
  --dtype bfloat16
```

**多卡训练（推荐）**:
```bash
torchrun --nproc_per_node=2 train/pretrain.py \
  --data_path /path/to/pretrain_data/pretrain_512 \
  --save_dir ./out_pretrain/exp_1 \
  --hidden_size 768 \
  --num_hidden_layers 12 \
  --max_seq_len 512 \
  --batch_size 128 \
  --learning_rate 1e-3 \
  --epochs 2 \
  --dtype bfloat16 \
  --use_swanlab 1
```

### 关键参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--learning_rate` | 初始学习率 | 1e-3 |
| `--batch_size` | 每卡批次大小 | 128 |
| `--accumulation_steps` | 梯度累积步数 | 1 |
| `--grad_clip` | 梯度裁剪阈值 | 1.0 |
| `--dtype` | 混合精度类型 | bfloat16 |
| `--save_interval` | 保存间隔（步数） | 3000 |
| `--eval_interval` | 评测间隔（步数） | 1000 |

### 训练特性

- **学习率调度**: 3% warmup + cosine decay
- **梯度累积**: 支持小显存训练
- **断点续训**: 自动保存 optimizer、scaler 状态
- **实时评测**: 支持 C3/XCOPA benchmark 评测
- **实验追踪**: 集成 SwanLab 可视化

### 输出文件

训练完成后会在 `save_dir` 下生成：

```
out_pretrain/exp_1/h768_l12_bs128_lr0.001/
├── global_step_3000/
│   ├── pretrain_768.pth      # 模型权重
│   └── resume.pth             # 断点文件
├── global_step_6000/
│   └── ...
```

![Pretrain Results](assets/pretrain_result.png)

---

## 阶段二：SFT（监督微调）

SFT 阶段使用对话数据对预训练模型进行微调，使其能够理解和遵循用户指令。

![SFT Training Process](assets/sft.png)

SFT 阶段基于 Pretrain 权重进行监督微调，使用对话数据训练模型理解和遵循用户指令。相比 Pretrain，SFT 使用更小的学习率（2e-5）进行精细调整。

### 数据格式

**JSONL 格式**:
```json
{
  "conversations": [
    {"role": "user", "content": "介绍一下你自己"},
    {"role": "assistant", "content": "我是张小凡，一个AI助手..."}
  ]
}
```

**多轮对话示例**:
```json
{
  "conversations": [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！有什么可以帮助你的吗？"},
    {"role": "user", "content": "介绍一下你自己"},
    {"role": "assistant", "content": "我是张小凡..."}
  ]
}
```

### 训练配置

**基于 Pretrain 权重微调**:
```bash
torchrun --nproc_per_node=2 train/train_sft.py \
  --data_path /path/to/sft_data.jsonl \
  --tokenizer_path ./tokenizer_15k \
  --from_weight ./pretrain_768.pth \
  --save_dir ./out_sft/exp_1 \
  --hidden_size 768 \
  --num_hidden_layers 12 \
  --max_seq_len 512 \
  --batch_size 128 \
  --learning_rate 2e-5 \
  --epochs 3 \
  --dtype bfloat16
```

**从头训练（不推荐）**:
```bash
torchrun --nproc_per_node=2 train/train_sft.py \
  --data_path /path/to/sft_data.jsonl \
  --tokenizer_path ./tokenizer_15k \
  --save_dir ./out_sft/exp_1 \
  --hidden_size 768 \
  --num_hidden_layers 12 \
  --batch_size 128 \
  --learning_rate 2e-5 \
  --epochs 3
```

### 关键参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--learning_rate` | 学习率（比 Pretrain 小） | 2e-5 |
| `--batch_size` | 每卡批次大小 | 128 |
| `--epochs` | 训练轮数 | 3 |
| `--from_weight` | Pretrain 权重路径 | ./pretrain_768.pth |
| `--tokenizer_path` | Tokenizer 路径 | ./tokenizer_15k |

### SFT 特性

- **Loss 计算**: 仅计算 assistant 部分的 loss，忽略 user 输入
- **对话格式**: 使用 special tokens 标记角色（`<|user|>`, `<|assistant|>`）
- **评测方式**: 支持 mini_bench 生成式评测 + DeepSeek Judge 打分
- **断点续训**: 与 Pretrain 相同的续训机制

### 输出文件

```
out_sft/exp_1/h768_l12_bs128_lr2e-05/
├── global_step_1000/
│   ├── sft_768.pth           # SFT 模型权重
│   └── resume.pth            # 断点文件
```

![SFT Results](assets/sft_result.png)

---

## 阶段三：GRPO（强化学习优化） 

#### 仅在claude交互下跑完了400+step 中途deepseek API还欠费了😅

GRPO 阶段使用强化学习方法，基于 DeepSeek Judge 的奖励信号进一步优化模型输出质量。

### 快速开始

```bash
torchrun --nproc_per_node=2 train/train_grpo.py \
  --data_path /path/to/grpo_prompts.jsonl \
  --tokenizer_path ./tokenizer_15k \
  --sft_model_path ./sft_768.pth \
  --judge_api_key $DEEPSEEK_API_KEY \
  --save_dir ./out_grpo/exp_1 \
  --batch_size 16 \
  --num_generations 4 \
  --learning_rate 5e-7
```

GRPO 是可选阶段，适合在 SFT 效果满意后进一步优化。详细配置请参考原 README。

---

## 实验结果展示

### GRPO 阶段成果

![GRPO Results 1](assets/grpo1.png)

![GRPO Results 2](assets/grpo2.png)

经过完整的训练流程后，模型能够进行流畅的中文对话，理解用户意图并给出合理的回复。上图展示了模型在不同场景下的表现。

---

## 快速开始

### 环境要求

- Python 3.10+
- CUDA 11.8+ / CUDA 12.x
- PyTorch 2.1+
- 推荐硬件: NVIDIA RTX 4090 / 5090

### 安装依赖

```bash
pip install torch transformers datasets tokenizers swanlab tqdm numpy
```

### 交互式推理

**SFT 模型推理**:
```bash
python eval.py \
  --model_path ./sft_768.pth \
  --tokenizer_path ./tokenizer_15k \
  --model_type sft \
  --multi_turn
```

**GRPO 模型推理**:
```bash
python eval.py \
  --model_path ./grpo_768.pth \
  --tokenizer_path ./tokenizer_15k \
  --model_type grpo \
  --multi_turn
```

### 断点续训

所有训练脚本都支持断点续训，只需添加 `--from_resume 1`:

```bash
torchrun --nproc_per_node=2 train/train_sft.py \
  --from_resume 1 \
  --data_path /path/to/sft.jsonl \
  --save_dir ./out_sft/exp_1 \
  # ... 其他参数保持不变
```

脚本会自动在 `save_dir` 下查找最新的 checkpoint 并恢复训练。

---

## 注意事项

### 安全建议

- ⚠️ 不要将 API Key 硬编码到代码中
- ✅ 使用环境变量管理敏感信息（`DEEPSEEK_API_KEY`、`SWANLAB_API_KEY`）
- ✅ 提交代码前检查 `.gitignore` 配置

### 性能优化

- 使用 `bfloat16` 混合精度（RTX 30/40/50 系列）
- 启用 Flash Attention（`flash_attn=True`）
- 多卡训练时调整 `batch_size` 和 `accumulation_steps`
- 使用 `torch.compile` 加速（PyTorch 2.0+）

### 已知限制

- 依赖固定的对话 special tokens
- GRPO 训练需要稳定的网络连接（调用 DeepSeek API）
- 部分默认参数针对特定硬件优化
