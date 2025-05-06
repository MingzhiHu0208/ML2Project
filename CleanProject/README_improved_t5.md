# 改进版T5模型文本摘要系统 (PyTorch版本)

本项目实现了一个改进版的T5模型文本摘要系统，使用PyTorch框架，引入了多项大模型优化方法，提升了摘要生成的质量和效率。项目支持使用Hugging Face数据集（如CNN/DailyMail）进行训练和评估，特别优化了在NVIDIA GPU（如RTX 4060 8GB）上的运行性能。

## 主要改进

与原始T5模型实现相比，本项目的改进包括：

1. **更先进的预训练模型**：使用FLAN-T5系列作为基础模型，相比原始T5拥有更强的语义理解能力
2. **使用PyTorch框架**：采用PyTorch实现，提供更灵活的开发和优化选项
3. **高效数据处理**：使用Hugging Face datasets库高效加载和处理数据集
4. **混合精度训练**：支持自动混合精度训练，加速训练过程并减少显存使用
5. **针对NVIDIA GPU优化**：
   - 自动混合精度训练 (AMP)
   - 梯度累积，支持更大的有效批量大小
   - torch.compile 加速（PyTorch 2.0+）
   - 内存和显存优化策略
6. **改进的训练流程**：
   - 学习率预热（Warmup）策略
   - 梯度裁剪防止梯度爆炸
   - 早停机制避免过拟合
   - 更优雅的数据批处理
7. **更全面的评估方法**：使用多种ROUGE指标进行评估
8. **实验追踪**：使用Weights & Biases (wandb)和TensorBoard进行实验追踪和可视化
9. **多样化生成策略**：集成beam search和多样化摘要生成技术

## 代码结构和功能

本系统的核心函数和模块包括：

1. **set_seed(seed_val=42)** - 设置随机种子以确保可复现性
2. **parse_args()** - 解析命令行参数，提供丰富的配置选项
3. **prepare_dataset(tokenizer, args)** - 准备和处理数据集，支持Hugging Face数据源
4. **prepare_model_and_tokenizer(args, device)** - 准备模型和tokenizer，支持模型优化
5. **compute_metrics_hf(eval_preds, tokenizer)** - 使用Hugging Face evaluate库计算ROUGE分数
6. **train_epoch(model, dataloader, optimizer, scheduler, device, scaler, args, current_epoch)** - 训练单个epoch，支持混合精度
7. **evaluate_model(model, dataloader, tokenizer, device, args, desc="评估")** - 评估模型性能
8. **evaluate_samples(model, tokenizer, raw_dataset, device, args, num_samples=3, desc_prefix="Sample")** - 对样本进行定性评估
9. **main()** - 主函数，协调整个训练和评估流程

## 硬件要求和优化

本系统专为NVIDIA GPU优化，具体要求和建议：

- **推荐GPU**：NVIDIA RTX 4060或更高，至少8GB显存
- **针对显存限制的优化**：
  - 批量大小自动适应显存大小
  - 混合精度训练减少显存占用
  - 梯度累积支持更大的有效批量
  - 针对中小型模型(T5-small/T5-base)的特殊优化

## 使用方法

### 环境准备

首先安装所需依赖：

```bash
pip install -r requirements.txt
```

### 运行训练

基本训练示例：

```bash
python improved_t5_model.py --model_name="google/flan-t5-small" --batch_size=4 --dataset_name="abisee/cnn_dailymail" --dataset_config_name="2.0.0"
```

使用混合精度训练加速（默认已启用）：

```bash
python improved_t5_model.py --model_name="google/flan-t5-base" --batch_size=2 --mixed_precision --gradient_accumulation_steps=2
```

启用高级优化（PyTorch 2.0+）：

```bash
python improved_t5_model.py --model_name="google/flan-t5-small" --use_compile --mixed_precision
```

### 命令行参数说明

主要参数说明：

- `--model_name`: 预训练模型名称，默认为"google/flan-t5-small"
- `--dataset_name`: Hugging Face数据集名称，默认为"abisee/cnn_dailymail"
- `--dataset_config_name`: 数据集配置名称，默认为"2.0.0"
- `--mixed_precision`: 是否使用混合精度训练（默认启用）
- `--learning_rate`: 学习率，默认为5e-5
- `--batch_size`: 训练批量大小，默认为4（根据GPU显存调整）
- `--eval_batch_size`: 评估批量大小，默认为8
- `--epochs`: 训练轮数，默认为3
- `--use_wandb`: 是否使用wandb记录实验
- `--output_dir`: 输出目录，默认为"./improved_t5_summary_pytorch"
- `--gradient_accumulation_steps`: 梯度累积步数，默认为1
- `--use_compile`: 是否使用torch.compile优化（需要PyTorch 2.0+）
- `--early_stopping_patience`: 早停耐心轮数，默认为3

## GPU显存优化建议

根据不同型号的GPU，建议使用以下参数配置：

| GPU型号 | 显存 | 模型大小 | 推荐批量大小 | 梯度累积步数 | 其他优化 |
|---------|------|---------|------------|-------------|---------|
| RTX 4060 | 8GB | T5-small | 4-8 | 1 | 混合精度 |
| RTX 4060 | 8GB | T5-base | 2-4 | 2 | 混合精度 |
| RTX 4060 | 8GB | T5-large | 1 | 4-8 | 混合精度 |
| RTX 3090/4090 | 24GB+ | T5-xl | 2-4 | 1 | 混合精度 |

## 代码注意事项

使用本代码时需要注意以下几点：

1. **数据处理**：默认使用了Hugging Face datasets库，如需使用自定义CSV数据，需修改prepare_dataset函数
2. **显存管理**：对于小显存GPU，请适当减小batch_size或增加gradient_accumulation_steps
3. **torch.compile**：此功能需要PyTorch 2.0+，且在第一次运行时会有编译延迟
4. **模型大小选择**：
   - T5-small (~60M参数)：适合快速实验和小显存GPU
   - T5-base (~220M参数)：平衡性能和资源消耗
   - T5-large (~770M参数)：需要较大显存，适合最终性能优化
5. **警告处理**：可能会出现关于tokenizer和模型版本的警告，通常可以忽略

## 函数功能详解

### set_seed(seed_val=42)
设置随机种子确保实验可重复性，影响PyTorch、NumPy和Python内置随机模块，同时设置CUDA的确定性行为。

### prepare_dataset(tokenizer, args)
准备数据集和数据加载器，包含以下步骤：
- 从Hugging Face加载数据集
- 对输入文本添加前缀："summarize: "
- 分词并准备模型输入
- 创建高效的PyTorch DataLoader

### train_epoch(model, dataloader, optimizer, scheduler, device, scaler, args, current_epoch)
训练单个epoch，实现了：
- 混合精度训练
- 梯度累积
- 梯度裁剪
- 进度监控
- wandb集成（可选）

### evaluate_model(model, dataloader, tokenizer, device, args, desc="评估")
评估模型性能：
- 计算验证/测试损失
- 生成摘要并计算ROUGE分数
- 返回完整评估指标

### evaluate_samples(model, tokenizer, raw_dataset, device, args, num_samples, desc_prefix)
对随机样本进行定性评估，直观展示模型效果：
- 输出原文、参考摘要和生成摘要
- 计算单个样本的ROUGE分数
- 支持wandb可视化（可选）

## 未来改进方向

1. 支持更灵活的自定义数据集格式
2. 添加更多模型架构选项（如BART, Pegasus等）
3. 实现更多高级特性：
   - 知识蒸馏加速和压缩
   - 量化训练支持
   - DeepSpeed集成
4. 增强评估指标，如BERTScore或基于大模型的评估
5. 支持多语言摘要能力
6. 添加摘要事实一致性评估 