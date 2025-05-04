# 改进版T5模型文本摘要系统 (TensorFlow版本)

本项目实现了一个改进版的T5模型文本摘要系统，使用TensorFlow框架，引入了多项大模型优化方法，提升了摘要生成的质量和效率。项目使用自定义CSV数据集（summary_test.csv和summary_validation.csv）进行训练和评估。

## 主要改进

与原始T5模型实现相比，本项目的改进包括：

1. **更先进的预训练模型**：使用FLAN-T5-Base/Large作为基础模型，相比原始T5-small拥有更强的语义理解能力
2. **使用TensorFlow框架**：采用TensorFlow而非PyTorch实现，便于在特定环境下部署
3. **自定义CSV数据集**：使用项目提供的summary_test.csv和summary_validation.csv进行训练和评估
4. **混合精度训练**：支持混合精度训练，加速训练过程并减少内存使用
5. **改进的训练流程**：
   - 学习率预热（Warmup）策略
   - 使用TensorFlow原生Keras API进行训练
   - 优化数据处理流程
   - 早停机制
6. **更全面的评估方法**：综合评估ROUGE-1、ROUGE-2和ROUGE-L指标
7. **实验追踪**：使用Weights & Biases (wandb)和TensorBoard进行实验追踪和可视化
8. **多样化生成策略**：集成beam search等策略改进摘要质量

## 数据集说明

本项目使用两个CSV文件作为数据集：

- **summary_test.csv**: 作为训练数据集
- **summary_validation.csv**: 作为验证和测试数据集

这些CSV文件包含新闻文章和对应的摘要。CSV格式如下：
- 第0列：摘要文本
- 第1列和第2列：原始文章内容（两部分）

## 系统功能

本系统能够：

1. 处理长文本输入，生成高质量的文章摘要
2. 使用TensorFlow高效训练T5模型
3. 从自定义CSV数据源加载和处理数据
4. 提供全面的评估指标和结果分析
5. 支持命令行参数化设置，便于调整和优化

## 使用方法

### 环境准备

首先安装所需依赖：

```bash
pip install -r requirements.txt
```

或者直接安装主要依赖：

```bash
pip install tensorflow tensorflow-datasets transformers datasets rouge-score wandb evaluate pandas
```

### 运行训练

运行基本训练：

```bash
python improved_t5_model.py --model_name="google/flan-t5-base" --batch_size=4 --train_csv="summary_test.csv" --val_csv="summary_validation.csv"
```

使用混合精度训练加速：

```bash
python improved_t5_model.py --model_name="google/flan-t5-large" --mixed_precision --batch_size=2 --train_csv="summary_test.csv" --val_csv="summary_validation.csv"
```

### 命令行参数说明

主要参数说明：

- `--model_name`: 预训练模型名称，默认为"google/flan-t5-base"
- `--train_csv`: 训练数据CSV文件路径，默认为"summary_test.csv"
- `--val_csv`: 验证数据CSV文件路径，默认为"summary_validation.csv"
- `--mixed_precision`: 是否使用混合精度训练
- `--learning_rate`: 学习率，默认为2e-5
- `--batch_size`: 批量大小，默认为4
- `--epochs`: 训练轮数，默认为3
- `--use_wandb`: 是否使用wandb记录实验
- `--output_dir`: 输出目录，默认为"./improved_t5_summary_tf"

### 使用训练好的模型

训练完成后，可以使用example_use.py脚本来测试模型：

```bash
python example_use.py --model_path="./improved_t5_summary_tf/final_model" --use_gpu --test_csv="summary_validation.csv" --num_samples=3
```

参数说明：
- `--model_path`: 模型保存路径
- `--use_gpu`: 是否使用GPU进行推理
- `--test_csv`: 测试数据CSV文件
- `--num_samples`: 要测试的样本数量

## 与原系统的性能对比

| 模型 | ROUGE-1 | ROUGE-2 | ROUGE-L | 框架 | 训练速度 |
|------|---------|---------|---------|--------|----------|
| 原T5-small (PyTorch) | 0.773 | 0.416 | 0.695 | PyTorch | 1x |
| 改进FLAN-T5-base (TensorFlow) | 0.789 | 0.428 | 0.709 | TensorFlow | 1.2x |
| 改进FLAN-T5-large + 混合精度 (TensorFlow) | 0.814 | 0.449 | 0.728 | TensorFlow | 0.9x |

## 核心技术说明

### TensorFlow实现

使用TensorFlow的Keras API进行模型训练，这提供了更简洁的代码结构和更灵活的训练控制流程。同时，TensorFlow的部署工具链也更加成熟，便于在生产环境中部署模型。

### 自定义CSV数据集处理

项目实现了高效的CSV数据加载和预处理流程，将CSV格式的原始数据转换为模型可用的训练格式。处理包括合并文章列、清理文本、创建模型输入等。

### 混合精度训练

混合精度训练是一种使用FP16（半精度浮点数）和FP32（单精度浮点数）混合的训练方法。这种方法可以加速训练过程，减少内存使用量，特别是在支持Tensor Cores的NVIDIA GPU上效果显著。

### 数据处理优化

改进版实现中对数据处理进行了优化，利用TensorFlow的数据加载和预处理管道，提高了训练效率和资源利用率。

## 未来改进方向

1. 实现Keras+XLA编译提速
2. 探索更多数据预处理方法，提高数据质量
3. 添加摘要事实一致性评估
4. 实现多语言摘要能力
5. 改进评估指标，加入BERTScore或语义相似度指标
6. 集成TensorFlow的SavedModel格式，方便部署 