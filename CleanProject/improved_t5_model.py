"""
改进版T5模型文本摘要实现 (TensorFlow版本)
包含更先进的微调技术和评估方法
使用自定义CSV数据集（summary_test.csv和summary_validation.csv）
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer, 
    TFT5ForConditionalGeneration, 
    DataCollatorForSeq2Seq, 
    TFTrainer,
    TFTrainingArguments,
    pipeline,
    GenerationConfig,
    create_optimizer
)
from rouge import Rouge
import tensorflow_datasets as tfds
from tqdm.auto import tqdm
import wandb
from sklearn.model_selection import train_test_split
import logging
import argparse
from transformers.keras_callbacks import KerasMetricCallback
import evaluate

# 确保TensorFlow使用合适的GPU内存增长方式
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print(f"找到 {len(physical_devices)} 个GPU设备并设置为动态内存分配")

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ROUGE评估指标
rouge = Rouge()

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="改进的T5模型文本摘要 (TensorFlow版本)")
    parser.add_argument("--model_name", type=str, default="google/flan-t5-base", help="预训练模型名称")
    parser.add_argument("--max_source_length", type=int, default=1024, help="源文本最大长度")
    parser.add_argument("--max_target_length", type=int, default=128, help="目标摘要最大长度")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="学习率")
    parser.add_argument("--batch_size", type=int, default=4, help="批量大小")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb记录实验")
    parser.add_argument("--output_dir", type=str, default="./improved_t5_summary_tf", help="输出目录")
    parser.add_argument("--save_steps", type=int, default=500, help="模型保存步数")
    parser.add_argument("--eval_steps", type=int, default=500, help="评估步数")
    parser.add_argument("--warmup_steps", type=int, default=200, help="预热步数")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减率")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--mixed_precision", action="store_true", help="是否使用混合精度训练")
    return parser.parse_args()

def load_csv_dataset(csv_file_path, seed=42):
    """
    从CSV文件加载数据集
    CSV格式应该包含原文和摘要
    """
    logger.info(f"从CSV文件加载数据: {csv_file_path}")
    
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_file_path)
        
        # 处理CSV数据，假设CSV文件有3列：索引列、摘要列、原文列1、原文列2
        # 根据实际CSV格式调整以下内容
        articles = []
        highlights = []
        
        # 遍历每一行处理数据
        for _, row in df.iterrows():
            # 合并原文列
            article = ' '.join([str(row['1']), str(row['2'])]).strip()
            # 摘要列
            highlight = str(row['0']).strip()
            
            # 防止空字符串
            if article and highlight and article != 'nan' and highlight != 'nan':
                articles.append(article)
                highlights.append(highlight)
        
        # 创建HuggingFace数据集
        data_dict = {
            "article": articles,
            "highlights": highlights
        }
        
        dataset = Dataset.from_dict(data_dict)
        dataset = dataset.shuffle(seed=seed)
        
        return dataset
    
    except Exception as e:
        logger.error(f"加载CSV数据时出错: {e}")
        raise

def prepare_dataset(tokenizer, args):
    """准备和处理数据集"""
    
    try:
        logger.info("加载CNN/DailyMail数据集...")
        # 加载数据集
        dataset = load_dataset("abisee/cnn_dailymail", "2.0.0")
        
        # 获取训练集、验证集和测试集
        train_ds = dataset["train"]
        val_ds = dataset["validation"]
        test_ds = dataset["test"]
        
        logger.info(f"加载数据成功 - 训练集: {len(train_ds)}个样本, 验证集: {len(val_ds)}个样本, 测试集: {len(test_ds)}个样本")
        
    except Exception as e:
        logger.error(f"加载数据集时出错: {e}")
        raise
    
    # 数据预处理函数
    def preprocess_function(examples):
        # 添加特定前缀，帮助模型理解任务
        inputs = ["summarize: " + doc for doc in examples["article"]]
        model_inputs = tokenizer(
            inputs, 
            max_length=args.max_source_length, 
            padding="max_length", 
            truncation=True
        )
        
        # 设置目标文本（使用highlights作为摘要）
        labels = tokenizer(
            text_target=examples["highlights"], 
            max_length=args.max_target_length, 
            padding="max_length", 
            truncation=True
        )
        
        model_inputs["labels"] = labels["input_ids"]
        
        # 将-100作为填充标记的标签
        for i in range(len(labels["input_ids"])):
            model_inputs["labels"][i] = [
                -100 if token == tokenizer.pad_token_id else token 
                for token in model_inputs["labels"][i]
            ]
            
        return model_inputs
    
    # 应用预处理
    logger.info("预处理训练集...")
    tokenized_train_ds = train_ds.map(
        preprocess_function,
        batched=True,
        remove_columns=train_ds.column_names,
        desc="预处理训练集"
    )
    
    logger.info("预处理验证集...")
    tokenized_val_ds = val_ds.map(
        preprocess_function,
        batched=True,
        remove_columns=val_ds.column_names,
        desc="预处理验证集"
    )
    
    logger.info("预处理测试集...")
    tokenized_test_ds = test_ds.map(
        preprocess_function,
        batched=True,
        remove_columns=test_ds.column_names,
        desc="预处理测试集"
    )
    
    # 转换为TensorFlow数据集
    def convert_to_tf_dataset(dataset, batch_size):
        return dataset.to_tf_dataset(
            columns=["input_ids", "attention_mask", "labels"],
            shuffle=True,
            batch_size=batch_size,
            collate_fn=DataCollatorForSeq2Seq(
                tokenizer,
                padding="longest",
                return_tensors="tf"
            )
        )
    
    tf_train_dataset = convert_to_tf_dataset(tokenized_train_ds, args.batch_size)
    tf_eval_dataset = convert_to_tf_dataset(tokenized_val_ds, args.batch_size)
    tf_test_dataset = convert_to_tf_dataset(tokenized_test_ds, args.batch_size)
    
    return {
        "train": tf_train_dataset,
        "val": tf_eval_dataset,
        "test": tf_test_dataset,
        "raw_train": train_ds,
        "raw_val": val_ds,
        "raw_test": test_ds,
        "tokenized_train": tokenized_train_ds,
        "tokenized_val": tokenized_val_ds,
        "tokenized_test": tokenized_test_ds
    }

def compute_metrics(eval_preds, tokenizer):
    """计算评估指标"""
    rouge_metric = evaluate.load('rouge')
    
    preds, labels = eval_preds
    # 解码预测结果
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    # 替换-100为tokenizer.pad_token_id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # 一些简单的后处理
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]
    
    # 计算ROUGE分数
    result = rouge_metric.compute(
        predictions=decoded_preds, 
        references=decoded_labels, 
        use_stemmer=True
    )
    
    # 提取中值分数
    result = {k: round(v * 100, 4) for k, v in result.items()}
    
    # 添加平均生成长度
    prediction_lens = [len(pred.split()) for pred in decoded_preds]
    result["gen_len"] = np.mean(prediction_lens)
    
    return result

def prepare_model(args):
    """准备模型和tokenizer"""
    logger.info(f"加载模型: {args.model_name}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True
    )
    
    # 确保tokenizer有pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 设置混合精度策略
    if args.mixed_precision:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        logger.info("启用混合精度训练")
    
    # 加载模型
    model = TFT5ForConditionalGeneration.from_pretrained(
        args.model_name
    )
    
    # 为生成设置配置
    generation_config = GenerationConfig.from_pretrained(
        args.model_name,
        max_length=args.max_target_length,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3
    )
    model.generation_config = generation_config
    
    return model, tokenizer

def train_model(model, tokenizer, datasets, args):
    """训练模型"""
    logger.info("开始训练模型...")
    
    # 计算训练步数来设置学习率调度
    num_train_examples = len(datasets["tokenized_train"])
    train_steps_per_epoch = num_train_examples // args.batch_size
    total_train_steps = train_steps_per_epoch * args.epochs
    
    # 创建优化器
    optimizer, schedule = create_optimizer(
        init_lr=args.learning_rate,
        num_train_steps=total_train_steps,
        num_warmup_steps=args.warmup_steps,
        weight_decay_rate=args.weight_decay
    )
    
    # 初始化wandb
    if args.use_wandb:
        wandb.init(
            project="improved-t5-summarization-tf",
            name=f"{args.model_name.split('/')[-1]}_{args.learning_rate}",
            config=vars(args)
        )
        wandb_callback = wandb.keras.WandbCallback(
            monitor="val_loss",
            save_model=False
        )
    
    # 编译模型
    model.compile(optimizer=optimizer)
    
    # 设置评估回调
    rouge_metric = evaluate.load('rouge')
    
    def compute_metrics_for_keras(eval_pred):
        predictions, labels = eval_pred
        # 生成摘要
        predictions = model.generate(
            input_ids=predictions["input_ids"],
            attention_mask=predictions["attention_mask"],
            max_length=args.max_target_length,
            num_beams=4
        )
        
        # 解码预测结果
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # 替换-100为tokenizer.pad_token_id
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # 后处理
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        
        # 计算ROUGE分数
        result = rouge_metric.compute(
            predictions=decoded_preds, 
            references=decoded_labels, 
            use_stemmer=True
        )
        
        # 提取中值分数
        result = {k: round(v * 100, 4) for k, v in result.items()}
        return result
    
    metric_callback = KerasMetricCallback(
        compute_metrics_for_keras,
        eval_dataset=datasets["val"],
        batch_size=args.batch_size,
        save_best_model=True
    )
    
    # 设置模型检查点
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(args.output_dir, "checkpoint-{epoch}"),
        save_weights_only=True,
        save_freq="epoch",
        monitor="val_loss",
        save_best_only=True
    )
    
    # 早停回调
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )
    
    # 准备回调列表
    callbacks = [
        checkpoint_callback,
        metric_callback,
        early_stopping,
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(args.output_dir, "logs"),
            update_freq="epoch"
        )
    ]
    
    if args.use_wandb:
        callbacks.append(wandb_callback)
    
    # 训练模型
    history = model.fit(
        datasets["train"],
        validation_data=datasets["val"],
        epochs=args.epochs,
        callbacks=callbacks
    )
    
    # 保存最终模型
    model.save_pretrained(f"{args.output_dir}/final_model")
    tokenizer.save_pretrained(f"{args.output_dir}/final_model")
    
    # 评估模型
    logger.info("在测试集上评估模型...")
    results = model.evaluate(datasets["test"])
    logger.info(f"测试损失: {results}")
    
    # 关闭wandb
    if args.use_wandb:
        wandb.finish()
    
    return model, tokenizer, history

def evaluate_samples(model, tokenizer, datasets, args):
    """评估一些样本"""
    logger.info("生成样本摘要...")
    
    # 创建摘要生成管道
    summarizer = pipeline(
        "summarization", 
        model=model, 
        tokenizer=tokenizer, 
        framework="tf"
    )
    
    # 随机选择样本
    sample_indices = np.random.choice(len(datasets["raw_test"]), min(5, len(datasets["raw_test"])), replace=False)
    
    results = []
    for idx in sample_indices:
        article = datasets["raw_test"][idx]["article"]
        reference = datasets["raw_test"][idx]["highlights"]
        
        # 生成摘要
        input_text = f"summarize: {article}"
        summary = summarizer(
            input_text, 
            max_length=args.max_target_length, 
            min_length=30, 
            num_beams=4
        )[0]["summary_text"]
        
        # 计算ROUGE分数
        scores = rouge.get_scores(summary, reference)[0]
        
        results.append({
            "article": article[:200] + "...",  # 只显示文章开头
            "reference": reference,
            "generated": summary,
            "rouge-1": scores["rouge-1"]["f"],
            "rouge-2": scores["rouge-2"]["f"],
            "rouge-l": scores["rouge-l"]["f"]
        })
    
    # 显示结果
    for i, result in enumerate(results):
        logger.info(f"\n样本 {i+1}:")
        logger.info(f"文章开头: {result['article']}")
        logger.info(f"参考摘要: {result['reference']}")
        logger.info(f"生成摘要: {result['generated']}")
        logger.info(f"ROUGE-1: {result['rouge-1']:.4f}")
        logger.info(f"ROUGE-2: {result['rouge-2']:.4f}")
        logger.info(f"ROUGE-L: {result['rouge-l']:.4f}")
        logger.info("-" * 80)
    
    # 计算平均ROUGE分数
    avg_rouge1 = np.mean([r["rouge-1"] for r in results])
    avg_rouge2 = np.mean([r["rouge-2"] for r in results])
    avg_rougeL = np.mean([r["rouge-l"] for r in results])
    
    logger.info(f"样本平均ROUGE-1: {avg_rouge1:.4f}")
    logger.info(f"样本平均ROUGE-2: {avg_rouge2:.4f}")
    logger.info(f"样本平均ROUGE-L: {avg_rougeL:.4f}")
    
    return results

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 检查GPU可用性
    logger.info(f"TensorFlow 版本: {tf.__version__}")
    logger.info(f"GPU 可用性: {tf.config.list_physical_devices('GPU')}")
    
    # 准备模型和tokenizer
    model, tokenizer = prepare_model(args)
    
    # 准备数据集
    datasets = prepare_dataset(tokenizer, args)
    
    # 训练和评估模型
    model, tokenizer, history = train_model(model, tokenizer, datasets, args)
    
    # 评估样本
    sample_results = evaluate_samples(model, tokenizer, datasets, args)
    
    logger.info("训练和评估完成！")

if __name__ == "__main__":
    main() 