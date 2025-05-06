"""
改进版T5模型文本摘要实现 (PyTorch版本)
包含更先进的微调技术和评估方法
使用Hugging Face datasets (例如: abisee/cnn_dailymail)
针对NVIDIA GPU (如4060 8GB) 优化
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter # For TensorBoard
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    get_linear_schedule_with_warmup,
    GenerationConfig
)
from rouge import Rouge # Note: consider using `evaluate.load('rouge')` for consistency with Hugging Face
import evaluate # For ROUGE metric from Hugging Face
import logging
import argparse
import random
from tqdm.auto import tqdm
import wandb # For Weights & Biases

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ROUGE评估指标 (Hugging Face `evaluate` 库版本)
hf_rouge_metric = evaluate.load('rouge')
# 老版本 rouge 库，用于 evaluate_samples，可以考虑统一
legacy_rouge_evaluator = Rouge()


def set_seed(seed_val=42):
    """设置随机种子以确保可复现性"""
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False # 重要：对于可复现性，禁用benchmark

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="改进的T5模型文本摘要 (PyTorch版本)")
    parser.add_argument("--model_name", type=str, default="google/flan-t5-small", help="预训练模型名称 (例如: google/flan-t5-small, google/flan-t5-base)")
    parser.add_argument("--dataset_name", type=str, default="abisee/cnn_dailymail", help="Hugging Face数据集名称")
    parser.add_argument("--dataset_config_name", type=str, default="2.0.0", help="Hugging Face数据集配置名称 (例如: cnn_dailymail 的 3.0.0 或 2.0.0)")
    parser.add_argument("--max_source_length", type=int, default=512, help="源文本最大长度")
    parser.add_argument("--max_target_length", type=int, default=128, help="目标摘要最大长度")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率") # Adjusted for PyTorch common practice
    parser.add_argument("--batch_size", type=int, default=4, help="训练批量大小 (请根据GPU显存调整，4060 8GB 对 small 模型可能需要 2-4)")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="评估批量大小")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb记录实验")
    parser.add_argument("--wandb_project", type=str, default="improved-t5-summarization-pytorch", help="WandB项目名称")
    parser.add_argument("--output_dir", type=str, default="./improved_t5_summary_pytorch", help="输出目录")
    parser.add_argument("--warmup_steps", type=int, default=500, help="预热步数")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减率")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--mixed_precision", action="store_true", default=True, help="是否使用混合精度训练 (推荐)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数 (用于模拟更大batch size)")
    parser.add_argument("--use_compile", action="store_true", help="是否使用 torch.compile (PyTorch 2.0+, 实验性)")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="早停耐心轮数 (基于验证集损失)")
    return parser.parse_args()

def prepare_dataset(tokenizer, args):
    """准备和处理数据集"""
    try:
        logger.info(f"加载数据集: {args.dataset_name} (配置: {args.dataset_config_name})")
        dataset = load_dataset(args.dataset_name, args.dataset_config_name)
        
        train_ds = dataset["train"]
        val_ds = dataset["validation"]
        # 如果测试集非常大，可以考虑只取一部分进行最后的评估
        test_ds = dataset["test"] #.select(range(1000)) # Example: select a subset for faster testing
        
        logger.info(f"加载数据成功 - 训练集: {len(train_ds)}个样本, 验证集: {len(val_ds)}个样本, 测试集: {len(test_ds)}个样本")
        
    except Exception as e:
        logger.error(f"加载数据集时出错: {e}")
        # Fallback to a dummy dataset for structure testing if needed
        # logger.warning("Falling back to a dummy dataset for testing purposes.")
        # dummy_data = {"article": ["This is a test article."] * 10, "highlights": ["Test summary."] * 10}
        # train_ds = Dataset.from_dict(dummy_data)
        # val_ds = Dataset.from_dict(dummy_data)
        # test_ds = Dataset.from_dict(dummy_data)
        raise

    def preprocess_function(examples):
        inputs = ["summarize: " + doc for doc in examples["article"]]
        model_inputs = tokenizer(
            inputs,
            max_length=args.max_source_length,
            padding="max_length", # Pad to max_length
            truncation=True,
            return_tensors="pt" # Return PyTorch tensors
        )
        
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["highlights"],
                max_length=args.max_target_length,
                padding="max_length", # Pad to max_length
                truncation=True,
                return_tensors="pt" # Return PyTorch tensors
            )
        
        # Replace pad_token_id in labels with -100 so it's ignored in loss calculation
        model_inputs["labels"] = labels["input_ids"].clone()
        model_inputs["labels"][model_inputs["labels"] == tokenizer.pad_token_id] = -100
        
        return model_inputs

    logger.info("预处理数据集...")
    tokenized_train_ds = train_ds.map(preprocess_function, batched=True, remove_columns=train_ds.column_names, desc="预处理训练集")
    tokenized_val_ds = val_ds.map(preprocess_function, batched=True, remove_columns=val_ds.column_names, desc="预处理验证集")
    tokenized_test_ds = test_ds.map(preprocess_function, batched=True, remove_columns=test_ds.column_names, desc="预处理测试集")
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=None, return_tensors="pt") # model is optional here

    train_dataloader = DataLoader(tokenized_train_ds, shuffle=True, collate_fn=data_collator, batch_size=args.batch_size, pin_memory=True, num_workers=os.cpu_count() // 2 if os.cpu_count() else 1)
    val_dataloader = DataLoader(tokenized_val_ds, collate_fn=data_collator, batch_size=args.eval_batch_size, pin_memory=True, num_workers=os.cpu_count() // 2 if os.cpu_count() else 1)
    test_dataloader = DataLoader(tokenized_test_ds, collate_fn=data_collator, batch_size=args.eval_batch_size, pin_memory=True, num_workers=os.cpu_count() // 2 if os.cpu_count() else 1)

    return {
        "train": train_dataloader,
        "val": val_dataloader,
        "test": test_dataloader,
        "raw_val": val_ds, # Keep raw for qualitative eval
        "raw_test": test_ds # Keep raw for qualitative eval
    }

def prepare_model_and_tokenizer(args, device):
    """准备模型和tokenizer"""
    logger.info(f"加载模型: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    model.to(device)

    if args.use_compile and hasattr(torch, "compile"):
        logger.info("使用 torch.compile() 优化模型 (实验性)...")
        try:
            model = torch.compile(model, mode="reduce-overhead") # "default" or "reduce-overhead" or "max-autotune"
        except Exception as e:
            logger.warning(f"torch.compile 失败: {e}. 继续而不编译.")
    
    # 设置生成参数
    generation_config = GenerationConfig.from_pretrained(
        args.model_name,
        max_length=args.max_target_length,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3,
        # length_penalty=2.0 # Example: Add length penalty
    )
    model.generation_config = generation_config
    
    return model, tokenizer

def compute_metrics_hf(eval_preds, tokenizer):
    """使用Hugging Face evaluate库计算ROUGE分数"""
    preds, labels = eval_preds
    
    # 将-100替换为pad_token_id进行解码
    labels[labels == -100] = tokenizer.pad_token_id
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # 去除首尾空格
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]
    
    # ROUGE expects a list of lists for references
    # decoded_labels = [[label] for label in decoded_labels]
    
    result = hf_rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    
    # 提取主要分数并乘以100
    result = {key: value * 100 for key, value in result.items()}
    
    # 计算生成长度
    prediction_lens = [len(pred.split()) for pred in decoded_preds]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}

def train_epoch(model, dataloader, optimizer, scheduler, device, scaler, args, current_epoch):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"训练 Epoch {current_epoch+1}/{args.epochs}", leave=False)

    for step, batch in enumerate(progress_bar):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        with autocast(enabled=args.mixed_precision):
            outputs = model(**batch)
            loss = outputs.loss
        
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer) # Unscale before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * args.gradient_accumulation_steps # Correct loss accumulation
        progress_bar.set_postfix({"loss": total_loss / (step + 1)})
        
        if args.use_wandb:
            wandb.log({"train_loss_step": loss.item() * args.gradient_accumulation_steps, "learning_rate": scheduler.get_last_lr()[0]})

    avg_train_loss = total_loss / len(dataloader)
    logger.info(f"Epoch {current_epoch+1} 平均训练损失: {avg_train_loss:.4f}")
    return avg_train_loss

def evaluate_model(model, dataloader, tokenizer, device, args, desc="评估"):
    model.eval()
    all_preds = []
    all_labels = []
    total_eval_loss = 0
    
    progress_bar = tqdm(dataloader, desc=desc, leave=False)
    with torch.no_grad():
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            with autocast(enabled=args.mixed_precision): # AMP for evaluation too
                outputs = model(**batch)
                loss = outputs.loss
            total_eval_loss += loss.item()

            # 生成摘要用于ROUGE评估
            generated_ids = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                generation_config=model.generation_config
            )
            
            # Pad shorter generated sequences to the max_target_length for consistent tensor shapes
            # This might not be strictly necessary if compute_metrics_hf handles varying lengths,
            # but it ensures all_preds can be stacked into a single tensor if needed.
            # For ROUGE, varying lengths are fine as long as batch_decode handles it.
            
            # Collect predictions and labels
            all_preds.extend(generated_ids.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    avg_eval_loss = total_eval_loss / len(dataloader)
    
    # Pad predictions if necessary for stacking (can be complex with dynamic lengths)
    # Instead, we'll decode them directly. compute_metrics_hf can handle lists of decoded strings.
    # No need to convert to a single tensor if the metric function supports lists.

    metrics = compute_metrics_hf((all_preds, all_labels), tokenizer)
    metrics["eval_loss"] = round(avg_eval_loss, 4)
    
    logger.info(f"{desc} 结果: {metrics}")
    return metrics

def main():
    args = parse_args()
    set_seed(args.seed)

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU名称: {torch.cuda.get_device_name(0)}")
        logger.info(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # WandB 初始化
    if args.use_wandb:
        wandb.init(project=args.wandb_project, config=vars(args), name=f"{args.model_name.split('/')[-1]}-{args.dataset_name.split('/')[-1]}")
    
    # TensorBoard Writer
    tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))

    # 准备模型和tokenizer
    model, tokenizer = prepare_model_and_tokenizer(args, device)

    # 准备数据集
    datasets = prepare_dataset(tokenizer, args)
    train_dataloader = datasets["train"]
    val_dataloader = datasets["val"]
    test_dataloader = datasets["test"]

    # 优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    num_training_steps = args.epochs * len(train_dataloader) // args.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps)
    
    # 混合精度 Scaler
    scaler = GradScaler(enabled=args.mixed_precision)

    # 训练循环
    best_eval_loss = float('inf')
    epochs_no_improve = 0
    os.makedirs(args.output_dir, exist_ok=True)

    if args.use_wandb:
        wandb.watch(model, log="all", log_freq=100) # Watch gradients

    for epoch in range(args.epochs):
        logger.info(f"===== Epoch {epoch+1}/{args.epochs} =====")
        avg_train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, device, scaler, args, epoch)
        eval_metrics = evaluate_model(model, val_dataloader, tokenizer, device, args, desc="验证集评估")
        
        # 记录到TensorBoard
        tb_writer.add_scalar("Loss/train", avg_train_loss, epoch)
        tb_writer.add_scalar("Loss/eval", eval_metrics["eval_loss"], epoch)
        for metric_name, value in eval_metrics.items():
            if metric_name != "eval_loss":
                 tb_writer.add_scalar(f"Eval/{metric_name}", value, epoch)
        
        # 记录到WandB
        if args.use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss_epoch": avg_train_loss,
                "eval_loss_epoch": eval_metrics["eval_loss"],
                **{f"eval_{k}": v for k, v in eval_metrics.items() if k != "eval_loss"}
            })

        # 保存最佳模型 (基于验证集损失)
        if eval_metrics["eval_loss"] < best_eval_loss:
            logger.info(f"验证集损失从 {best_eval_loss:.4f} 提高到 {eval_metrics['eval_loss']:.4f}. 保存模型...")
            best_eval_loss = eval_metrics["eval_loss"]
            model.save_pretrained(os.path.join(args.output_dir, "best_model"))
            tokenizer.save_pretrained(os.path.join(args.output_dir, "best_model"))
            epochs_no_improve = 0
            # 保存相关参数
            torch.save(args, os.path.join(args.output_dir, "best_model", "training_args.bin"))
        else:
            epochs_no_improve += 1
            logger.info(f"验证集损失未提高. 当前最佳: {best_eval_loss:.4f}. 无改善轮数: {epochs_no_improve}/{args.early_stopping_patience}")

        # 早停
        if epochs_no_improve >= args.early_stopping_patience:
            logger.info("触发早停.")
            break
            
        # 定性评估一些样本 (可选，可以在训练结束后进行)
        if (epoch + 1) % 2 == 0 or (epoch + 1) == args.epochs : # Evaluate samples every 2 epochs or last epoch
             evaluate_samples(model, tokenizer, datasets["raw_val"], device, args, num_samples=3, desc_prefix=f"Epoch_{epoch+1}_Val")


    # 训练结束
    logger.info("训练完成.")
    tb_writer.close()

    # 加载最佳模型进行最终评估
    logger.info("加载最佳模型进行最终测试集评估...")
    model = T5ForConditionalGeneration.from_pretrained(os.path.join(args.output_dir, "best_model")).to(device)
    if args.use_compile and hasattr(torch, "compile"): # Re-compile if loaded
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            logger.warning(f"torch.compile 失败: {e}")

    test_metrics = evaluate_model(model, test_dataloader, tokenizer, device, args, desc="测试集评估")
    logger.info(f"最终测试集结果: {test_metrics}")
    if args.use_wandb:
        wandb.log({"final_test_loss": test_metrics["eval_loss"],
                   **{f"final_test_{k}": v for k, v in test_metrics.items() if k != "eval_loss"}})
        
    # 最终定性评估
    evaluate_samples(model, tokenizer, datasets["raw_test"], device, args, num_samples=5, desc_prefix="Final_Test")

    # 保存最终模型 (即使早停，也保存一下最后的状态，或者依赖best_model)
    # model.save_pretrained(os.path.join(args.output_dir, "final_model"))
    # tokenizer.save_pretrained(os.path.join(args.output_dir, "final_model"))
    # logger.info(f"最终模型保存在 {os.path.join(args.output_dir, 'final_model')}")

    if args.use_wandb:
        wandb.finish()

def evaluate_samples(model, tokenizer, raw_dataset, device, args, num_samples=3, desc_prefix="Sample"):
    """评估一些样本并打印结果 (使用旧版Rouge)"""
    model.eval()
    if not raw_dataset:
        logger.warning("没有提供用于样本评估的原始数据集。")
        return

    # 从原始数据集中随机选择样本
    if len(raw_dataset) == 0:
        logger.warning("原始数据集为空，无法评估样本。")
        return
    
    actual_num_samples = min(num_samples, len(raw_dataset))
    if actual_num_samples == 0:
        logger.warning("没有足够的样本进行评估。")
        return

    sample_indices = random.sample(range(len(raw_dataset)), actual_num_samples)
    
    logger.info(f"===== {desc_prefix} 摘要样本 (共 {actual_num_samples} 个) =====")
    
    results_for_logging = []

    for i, idx in enumerate(sample_indices):
        item = raw_dataset[idx]
        article = item["article"]
        reference_summary = item["highlights"]

        inputs = tokenizer(
            "summarize: " + article,
            return_tensors="pt",
            max_length=args.max_source_length,
            truncation=True,
            padding="longest" # Pad to longest in batch if batching, or just this single sample
        ).to(device)

        with torch.no_grad():
            with autocast(enabled=args.mixed_precision):
                summary_ids = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    generation_config=model.generation_config
                )
        generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        try:
            scores = legacy_rouge_evaluator.get_scores(generated_summary, reference_summary)[0]
            rouge_1 = scores["rouge-1"]["f"]
            rouge_2 = scores["rouge-2"]["f"]
            rouge_l = scores["rouge-l"]["f"]
        except Exception as e:
            logger.warning(f"计算样本ROUGE分数时出错: {e}. 生成摘要: '{generated_summary}', 参考: '{reference_summary}'")
            rouge_1, rouge_2, rouge_l = 0,0,0


        logger.info(f"--- 样本 {i+1}/{actual_num_samples} ---")
        logger.info(f"原文 (前200字符): {article[:200]}...")
        logger.info(f"参考摘要: {reference_summary}")
        logger.info(f"生成摘要: {generated_summary}")
        logger.info(f"ROUGE-1: {rouge_1:.4f}, ROUGE-2: {rouge_2:.4f}, ROUGE-L: {rouge_l:.4f}")
        
        results_for_logging.append({
            "article_preview": article[:200]+"...",
            "reference": reference_summary,
            "generated": generated_summary,
            "rouge-1": rouge_1, "rouge-2": rouge_2, "rouge-l": rouge_l
        })

    if args.use_wandb and results_for_logging:
        try:
            # Log as a table in WandB
            samples_table = wandb.Table(columns=["Article Preview", "Reference", "Generated", "ROUGE-1", "ROUGE-2", "ROUGE-L"])
            for res in results_for_logging:
                samples_table.add_data(res["article_preview"], res["reference"], res["generated"], res["rouge-1"], res["rouge-2"], res["rouge-l"])
            wandb.log({f"{desc_prefix}_Summaries": samples_table})
        except Exception as e:
            logger.error(f"记录样本到WandB时出错: {e}")


if __name__ == "__main__":
    main() 