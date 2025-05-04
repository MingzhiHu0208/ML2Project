#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
示例脚本：展示如何使用改进后的T5模型进行文本摘要 (TensorFlow版本)
基于自定义CSV数据集训练的模型
"""

import tensorflow as tf
import pandas as pd
from transformers import AutoTokenizer, TFT5ForConditionalGeneration, pipeline
import argparse
import logging
import random

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 确保TensorFlow使用合适的GPU内存增长方式
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print(f"找到 {len(physical_devices)} 个GPU设备并设置为动态内存分配")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="使用改进后的T5模型进行文本摘要示例 (TensorFlow版本)")
    parser.add_argument("--model_path", type=str, default="./improved_t5_summary_tf/final_model", 
                       help="微调后的模型路径")
    parser.add_argument("--use_gpu", action="store_true", help="是否使用GPU")
    parser.add_argument("--test_csv", type=str, default="summary_validation.csv", help="测试数据CSV文件路径")
    parser.add_argument("--num_samples", type=int, default=3, help="要测试的样本数量")
    return parser.parse_args()

def load_test_samples_from_csv(csv_path, num_samples=3):
    """从CSV文件加载测试样本"""
    try:
        df = pd.read_csv(csv_path)
        
        # 处理CSV数据
        samples = []
        
        # 遍历每一行处理数据
        for _, row in df.iterrows():
            # 合并原文列
            article = ' '.join([str(row['1']), str(row['2'])]).strip()
            # 摘要列
            reference = str(row['0']).strip()
            
            # 防止空字符串
            if article and reference and article != 'nan' and reference != 'nan':
                samples.append({
                    "article": article,
                    "reference": reference
                })
        
        # 随机抽取样本
        if len(samples) > num_samples:
            samples = random.sample(samples, num_samples)
            
        return samples
    
    except Exception as e:
        logger.error(f"加载CSV数据时出错: {e}")
        raise

def generate_summary(text, model_path, use_gpu):
    """使用模型生成文本摘要"""
    # 加载tokenizer和模型
    logger.info(f"加载模型：{model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 加载模型
    model = TFT5ForConditionalGeneration.from_pretrained(model_path)
    
    # 创建摘要pipeline
    summarizer = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        framework="tf",
        device=0 if use_gpu and len(tf.config.list_physical_devices('GPU')) > 0 else -1
    )
    
    # 添加提示词以帮助模型理解任务
    input_text = f"summarize: {text}"
    
    # 生成摘要
    logger.info("生成摘要...")
    summary = summarizer(
        input_text,
        max_length=128,
        min_length=30,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3
    )[0]["summary_text"]
    
    return summary

def main():
    """主函数"""
    args = parse_args()
    
    # 检查GPU可用性
    logger.info(f"TensorFlow 版本: {tf.__version__}")
    logger.info(f"GPU 可用性: {tf.config.list_physical_devices('GPU')}")
    
    # 从CSV加载测试样本
    logger.info(f"从CSV文件加载测试样本: {args.test_csv}")
    test_samples = load_test_samples_from_csv(args.test_csv, args.num_samples)
    
    # 对每个样本生成摘要
    for i, sample in enumerate(test_samples):
        article = sample["article"]
        reference = sample["reference"]
        
        # 生成摘要
        generated_summary = generate_summary(article, args.model_path, args.use_gpu)
        
        # 显示结果
        print(f"\n样本 {i+1}:")
        print("-" * 80)
        print(f"原文开头: {article[:300]}...")
        print("\n参考摘要:")
        print("-" * 80)
        print(reference)
        print("\n生成的摘要:")
        print("-" * 80)
        print(generated_summary)
        print("\n" + "=" * 100 + "\n")
    
    # 如果没有测试样本，显示一个固定示例
    if not test_samples:
        print("\n没有找到有效的测试样本，使用固定示例:")
        
        example_article = """
        The artificial intelligence company OpenAI has unveiled GPT-4o, its latest and most capable AI model that promises to blur the lines between text, audio, and visual understanding. At a livestreamed event on Monday, the Microsoft-backed company showcased the system's capabilities, which include high-quality voice interaction, near-instantaneous responses, and the ability to process and understand images and videos.
        """
        
        generated_summary = generate_summary(example_article, args.model_path, args.use_gpu)
        
        print("\n原始文章:")
        print("-" * 80)
        print(example_article)
        print("\n生成的摘要:")
        print("-" * 80)
        print(generated_summary)

if __name__ == "__main__":
    main() 