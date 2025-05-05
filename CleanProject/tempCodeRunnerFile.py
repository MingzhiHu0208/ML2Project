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