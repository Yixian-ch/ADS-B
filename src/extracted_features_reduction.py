import os
import gc
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import argparse
import json

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA

# 设置日志
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GroupedReduction")

def classify_features_intelligent(feature_names):
    """
    根据特征名称智能分类为瞬态和稳态特征
    """
    # 瞬态特征关键词 - 与信号变化、动态特性相关
    transient_keywords = [
        'variation', 'std', 'max', 'min', 'range', 'dynamic',
        'detail', 'peak', 'envelope_variation', 'phase_std',
        'freq_std', 'freq_max', 'freq_min', 'freq_range',
        'modulation_index', 'spectral_std', 'spectral_max',
        'wavelet_detail', 'euclidean_dist', 'dtw_dist',
        'spectral_skewness', 'spectral_kurtosis', 'spectral_crest'
    ]
    
    # 稳态特征关键词 - 与平均值、稳定状态相关
    steady_keywords = [
        'mean', 'approx', 'correlation', 'energy', 'entropy',
        'envelope_mean', 'freq_mean', 'phase_mean', 'spectral_mean',
        'wavelet_approx', 'bandwidth', 'odd_even_ratio',
        'mean_freq_offset', 'envelope_spectral', 'freq_spectral'
    ]
    
    transient_features = []
    steady_features = []
    
    for feature in feature_names:
        feature_lower = feature.lower()
        
        # 检查是否包含瞬态关键词
        is_transient = any(keyword in feature_lower for keyword in transient_keywords)
        
        # 检查是否包含稳态关键词
        is_steady = any(keyword in feature_lower for keyword in steady_keywords)
        
        if is_transient and not is_steady:
            transient_features.append(feature)
        elif is_steady and not is_transient:
            steady_features.append(feature)
        elif is_transient and is_steady:
            # 如果同时包含两种关键词，优先考虑更明显的特征
            if any(kw in feature_lower for kw in ['std', 'variation', 'max', 'min', 'range']):
                transient_features.append(feature)
            else:
                steady_features.append(feature)
        else:
            # 默认分为稳态（保守策略）
            steady_features.append(feature)
    
    logger.info(f"瞬态特征数量: {len(transient_features)}")
    logger.info(f"稳态特征数量: {len(steady_features)}")
    logger.info(f"瞬态特征示例: {transient_features[:5]}")
    logger.info(f"稳态特征示例: {steady_features[:5]}")
    
    return transient_features, steady_features

def integrate_features_chunked(enhanced_data_path, chunk_size=1000, output_file=None):
    """整合特征数据"""
    logger.info(f"开始整合特征数据")
    
    routes = ['I', 'Q']
    if output_file is None:
        output_file = os.path.join(os.path.dirname(enhanced_data_path), "integrated_features.csv")
    
    all_feature_files = []
    
    # 收集所有特征文件
    for route in routes:
        route_path = os.path.join(enhanced_data_path, f"{route}_data")
        if not os.path.exists(route_path):
            continue
        
        individual_dirs = [d for d in os.listdir(route_path) 
                          if os.path.isdir(os.path.join(route_path, d))]
        
        for individual_id in individual_dirs:
            features_file = os.path.join(route_path, individual_id, "enhanced_features.csv")
            if os.path.exists(features_file):
                all_feature_files.append((features_file, individual_id, route))
    
    logger.info(f"找到 {len(all_feature_files)} 个特征文件")
    
    # 整合数据
    first_chunk = True
    total_samples = 0
    
    for i in tqdm(range(0, len(all_feature_files), chunk_size), desc="整合特征"):
        chunk_files = all_feature_files[i:i+chunk_size]
        chunk_dfs = []
        
        for file_path, individual_id, route in chunk_files:
            try:
                df = pd.read_csv(file_path)
                chunk_dfs.append(df)
            except Exception as e:
                logger.warning(f"读取失败: {file_path}")
        
        if chunk_dfs:
            chunk_df = pd.concat(chunk_dfs, ignore_index=True)
            
            if first_chunk:
                chunk_df.to_csv(output_file, index=False, mode='w')
                first_chunk = False
            else:
                chunk_df.to_csv(output_file, index=False, mode='a', header=False)
            
            total_samples += len(chunk_df)
            del chunk_df, chunk_dfs
            gc.collect()
    
    logger.info(f"整合完成: {total_samples} 个样本")
    return total_samples, output_file

def preprocess_features_grouped(input_file, chunk_size=5000):
    """预处理并分组特征"""
    logger.info("开始预处理和分组特征")
    
    # 读取数据获取特征信息
    first_chunk = pd.read_csv(input_file, nrows=1000)
    
    # 识别数值特征（排除ID和元数据列）
    exclude_cols = ['individual_id', 'signal_idx', 'route']
    all_feature_cols = [col for col in first_chunk.columns 
                       if col not in exclude_cols and 
                       first_chunk[col].dtype in ['int64', 'float64', 'int32', 'float32']]
    
    logger.info(f"总特征数: {len(all_feature_cols)}")
    
    # 智能分组特征
    transient_features, steady_features = classify_features_intelligent(all_feature_cols)
    
    # 🔧 修复目标编码 - 一次性收集所有信息
    logger.info("收集数据信息...")
    all_individual_ids = set()
    total_samples = 0
    
    # 一次性读取所有数据来收集信息（对于小文件更可靠）
    try:
        # 先尝试读取整个文件
        full_df = pd.read_csv(input_file)
        total_samples = len(full_df)
        
        if 'individual_id' in full_df.columns:
            # 清理individual_id数据
            valid_ids = full_df['individual_id'].dropna().astype(str).str.strip()
            valid_ids = valid_ids[valid_ids != '']  # 去除空字符串
            all_individual_ids = set(valid_ids)
            
            logger.info(f"成功读取整个文件: {total_samples} 行")
        else:
            logger.error("缺少individual_id列")
            return None
            
    except Exception as e:
        logger.warning(f"无法读取整个文件 ({e})，改用分块读取...")
        
        # 如果文件太大，退回到分块读取
        chunk_iter = pd.read_csv(input_file, chunksize=chunk_size)
        for chunk in tqdm(chunk_iter, desc="收集目标标签"):
            if 'individual_id' in chunk.columns:
                valid_ids = chunk['individual_id'].dropna().astype(str).str.strip()
                valid_ids = valid_ids[valid_ids != '']
                all_individual_ids.update(valid_ids)
                total_samples += len(chunk)
            else:
                logger.error("缺少individual_id列")
                return None
    
    # 创建完整的目标编码映射
    unique_individual_ids = sorted(list(all_individual_ids))
    target_mapping = {individual_id: i for i, individual_id in enumerate(unique_individual_ids)}
    
    logger.info(f"总样本数: {total_samples}")
    logger.info(f"唯一个体数: {len(unique_individual_ids)}")
    logger.info(f"个体ID列表: {unique_individual_ids}")
    
    # 验证映射
    logger.info("目标映射验证:")
    for individual_id, target_id in list(target_mapping.items())[:10]:
        logger.info(f"  {individual_id} -> {target_id}")
    
    # 创建标准化器
    transient_scaler = StandardScaler()
    steady_scaler = StandardScaler()
    
    # 拟合标准化器
    logger.info("拟合标准化器...")
    chunk_iter = pd.read_csv(input_file, chunksize=chunk_size)
    for chunk in tqdm(chunk_iter, desc="拟合标准化器"):
        if transient_features:
            transient_data = chunk[transient_features].fillna(0)
            transient_scaler.partial_fit(transient_data)
        
        if steady_features:
            steady_data = chunk[steady_features].fillna(0)
            steady_scaler.partial_fit(steady_data)
    
    # 应用预处理
    processed_file = input_file.replace('.csv', '_processed_grouped.csv')
    first_chunk = True
    processed_samples = 0
    skipped_samples = 0
    
    logger.info("应用预处理...")
    chunk_iter = pd.read_csv(input_file, chunksize=chunk_size)
    for chunk in tqdm(chunk_iter, desc="预处理数据"):
        # 🔧 检查并处理缺失的individual_id
        if 'individual_id' not in chunk.columns:
            logger.error("chunk中缺少individual_id列")
            continue
            
        # 过滤掉individual_id为空的行
        valid_mask = chunk['individual_id'].notna()
        valid_chunk = chunk[valid_mask]
        
        if len(valid_chunk) == 0:
            logger.warning("当前chunk没有有效的individual_id")
            continue
        
        # 🔧 确保所有individual_id都在映射中
        missing_ids = set(valid_chunk['individual_id'].unique()) - set(target_mapping.keys())
        if missing_ids:
            logger.warning(f"发现缺失的individual_id: {missing_ids}")
            # 为缺失的ID创建新的映射
            for missing_id in missing_ids:
                target_mapping[missing_id] = len(target_mapping)
        
        # 标准化瞬态特征
        if transient_features:
            transient_scaled = transient_scaler.transform(valid_chunk[transient_features].fillna(0))
            transient_df = pd.DataFrame(transient_scaled, 
                                      columns=[f"transient_{col}" for col in transient_features])
        else:
            transient_df = pd.DataFrame()
        
        # 标准化稳态特征
        if steady_features:
            steady_scaled = steady_scaler.transform(valid_chunk[steady_features].fillna(0))
            steady_df = pd.DataFrame(steady_scaled,
                                   columns=[f"steady_{col}" for col in steady_features])
        else:
            steady_df = pd.DataFrame()
        
        # 合并特征和元数据
        result_df = pd.concat([transient_df, steady_df], axis=1)
        
        # 🔧 安全地处理目标编码
        try:
            result_df['target'] = [target_mapping[t] for t in valid_chunk['individual_id']]
            result_df['individual_id'] = valid_chunk['individual_id'].values
            result_df['route'] = valid_chunk['route'].values
            
            processed_samples += len(result_df)
            
        except KeyError as e:
            logger.error(f"目标编码错误: {e}")
            skipped_samples += len(valid_chunk)
            continue
        
        if first_chunk:
            result_df.to_csv(processed_file, index=False, mode='w')
            first_chunk = False
        else:
            result_df.to_csv(processed_file, index=False, mode='a', header=False)
        
        del result_df, transient_df, steady_df, valid_chunk
        gc.collect()
    
    logger.info(f"预处理完成: 处理了{processed_samples}个样本，跳过了{skipped_samples}个样本")
    logger.info(f"最终目标映射包含{len(target_mapping)}个个体")
    
    return (transient_scaler, steady_scaler, transient_features, steady_features, 
            target_mapping, processed_file)

def apply_grouped_pca(input_file, transient_components=50, steady_components=50, chunk_size=5000):
    """对瞬态和稳态特征分别进行PCA降维"""
    logger.info(f"开始分组PCA降维: 瞬态{transient_components}维, 稳态{steady_components}维")
    
    # 读取数据获取特征信息
    first_chunk = pd.read_csv(input_file, nrows=1000)
    
    # 识别瞬态和稳态特征列
    transient_cols = [col for col in first_chunk.columns if col.startswith('transient_')]
    steady_cols = [col for col in first_chunk.columns if col.startswith('steady_')]
    
    logger.info(f"瞬态特征列数: {len(transient_cols)}")
    logger.info(f"稳态特征列数: {len(steady_cols)}")
    
    # 调整组件数量
    actual_transient_components = min(transient_components, len(transient_cols))
    actual_steady_components = min(steady_components, len(steady_cols))
    
    # 初始化PCA
    transient_pca = IncrementalPCA(n_components=actual_transient_components) if transient_cols else None
    steady_pca = IncrementalPCA(n_components=actual_steady_components) if steady_cols else None
    
    # 训练PCA
    logger.info("训练PCA模型...")
    chunk_iter = pd.read_csv(input_file, chunksize=chunk_size)
    for chunk in tqdm(chunk_iter, desc="训练PCA"):
        if transient_pca and transient_cols:
            transient_data = chunk[transient_cols].values
            transient_pca.partial_fit(transient_data)
        
        if steady_pca and steady_cols:
            steady_data = chunk[steady_cols].values
            steady_pca.partial_fit(steady_data)
        
        gc.collect()
    
    # 应用PCA变换
    output_file = input_file.replace('.csv', f'_pca_t{actual_transient_components}_s{actual_steady_components}.csv')
    first_chunk = True
    
    logger.info("应用PCA变换...")
    chunk_iter = pd.read_csv(input_file, chunksize=chunk_size)
    for chunk in tqdm(chunk_iter, desc="PCA变换"):
        result_dfs = []
        
        # 瞬态PCA变换
        if transient_pca and transient_cols:
            transient_transformed = transient_pca.transform(chunk[transient_cols].values)
            transient_df = pd.DataFrame(transient_transformed,
                                      columns=[f"transient_component_{i+1}" 
                                             for i in range(actual_transient_components)])
            result_dfs.append(transient_df)
        
        # 稳态PCA变换
        if steady_pca and steady_cols:
            steady_transformed = steady_pca.transform(chunk[steady_cols].values)
            steady_df = pd.DataFrame(steady_transformed,
                                   columns=[f"steady_component_{i+1}" 
                                          for i in range(actual_steady_components)])
            result_dfs.append(steady_df)
        
        # 合并结果
        if result_dfs:
            result_df = pd.concat(result_dfs, axis=1)
            
            # 🔧 重置索引确保对齐
            result_df = result_df.reset_index(drop=True)
            chunk = chunk.reset_index(drop=True)
            
            # 确保长度一致
            if len(result_df) != len(chunk):
                logger.error(f"数据长度不一致: result_df={len(result_df)}, chunk={len(chunk)}")
                continue
        else:
            result_df = pd.DataFrame()
        
        # 添加元数据 - 🔧 确保数据对齐
        try:
            result_df['target'] = chunk['target'].values
            result_df['individual_id'] = chunk['individual_id'].values  
            result_df['route'] = chunk['route'].values
            
            # 验证没有NaN
            if result_df['target'].isna().any():
                logger.error(f"发现target NaN值，数量: {result_df['target'].isna().sum()}")
                continue
                
        except Exception as e:
            logger.error(f"元数据添加失败: {e}")
            continue
        
        if first_chunk:
            result_df.to_csv(output_file, index=False, mode='w')
            first_chunk = False
        else:
            result_df.to_csv(output_file, index=False, mode='a', header=False)
        
        del result_df
        gc.collect()
    
    logger.info(f"PCA完成，总维度: {actual_transient_components + actual_steady_components}")
    
    return transient_pca, steady_pca, output_file

def verify_output_data(output_file):
    """验证输出数据的完整性"""
    logger.info("验证输出数据完整性...")
    
    try:
        # 读取输出文件
        df = pd.read_csv(output_file, low_memory=False)
        logger.info(f"输出文件形状: {df.shape}")
        
        # 检查必要的列
        required_cols = ['target', 'individual_id', 'route']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"缺少必要列: {missing_cols}")
            return False
        
        # 检查NaN值
        nan_counts = df[required_cols].isnull().sum()
        for col, count in nan_counts.items():
            if count > 0:
                logger.error(f"{col}列有{count}个NaN值")
                logger.error(f"NaN行示例: {df[df[col].isnull()].index.tolist()[:10]}")
                return False
        
        # 检查空字符串
        for col in ['individual_id', 'route']:
            empty_count = (df[col].astype(str).str.strip() == '').sum()
            if empty_count > 0:
                logger.error(f"{col}列有{empty_count}个空字符串")
                return False
        
        # 统计信息
        logger.info(f"唯一individual_id数量: {df['individual_id'].nunique()}")
        logger.info(f"唯一target数量: {df['target'].nunique()}")
        logger.info(f"route分布: {df['route'].value_counts().to_dict()}")
        logger.info(f"target范围: {df['target'].min()} - {df['target'].max()}")
        
        # 显示前几行
        logger.info("前5行数据:")
        for i in range(min(5, len(df))):
            row = df.iloc[i]
            logger.info(f"  行{i}: target={row['target']}, id={row['individual_id']}, route={row['route']}")
        
        logger.info("✅ 数据验证通过") 
        return True
        
    except Exception as e:
        logger.error(f"数据验证失败: {e}")
        return False
    """清理临时文件"""
    for file_path in file_paths:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass

def cleanup_output_data(output_file):
    """清理输出数据中的问题行"""
    logger.info("清理输出数据...")
    
    try:
        # 读取数据
        df = pd.read_csv(output_file, low_memory=False)
        original_len = len(df)
        logger.info(f"原始数据长度: {original_len}")
        
        # 删除individual_id或route为空的行
        df = df.dropna(subset=['individual_id', 'route', 'target'])
        logger.info(f"删除NaN后长度: {len(df)}")
        
        # 删除individual_id或route为空字符串的行
        df = df[(df['individual_id'].astype(str).str.strip() != '') & 
                (df['route'].astype(str).str.strip() != '')]
        logger.info(f"删除空字符串后长度: {len(df)}")
        
        # 确保target是整数
        df['target'] = df['target'].astype(int)
        
        # 重新保存
        df.to_csv(output_file, index=False)
        logger.info(f"✅ 数据清理完成，最终长度: {len(df)}")
        
        return len(df) > 0
        
    except Exception as e:
        logger.error(f"数据清理失败: {e}")
        return False

def cleanup_temp_files(*file_paths):
    """清理临时文件"""
    for file_path in file_paths:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"已删除临时文件: {file_path}")
            except Exception as e:
                logger.warning(f"删除临时文件失败: {file_path}, 错误: {str(e)}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="智能分组特征降维")
    
    parser.add_argument("--data_path", type=str, required=True, help="增强数据路径")
    parser.add_argument("--output_path", type=str, default="./grouped_results", help="输出路径")
    parser.add_argument("--transient_components", type=int, default=15, help="瞬态特征PCA组件数")
    parser.add_argument("--steady_components", type=int, default=15, help="稳态特征PCA组件数")
    parser.add_argument("--chunk_size", type=int, default=2000, help="分块大小")
    parser.add_argument("--keep_temp_files", action="store_true", help="保留临时文件")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_path, exist_ok=True)
    
    logger.info("开始智能分组特征降维")
    logger.info(f"瞬态组件: {args.transient_components}, 稳态组件: {args.steady_components}")
    
    temp_files = []
    
    try:
        # 1. 整合特征
        logger.info("=== 步骤1: 整合特征 ===")
        total_samples, integrated_file = integrate_features_chunked(
            args.data_path, chunk_size=args.chunk_size
        )
        temp_files.append(integrated_file)
        
        # 2. 预处理和分组
        logger.info("=== 步骤2: 预处理和智能分组 ===")
        (transient_scaler, steady_scaler, transient_features, steady_features,
         target_mapping, processed_file) = preprocess_features_grouped(
            integrated_file, chunk_size=args.chunk_size
        )
        temp_files.append(processed_file)
        
        # 3. 分组PCA降维
        logger.info("=== 步骤3: 分组PCA降维 ===")
        transient_pca, steady_pca, final_file = apply_grouped_pca(
            processed_file,
            transient_components=args.transient_components,
            steady_components=args.steady_components,
            chunk_size=args.chunk_size
        )
        
        # 4. 验证和清理输出数据
        logger.info("=== 步骤4: 验证和清理输出数据 ===")
        if not verify_output_data(final_file):
            logger.warning("⚠️ 数据验证失败，尝试清理...")
            if not cleanup_output_data(final_file):
                logger.error("❌ 数据清理失败")
                return
            # 重新验证
            if not verify_output_data(final_file):
                logger.error("❌ 清理后数据仍有问题")
                return
        
        # 4. 保存最终文件
        final_output_path = os.path.join(
            args.output_path, 
            f"grouped_reduced_t{args.transient_components}_s{args.steady_components}.csv"
        )
        
        if os.path.exists(final_file):
            os.rename(final_file, final_output_path)
            
            # 验证文件
            verify_df = pd.read_csv(final_output_path, nrows=5)
            logger.info(f"✅ 最终文件: {final_output_path}")
            logger.info(f"✅ 列数: {len(verify_df.columns)}")
            logger.info(f"✅ 特征列: {[c for c in verify_df.columns if 'component' in c]}")
            del verify_df
        
        # 5. 保存模型和映射
        import joblib
        
        # 保存标准化器
        joblib.dump(transient_scaler, os.path.join(args.output_path, "transient_scaler.joblib"))
        joblib.dump(steady_scaler, os.path.join(args.output_path, "steady_scaler.joblib"))
        
        # 保存PCA模型
        if transient_pca:
            joblib.dump(transient_pca, os.path.join(args.output_path, "transient_pca.joblib"))
        if steady_pca:
            joblib.dump(steady_pca, os.path.join(args.output_path, "steady_pca.joblib"))
        
        # 保存特征映射
        feature_info = {
            'transient_features': transient_features,
            'steady_features': steady_features,
            'target_mapping': target_mapping,
            'transient_components': args.transient_components,
            'steady_components': args.steady_components
        }
        
        with open(os.path.join(args.output_path, "feature_info.json"), 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        # 6. 生成报告
        with open(os.path.join(args.output_path, "grouped_report.txt"), 'w') as f:
            f.write("智能分组特征降维报告\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"总样本数: {total_samples}\n")
            f.write(f"原始瞬态特征数: {len(transient_features)}\n")
            f.write(f"原始稳态特征数: {len(steady_features)}\n")
            f.write(f"降维后瞬态维度: {args.transient_components}\n")
            f.write(f"降维后稳态维度: {args.steady_components}\n")
            f.write(f"总降维维度: {args.transient_components + args.steady_components}\n")
            f.write(f"目标类别数: {len(target_mapping)}\n\n")
            
            f.write("瞬态特征示例:\n")
            for feature in transient_features[:10]:
                f.write(f"  - {feature}\n")
            f.write("\n稳态特征示例:\n")
            for feature in steady_features[:10]:
                f.write(f"  - {feature}\n")
            
            if transient_pca:
                f.write(f"\n瞬态PCA解释方差比: {transient_pca.explained_variance_ratio_[:5].tolist()}\n")
            if steady_pca:
                f.write(f"稳态PCA解释方差比: {steady_pca.explained_variance_ratio_[:5].tolist()}\n")
        
        logger.info("处理完成！")
        logger.info(f"最终特征维度: 瞬态 {args.transient_components} + 稳态 {args.steady_components} = 总计 {args.transient_components + args.steady_components}")
        logger.info(f"瞬态特征数: {len(transient_features)}")
        logger.info(f"稳态特征数: {len(steady_features)}")
        
    except Exception as e:
        logger.error(f"处理出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
    finally:
        if not args.keep_temp_files:
            cleanup_temp_files(*temp_files)
        gc.collect()

if __name__ == "__main__":
    main()