import os
import numpy as np
import h5py
from collections import defaultdict

def process_ads_b_data(mat_file_path, output_dir, skip_zeros=256):
    """
    处理ADS-B数据，根据标签分类数据并保存为I/Q路径
    通过映射将实际读取的标签值映射到1-107范围
    
    参数:
        mat_file_path: MATLAB .mat文件路径
        output_dir: 输出目录
        skip_zeros: 跳过前几列零值数据
    """
    print(f"处理ADS-B数据: {mat_file_path}")
    
    # 使用h5py加载MATLAB v7.3格式的数据文件
    with h5py.File(mat_file_path, 'r') as file:
        # 查找数据键
        baseband_key = "selectedBasebandData"
        raw_comp_key = "selectedRawCompData"
        
        # 读取数据
        baseband_dataset = file[baseband_key]
        raw_comp_dataset = file[raw_comp_key]
        
        print(f"基带数据集形状: {baseband_dataset.shape}")
        print(f"原始复数数据集形状: {raw_comp_dataset.shape}")
        
        # 正确处理MATLAB数据的转置问题
        baseband_data = baseband_dataset[()].T  # 转置
        
        # 处理复数数据
        if raw_comp_dataset.dtype.names and 'real' in raw_comp_dataset.dtype.names and 'imag' in raw_comp_dataset.dtype.names:
            print("检测到复数数据结构，提取实部和虚部")
            temp_data = raw_comp_dataset[()]
            real_part = temp_data['real'].T  # 转置
            imag_part = temp_data['imag'].T  # 转置
            raw_comp_data = real_part + 1j * imag_part
        else:
            raw_comp_data = raw_comp_dataset[()].T  # 转置
        
        print(f"转置后基带数据形状: {baseband_data.shape}")
        print(f"转置后原始复数数据形状: {raw_comp_data.shape}")
        
        # 确保行数匹配
        if raw_comp_data.shape[0] != baseband_data.shape[0]:
            raise ValueError(f"原始复数数据行数({raw_comp_data.shape[0]})与基带数据行数({baseband_data.shape[0]})不匹配")
        
        # 提取标签 (标签在第一列)
        original_labels = baseband_data[:, 0]
        original_labels = np.round(original_labels).astype(int)
        
        print(f"读取的原始标签前20个: {original_labels[:20]}")
        print(f"原始标签范围: {np.min(original_labels)} 到 {np.max(original_labels)}")
        
        # 获取唯一标签并排序
        unique_original_labels = sorted(np.unique(original_labels))
        print(f"唯一原始标签数量: {len(unique_original_labels)}")
        print(f"唯一原始标签: {unique_original_labels}")
        
        # 创建从原始标签到1-107范围的映射
        label_mapping = {}
        for i, label in enumerate(unique_original_labels, 1):
            if i <= 107:  # 只映射前107个标签
                label_mapping[label] = i
        
        print(f"创建的标签映射: {label_mapping}")
        
        # 应用映射到标签
        labels = np.array([label_mapping.get(label, 0) for label in original_labels])
        
        # 获取映射后的唯一标签
        unique_labels = np.unique(labels[labels > 0])  # 排除未映射的标签(0)
        print(f"映射后的唯一标签数量: {len(unique_labels)}")
        print(f"映射后的标签范围: {np.min(unique_labels) if len(unique_labels) > 0 else 0} 到 {np.max(unique_labels) if len(unique_labels) > 0 else 0}")
        
        # 统计每个标签的样本数
        label_counts = {}
        for label in labels:
            if label > 0:  # 只统计有效标签
                label_counts[label] = label_counts.get(label, 0) + 1
        
        print("\n每个标签的样本数:")
        for label, count in sorted(label_counts.items())[:20]:  # 只显示前20个
            print(f"标签 {label}: {count} 个样本")
        
        if len(label_counts) > 20:
            print("...")
        
        # 创建I路和Q路的根目录
        i_root_dir = os.path.join(output_dir, 'I_data')
        q_root_dir = os.path.join(output_dir, 'Q_data')
        
        os.makedirs(i_root_dir, exist_ok=True)
        os.makedirs(q_root_dir, exist_ok=True)
        
        # 创建1-107所有标签的目录
        for i in range(1, 108):
            aircraft_name = f"aircraft_{i:03d}"
            i_aircraft_dir = os.path.join(i_root_dir, aircraft_name)
            q_aircraft_dir = os.path.join(q_root_dir, aircraft_name)
            
            os.makedirs(i_aircraft_dir, exist_ok=True)
            os.makedirs(q_aircraft_dir, exist_ok=True)
        
        # 提取有效的复数数据部分 (去掉前skip_zeros列的零值数据)
        valid_cols = slice(skip_zeros, raw_comp_data.shape[1])
        
        # 按标签分组数据
        label_indices = defaultdict(list)
        for i, label in enumerate(labels):
            if label > 0:  # 只处理有效标签
                label_indices[label].append(i)
        
        # 处理每个标签的数据
        processed_labels = []
        
        for label in sorted(label_indices.keys()):
            indices = label_indices[label]
            processed_labels.append(label)
            
            aircraft_name = f"aircraft_{label:03d}"
            i_aircraft_dir = os.path.join(i_root_dir, aircraft_name)
            q_aircraft_dir = os.path.join(q_root_dir, aircraft_name)
            
            print(f"处理标签 {label}，共 {len(indices)} 个样本")
            
            # 限制每个标签最多处理400个样本
            if len(indices) > 400:
                print(f"标签 {label} 有 {len(indices)} 个样本，超过上限400，将只处理前400个")
                indices = indices[:400]
            
            # 处理每个样本
            for sample_idx, row_idx in enumerate(indices):
                # 提取当前样本的复数数据
                sample_data = raw_comp_data[row_idx, valid_cols]
                
                # 分离I路和Q路
                i_data = np.real(sample_data)
                q_data = np.imag(sample_data)
                
                # 保存I路和Q路数据到txt文件
                i_file_path = os.path.join(i_aircraft_dir, f"signal_{sample_idx+1:04d}.txt")
                q_file_path = os.path.join(q_aircraft_dir, f"signal_{sample_idx+1:04d}.txt")
                
                np.savetxt(i_file_path, i_data, fmt='%.10f')
                np.savetxt(q_file_path, q_data, fmt='%.10f')
        
        print("\n数据处理完成!")
        print(f"总共处理了 {len(processed_labels)} 个标签的数据")
        
        # 检查是否处理了所有107个标签
        if len(processed_labels) < 107:
            missing_labels = [i for i in range(1, 108) if i not in processed_labels]
            print(f"警告: 有 {len(missing_labels)} 个标签没有处理")
            if len(missing_labels) <= 20:
                print(f"未处理的标签: {missing_labels}")
            else:
                print(f"未处理的标签数量太多，不全部显示")

if __name__ == "__main__":
    # 设置输入文件路径和输出目录
    mat_file_path = "adsb-107loaded.mat"
    output_dir = "../data/splitted_data"
    
    try:
        # 处理数据
        process_ads_b_data(mat_file_path, output_dir, skip_zeros=256)
    except Exception as e:
        import traceback
        print(f"处理数据时出错: {e}")
        print("错误详情:")
        traceback.print_exc()