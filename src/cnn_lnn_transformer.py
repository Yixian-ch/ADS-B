import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.metrics import confusion_matrix


# 配置日志记录
def setup_logger(save_dir):
    """设置日志记录器"""
    log_file = os.path.join(save_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# 设置随机种子，确保结果可复现
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设置随机种子
set_seed(42)

def load_and_preprocess_grouped_data(data_path, test_size=0.2, val_size=0.1):
    """
    加载和预处理分组降维后的数据
    适配新的CSV格式：瞬态特征 + 稳态特征 + target + individual_id + route
    """
    print(f"加载分组数据: {data_path}")
    df = pd.read_csv(data_path)
    
    print(f"原始数据形状: {df.shape}")
    print(f"数据列名: {list(df.columns)}")
    
    # 分离瞬态和稳态特征
    transient_cols = [f'transient_component_{i}' for i in range(1, 16)]  # 15个瞬态特征
    steady_cols = [f'steady_component_{i}' for i in range(1, 16)]        # 15个稳态特征
    
    # 检查所有特征列是否存在
    missing_cols = []
    for col in transient_cols + steady_cols:
        if col not in df.columns:
            missing_cols.append(col)
    
    if missing_cols:
        print(f"❌ 缺少特征列: {missing_cols}")
        return None, None, None, None
    
    # 提取特征和标签
    X_transient = df[transient_cols].values.astype(np.float32)
    X_steady = df[steady_cols].values.astype(np.float32)
    
    # 处理标签 - 将target转换为从0开始的连续整数
    unique_targets = sorted(df['target'].unique())
    target_mapping = {target: idx for idx, target in enumerate(unique_targets)}
    y = df['target'].map(target_mapping).values.astype(np.int64)
    
    print(f"瞬态特征维度: {X_transient.shape}")
    print(f"稳态特征维度: {X_steady.shape}")
    print(f"标签维度: {y.shape}")
    print(f"类别数量: {len(unique_targets)}")
    print(f"标签范围: {y.min()} - {y.max()}")
    
    # 合并瞬态和稳态特征为单一特征矩阵
    X = np.concatenate([X_transient, X_steady], axis=1)  # 30维特征
    
    print(f"合并后特征维度: {X.shape}")
    
    # 标准化特征
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 分割数据集
    from sklearn.model_selection import train_test_split
    
    # 首先分割出测试集
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # 从剩余数据中分割出验证集
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
    )
    
    print(f"数据分割完成. 训练集: {len(y_train)}, 验证集: {len(y_val)}, 测试集: {len(y_test)}")
    
    # 创建张量数据集
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )
    
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )
    
    return train_dataset, val_dataset, test_dataset, len(unique_targets), target_mapping, scaler

# 自注意力模块，用于特征内部的关系建模
class SelfAttention(nn.Module):
    """
    自注意力模块，用于特征内部的关系建模
    """
    def __init__(self, embed_size, heads=1):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert (self.head_dim * heads == embed_size), "Embed size needs to be divisible by heads"
        
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)
        self.scale = self.head_dim ** 0.5
        
    def forward(self, x):
        # x: [batch_size, embed_size]
        batch_size = x.shape[0]
        
        # 将输入转换为适合注意力机制的形状
        x_reshaped = x.unsqueeze(1)  # [batch_size, 1, embed_size]
        
        # 将输入通过线性层并分割成多头
        queries = self.query(x_reshaped)  # [batch_size, 1, embed_size]
        keys = self.key(x_reshaped)       # [batch_size, 1, embed_size]
        values = self.value(x_reshaped)   # [batch_size, 1, embed_size]
        
        # 将embed_size分割成heads个部分
        queries = queries.reshape(batch_size, 1, self.heads, self.head_dim)
        keys = keys.reshape(batch_size, 1, self.heads, self.head_dim)
        values = values.reshape(batch_size, 1, self.heads, self.head_dim)
        
        # 调整维度顺序，便于后续操作
        queries = queries.permute(0, 2, 1, 3)  # [batch_size, heads, 1, head_dim]
        keys = keys.permute(0, 2, 1, 3)        # [batch_size, heads, 1, head_dim]
        values = values.permute(0, 2, 1, 3)    # [batch_size, heads, 1, head_dim]
        
        # 计算注意力分数
        energy = torch.matmul(queries, keys.permute(0, 1, 3, 2))  # [batch_size, heads, 1, 1]
        
        # 缩放分数
        energy = energy / self.scale
        
        # 计算注意力权重
        attention = torch.softmax(energy, dim=-1)  # [batch_size, heads, 1, 1]
        
        # 应用注意力权重
        out = torch.matmul(attention, values)  # [batch_size, heads, 1, head_dim]
        
        # 重新排列维度
        out = out.permute(0, 2, 1, 3)  # [batch_size, 1, heads, head_dim]
        
        # 合并多头结果
        out = out.reshape(batch_size, 1, self.heads * self.head_dim)  # [batch_size, 1, embed_size]
        
        # 输出变换
        out = self.fc_out(out)  # [batch_size, 1, embed_size]
        
        # 去除序列维度
        out = out.squeeze(1)  # [batch_size, embed_size]
        
        return out, attention

class TransientPathway(nn.Module):
    """
    瞬态路径网络，处理前15个主成分（瞬态特征）
    """
    def __init__(self, input_dim=15, hidden_dim=128, dropout=0.3):
        super(TransientPathway, self).__init__()
        
        # 特征扩展层
        self.feature_expansion = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )
        
        # 一维卷积层，提取局部模式
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )
        
        # 特征重新映射
        self.feature_remap = nn.Sequential(
            nn.Linear(64 * hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )
        
        # 自注意力层，捕捉特征间的关系
        self.attention = SelfAttention(hidden_dim)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # 特征扩展
        x = self.feature_expansion(x)  # [batch_size, hidden_dim]
        
        # 调整输入维度用于一维卷积 [batch_size, 1, hidden_dim]
        x = x.unsqueeze(1)
        
        # 卷积处理
        conv_out = self.conv_layers(x)  # [batch_size, 64, hidden_dim]
        
        # 重塑为全连接层输入
        conv_out = conv_out.reshape(batch_size, -1)  # [batch_size, 64 * hidden_dim]
        
        # 特征重新映射
        mapped_features = self.feature_remap(conv_out)  # [batch_size, hidden_dim]
        
        # 应用自注意力
        attended, _ = self.attention(mapped_features)  # [batch_size, hidden_dim]
        
        # 输出层
        output = self.output_layer(attended)  # [batch_size, hidden_dim]
        
        return output

class SteadyPathway(nn.Module):
    """
    稳态路径网络，处理后15个主成分（稳态特征）
    """
    def __init__(self, input_dim=15, hidden_dim=128, dropout=0.3):
        super(SteadyPathway, self).__init__()
        
        # 特征扩展层
        self.feature_expansion = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )
        
        # 1D CNN用于频域分析
        self.conv_layers = nn.Sequential(
            # 第一层卷积
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            
            # 第二层卷积
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            
            # 第三层卷积
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
        )
        
        # 自适应池化层
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )
        
        # 自注意力层
        self.attention = SelfAttention(hidden_dim)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # 特征扩展
        x = self.feature_expansion(x)  # [batch_size, hidden_dim]
        
        # 将特征重塑为适合CNN的形状 [batch_size, 1, hidden_dim]
        x = x.unsqueeze(1)
        
        # 应用卷积层
        x = self.conv_layers(x)  # [batch_size, 128, hidden_dim]
        
        # 应用自适应池化
        x = self.adaptive_pool(x)  # [batch_size, 128, 1]
        x = x.squeeze(2)  # [batch_size, 128]
        
        # 应用全连接层
        x = self.fc_layers(x)  # [batch_size, hidden_dim]
        
        # 应用自注意力
        x, _ = self.attention(x)  # [batch_size, hidden_dim]
        
        return x

class DualPathNetwork(nn.Module):
    """
    双路径网络，适应分组PCA降维后的数据格式
    """
    def __init__(self, num_classes, hidden_dim=128):
        super(DualPathNetwork, self).__init__()
        
        # 瞬态路径 - 处理前15个主成分
        self.transient_pathway = TransientPathway(input_dim=15, hidden_dim=hidden_dim)
        
        # 稳态路径 - 处理后15个主成分
        self.steady_pathway = SteadyPathway(input_dim=15, hidden_dim=hidden_dim)
        
        # 特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x):
        # 分割输入特征
        # x: [batch_size, 30] - 30个主成分 (15瞬态 + 15稳态)
        x_transient = x[:, :15]   # 前15个主成分作为瞬态特征
        x_steady = x[:, 15:]      # 后15个主成分作为稳态特征
        
        # 处理瞬态特征
        transient_features = self.transient_pathway(x_transient)
        
        # 处理稳态特征
        steady_features = self.steady_pathway(x_steady)
        
        # 特征融合
        combined_features = torch.cat([transient_features, steady_features], dim=1)
        fused_features = self.fusion_layer(combined_features)
        
        # 分类
        logits = self.classifier(fused_features)
        
        return logits

class SimplifiedCNNPath(nn.Module):
    """简化的CNN路径，适应分组降维后的数据"""
    def __init__(self, input_dim, num_classes, dropout=0.3):
        super(SimplifiedCNNPath, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

class RobustLNNPath(nn.Module):
    """增强的LNN路径，更好地处理噪声环境"""
    def __init__(self, input_dim, num_classes, dropout=0.3):
        super(RobustLNNPath, self).__init__()
        
        # 输入投影层
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.BatchNorm1d(96),
            nn.GELU(),
            nn.Dropout(dropout/2)
        )
        
        # 使用多层GRU提高噪声鲁棒性
        self.gru = nn.GRU(input_size=96, 
                          hidden_size=48, 
                          num_layers=2,
                          batch_first=True,
                          dropout=0.2,
                          bidirectional=True)
        
        # 注意力机制，增强对关键特征的关注
        self.attention = nn.Sequential(
            nn.Linear(96, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        self.output = nn.Sequential(
            nn.Linear(96, num_classes),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # 输入投影
        x = self.input_projection(x)  # [batch, 96]
        
        # 添加时间维度并重复输入创建序列
        batch_size = x.size(0)
        x_repeated = x.unsqueeze(1).repeat(1, 3, 1)  # [batch, 3, 96]
        
        # GRU处理
        out, _ = self.gru(x_repeated)  # [batch, 3, 96]
        
        # 注意力机制
        attn_weights = self.attention(out)  # [batch, 3, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # 加权汇总
        context = torch.sum(out * attn_weights, dim=1)  # [batch, 96]
        
        # 输出层
        return self.output(context)

class ImprovedAdaptiveWeightedLoss(nn.Module):
    """改进的自适应加权损失函数"""
    def __init__(self, num_classes, snr_guidance_weight=0.1):
        super(ImprovedAdaptiveWeightedLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction='none', label_smoothing=0.1)  # 增加标签平滑
        self.num_classes = num_classes
        self.snr_guidance_weight = snr_guidance_weight
    
    def forward(self, final_logits, cnn_logits, lnn_logits, targets, snr, model_weights):
        # 计算各个分支的损失
        cnn_loss = self.ce_loss(cnn_logits, targets).mean()
        lnn_loss = self.ce_loss(lnn_logits, targets).mean()
        ensemble_loss = self.ce_loss(final_logits, targets).mean()
        
        # 归一化SNR
        normalized_snr = torch.clamp(snr / 30.0, -1.0, 1.0)
        
        # 平衡权重损失
        weight_balance_loss = torch.abs(model_weights[:, 0] - model_weights[:, 1]).mean()
        
        # 高SNR时引导CNN分支
        high_snr_mask = (normalized_snr > 0.3).squeeze()
        high_snr_guidance = torch.zeros_like(ensemble_loss)
        if high_snr_mask.any():
            cnn_weight_loss = F.relu(0.7 - model_weights[high_snr_mask, 0]).mean()
            high_snr_guidance = cnn_weight_loss * self.snr_guidance_weight
        
        # 低SNR时引导LNN分支
        low_snr_mask = (normalized_snr < -0.3).squeeze()
        low_snr_guidance = torch.zeros_like(ensemble_loss)
        if low_snr_mask.any():
            lnn_weight_loss = F.relu(0.7 - model_weights[low_snr_mask, 1]).mean()
            low_snr_guidance = lnn_weight_loss * self.snr_guidance_weight
        
        # 总损失
        total_loss = ensemble_loss + 0.2 * (cnn_loss + lnn_loss) + \
                     0.1 * weight_balance_loss + high_snr_guidance + low_snr_guidance
        
        component_losses = {
            'cnn_loss': cnn_loss.item(),
            'lnn_loss': lnn_loss.item(),
            'ensemble_loss': ensemble_loss.item(),
            'weight_balance_loss': weight_balance_loss.item(),
            'weighted_loss': total_loss.item()
        }
        
        return total_loss, component_losses

class ImprovedEnsembleADSBNetwork(nn.Module):
    """改进的集成ADS-B网络 - 适配分组数据"""
    def __init__(self, input_dim=30, num_classes=107, hidden_dim=128):
        super(ImprovedEnsembleADSBNetwork, self).__init__()
        self.num_classes = num_classes
        
        # 特征预处理层
        self.feature_processor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # 特征分割：前一半作为瞬态特征，后一半作为稳态特征
        self.transient_extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        self.steady_extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # 融合特征处理
        self.fused_extractor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 两个主要模型分支
        self.cnn_model = SimplifiedCNNPath(hidden_dim * 2, num_classes)
        self.lnn_model = RobustLNNPath(hidden_dim * 2, num_classes)
        
        # SNR估计器
        self.snr_estimator_freq = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU()
        )
        
        self.snr_estimator_energy = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        self.snr_peak_estimator = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU()
        )
        
        self.snr_fusion = nn.Sequential(
            nn.Linear(32 + 32 + 16, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1),
            nn.Tanh()
        )
        
        # 权重分配网络：只输出2个分支权重
        self.weight_net_fc1 = nn.Linear(1, 64)
        self.weight_net_fc2 = nn.Linear(64, 32)
        self.weight_net_out = nn.Linear(32, 2)
        
        # 融合层
        encoder_layer = nn.TransformerEncoderLayer(d_model=num_classes, nhead=1, dim_feedforward=64, dropout=0.1, batch_first=True)
        self.transformer_fusion = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fusion_fc = nn.Linear(num_classes, num_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        with torch.no_grad():
            nn.init.xavier_normal_(self.weight_net_fc1.weight)
            nn.init.xavier_normal_(self.weight_net_fc2.weight)
            nn.init.xavier_normal_(self.weight_net_out.weight)
            self.weight_net_out.bias.fill_(0.0)
    
    def _compute_adaptive_weights(self, snr):
        if snr.dim() > 1 and snr.size(1) == 1:
            x = snr
        else:
            x = snr.unsqueeze(1) if snr.dim() == 1 else snr
        x = F.leaky_relu(self.weight_net_fc1(x))
        x = F.leaky_relu(self.weight_net_fc2(x))
        weights_logits = self.weight_net_out(x)
        weights = F.softmax(weights_logits, dim=1)
        return weights
    
    def forward(self, x):
        # 特征预处理
        processed_features = self.feature_processor(x)
        
        # 分割特征为瞬态和稳态部分
        mid_dim = processed_features.size(1) // 2
        trans_part = processed_features[:, :mid_dim]
        steady_part = processed_features[:, mid_dim:]
        
        # 特征提取
        trans_features = self.transient_extractor(trans_part)
        steady_features = self.steady_extractor(steady_part)
        
        # 融合特征
        combined = torch.cat([trans_features, steady_features], dim=1)
        features = self.fused_extractor(combined)
        
        # SNR估计
        snr_freq = self.snr_estimator_freq(features)
        snr_energy = self.snr_estimator_energy(features)
        snr_peak = self.snr_peak_estimator(features)
        snr_combined = torch.cat([snr_freq, snr_energy, snr_peak], dim=1)
        snr = self.snr_fusion(snr_combined) * 30
        
        # 获取两个模型的输出
        cnn_logits = self.cnn_model(features)
        lnn_logits = self.lnn_model(features)
        
        # 计算自适应权重
        weights = self._compute_adaptive_weights(snr)
        
        # 加权融合
        weighted_cnn = cnn_logits * weights[:, 0].unsqueeze(1)
        weighted_lnn = lnn_logits * weights[:, 1].unsqueeze(1)
        
        # Transformer融合
        fusion_seq = torch.stack([weighted_cnn, weighted_lnn], dim=1)
        fusion_out = self.transformer_fusion(fusion_seq)
        fusion_pooled = fusion_out.mean(dim=1)
        final_logits = self.fusion_fc(fusion_pooled)
        
        return final_logits, snr, weights

def add_advanced_noise_to_batch(batch_x, min_snr=-10, max_snr=20, snr_high=0.90, snr_mid=0.09, snr_low=0.01):
    """为批次数据添加更高级的噪声模型"""
    batch_size = batch_x.size(0)
    noisy_x = torch.zeros_like(batch_x)
    
    # SNR区间
    snr_weights = {
        (15, 20): snr_high,
        (5, 15): snr_mid,
        (min_snr, 5): snr_low
    }
    
    def add_complex_noise(signal, snr_db):
        """添加复杂噪声模型，包括高斯、脉冲和频域噪声"""
        signal_numpy = signal.detach().cpu().numpy()
        signal_power = np.mean(signal_numpy ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), signal_numpy.shape)
        
        # 对于低SNR，添加额外的噪声类型
        if snr_db < 0:
            # 添加脉冲噪声
            impulse_mask = np.random.rand(*signal_numpy.shape) < 0.05
            impulse_noise = np.random.randn(*signal_numpy.shape) * 3
            signal_numpy[impulse_mask] += impulse_noise[impulse_mask]
        
        noisy_signal = signal_numpy + noise
        return torch.FloatTensor(noisy_signal)
    
    # 为每个样本选择SNR并添加噪声
    actual_snrs = []
    for i in range(batch_size):
        # 随机选择SNR范围
        snr_range = random.choices(list(snr_weights.keys()), weights=list(snr_weights.values()), k=1)[0]
        
        # 在选定范围内均匀随机选择SNR值
        snr_db = snr_range[0] + (snr_range[1] - snr_range[0]) * random.random()
        actual_snrs.append(snr_db)
        
        # 添加复杂噪声
        noisy_x[i] = add_complex_noise(batch_x[i], snr_db)
    
    return noisy_x, actual_snrs

class NoiseDataLoader:
    """添加噪声的数据加载器"""
    def __init__(self, dataset, batch_size=32, shuffle=True, 
                noise_prob=0.5, min_snr=-10, max_snr=20):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.noise_prob = noise_prob
        self.min_snr = min_snr
        self.max_snr = max_snr
        
        # 创建原始数据加载器
        self.dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle
        )
    
    def __iter__(self):
        self.iterator = iter(self.dataloader)
        return self
    
    def __next__(self):
        # 获取下一批数据
        batch_x, batch_y = next(self.iterator)
        
        # 为每个样本生成随机SNR值
        batch_size = batch_x.size(0)
        batch_snr = torch.zeros(batch_size, 1)
        
        # 有一定概率添加噪声
        if random.random() < self.noise_prob:
            batch_x, actual_snrs = add_advanced_noise_to_batch(
                batch_x, 
                min_snr=self.min_snr, max_snr=self.max_snr
            )
            batch_snr = torch.tensor(actual_snrs).float().unsqueeze(1)
        else:
            # 如果不添加噪声，假设SNR为最大值
            batch_snr.fill_(self.max_snr)
        
        return batch_x, batch_y, batch_snr
    
    def __len__(self):
        return len(self.dataloader)

class NoisySampler:
    """训练时动态选择部分数据并添加噪声的采样器"""
    def __init__(self, dataset, subset_ratio=0.3, change_epochs=5, 
                noise_prob=0.5, min_snr=-15, max_snr=25):
        self.dataset = dataset
        self.subset_ratio = subset_ratio
        self.change_epochs = change_epochs
        self.noise_prob = noise_prob
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.current_subset = None
        self.epoch = 0
        
        # 初始化子集
        self._create_new_subset()
    
    def _create_new_subset(self):
        """创建新的子集"""
        dataset_size = len(self.dataset)
        subset_size = int(dataset_size * self.subset_ratio)
        
        # 随机选择索引
        indices = torch.randperm(dataset_size)[:subset_size].tolist()
        self.current_subset = Subset(self.dataset, indices)
    
    def get_loader(self, batch_size, shuffle=True):
        """获取当前子集的数据加载器"""
        return NoiseDataLoader(
            self.current_subset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            noise_prob=self.noise_prob,
            min_snr=self.min_snr,
            max_snr=self.max_snr
        )
    
    def next_epoch(self):
        """进入下一个epoch，必要时更新子集"""
        self.epoch += 1
        
        # 每change_epochs个epoch更换一次子集
        if self.epoch % self.change_epochs == 0:
            self._create_new_subset()

def train_improved_ensemble(
        model, 
        train_dataset, 
        val_dataset, 
        num_epochs=40, 
        batch_size=32, 
        subset_ratio=0.5,
        change_epochs=5,
        noise_prob=0.7,
        min_snr=-10,
        max_snr=20,
        lr=0.001,
        save_dir='./improved_ensemble_results',
        snr_high=0.90,
        snr_mid=0.09,
        snr_low=0.01
    ):
    """
    训练改进的集成模型，增强低SNR环境下的性能
    """
    import os
    import time
    from tqdm import tqdm
    
    # 创建结果目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置日志记录器
    logger = setup_logger(save_dir)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = ImprovedAdaptiveWeightedLoss(num_classes=model.fusion_fc.out_features)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # 使用余弦退火调度器，带有热启动
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=lr/20
    )
    
    # 创建带噪声的数据采样器
    sampler = NoisySampler(
        train_dataset, 
        subset_ratio=subset_ratio, 
        change_epochs=change_epochs,
        noise_prob=noise_prob,
        min_snr=min_snr,
        max_snr=max_snr
    )
    
    # 创建验证数据加载器
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epoch_time': [],
        'mean_weights': []
    }
    
    # 最佳模型追踪
    best_val_acc = 0.0
    best_model_epoch = 0
    patience = 15  # 早停容忍次数
    early_stop_counter = 0
    
    logger.info(f"开始训练改进的集成模型，使用{subset_ratio*100:.1f}%的数据...")
    logger.info(f"噪声添加概率: {noise_prob*100:.1f}%, 信噪比范围: [{min_snr}, {max_snr}] dB")
    
    for epoch in range(num_epochs):
        # 更新子集
        sampler.next_epoch()
        # 使用改进的数据加载器
        train_loader = sampler.get_loader(batch_size, shuffle=True)
        
        # 初始化本epoch的SNR收集列表
        epoch_snrs = []
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        epoch_weights = []
        
        epoch_start_time = time.time()
        
        # 训练循环
        for batch_idx, batch_data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            # 解包数据 - 注意现在返回3个值，包括SNR
            batch_x, batch_y, batch_snr = batch_data
            batch_x, batch_y, batch_snr = (
                batch_x.to(device), 
                batch_y.to(device),
                batch_snr.to(device)
            )
            # 收集本batch的SNR
            epoch_snrs.extend(batch_snr.cpu().numpy().flatten().tolist())
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            final_logits, snr, model_weights = model(batch_x)
            # 获取各分支logits
            with torch.no_grad():
                # 重新计算特征
                processed_features = model.feature_processor(batch_x)
                mid_dim = processed_features.size(1) // 2
                trans_part = processed_features[:, :mid_dim]
                steady_part = processed_features[:, mid_dim:]
                trans_features = model.transient_extractor(trans_part)
                steady_features = model.steady_extractor(steady_part)
                combined = torch.cat([trans_features, steady_features], dim=1)
                features = model.fused_extractor(combined)
                cnn_logits = model.cnn_model(features)
                lnn_logits = model.lnn_model(features)
            
            # 损失函数调用
            loss, component_losses = criterion(final_logits, cnn_logits, lnn_logits, batch_y, snr, model_weights)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            
            # 优化步骤
            optimizer.step()
            
            # 统计
            train_loss += loss.item() * batch_y.size(0)
            _, predicted = torch.max(final_logits, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
            
            # 记录权重
            epoch_weights.append(model_weights.detach().cpu())
        
        # 计算平均训练损失和准确率
        train_loss = train_loss / train_total
        train_accuracy = train_correct / train_total
        
        # 计算平均权重
        epoch_weights = torch.cat(epoch_weights, dim=0)
        mean_weights = epoch_weights.mean(dim=0).numpy()
        
        # 验证阶段 - 每2个epoch验证一次
        if epoch % 2 == 0 or epoch == num_epochs - 1:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    
                    # 前向传播
                    final_logits, snr, model_weights = model(batch_x)
                    # 获取各分支logits
                    processed_features = model.feature_processor(batch_x)
                    mid_dim = processed_features.size(1) // 2
                    trans_part = processed_features[:, :mid_dim]
                    steady_part = processed_features[:, mid_dim:]
                    trans_features = model.transient_extractor(trans_part)
                    steady_features = model.steady_extractor(steady_part)
                    combined = torch.cat([trans_features, steady_features], dim=1)
                    features = model.fused_extractor(combined)
                    cnn_logits = model.cnn_model(features)
                    lnn_logits = model.lnn_model(features)
                    
                    # 损失函数调用
                    loss, _ = criterion(final_logits, cnn_logits, lnn_logits, batch_y, snr, model_weights)
                    
                    # 统计
                    val_loss += loss.item() * batch_y.size(0)
                    _, predicted = torch.max(final_logits, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            # 计算平均验证损失和准确率
            val_loss = val_loss / val_total
            val_accuracy = val_correct / val_total
            
            # 保存最佳模型
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                best_model_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(save_dir, 'best_improved_model.pth'))
                logger.info(f"保存新的最佳模型，验证准确率: {val_accuracy:.4f}")
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                logger.info(f"验证集准确率连续未提升次数: {early_stop_counter}/{patience}")
                if early_stop_counter >= patience:
                    logger.info(f"早停触发！验证集准确率连续{patience}次未提升，提前终止训练。")
                    break
        else:
            # 未进行验证的epoch使用上一个结果
            val_loss = history['val_loss'][-1] if history['val_loss'] else float('inf')
            val_accuracy = history['val_acc'][-1] if history['val_acc'] else 0.0
        
        # 更新学习率
        scheduler.step()
        
        # 计算epoch用时
        epoch_time = time.time() - epoch_start_time
        
        # 保存训练历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_accuracy)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_accuracy)
        history['epoch_time'].append(epoch_time)
        history['mean_weights'].append(mean_weights)
        
        # 记录训练信息到日志
        logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                   f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, "
                   f"Time: {epoch_time:.2f}s, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 记录权重分布到日志
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            logger.info(f"平均权重分布 - CNN: {mean_weights[0]:.3f}, LNN: {mean_weights[1]:.3f}")
        
        # 记录SNR分布到日志
        logger.info(f"Epoch {epoch+1} SNR分布: 均值={np.mean(epoch_snrs):.2f}, 方差={np.var(epoch_snrs):.2f}")
    
    # 加载最佳模型
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_improved_model.pth')))
    
    # 绘制训练历史
    plt.figure(figsize=(15, 10))
    
    # 损失曲线
    plt.subplot(2, 2, 1)
    epochs = range(1, len(history['train_loss']) + 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='training loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 准确率曲线
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='training accuracy')
    plt.plot(epochs, history['val_acc'], 'r-', label='validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # 权重分布曲线
    plt.subplot(2, 2, 3)
    weights_array = np.array(history['mean_weights'])
    plt.plot(epochs, weights_array[:, 0], 'r-', label='CNN weights')
    plt.plot(epochs, weights_array[:, 1], 'g-', label='LNN weights')
    plt.title('Model Weights Distribution Evolution')
    plt.xlabel('Epochs')
    plt.ylabel('Average Weights')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'improved_training_history.png'), dpi=300)
    plt.close()
    
    logger.info(f"训练完成！最佳验证准确率: {best_val_acc:.4f} (Epoch {best_model_epoch})")
    logger.info(f"平均每个epoch耗时: {np.mean(history['epoch_time']):.2f}秒")
    
    return model, best_val_acc

def test_snr_performance_improved(
        model, 
        test_dataset, 
        snr_range=None, 
        batch_size=32, 
        save_dir='./improved_ensemble_results'
    ):
    """测试改进模型在不同SNR下的性能"""
    # 如果没有指定SNR范围，使用默认值
    if snr_range is None:
        snr_range = [20, 15, 10, 5, 0, -5, -10]
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # 准备测试结果
    results = {
        'snr_db': snr_range,
        'accuracy': [],
        'cnn_weight': [],
        'lnn_weight': []
    }
    
    # 测试每个SNR
    for snr_db in snr_range:
        print(f"测试SNR = {snr_db} dB的性能...")
        
        # 创建测试加载器
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 评估指标
        correct = 0
        total = 0
        all_weights = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                # 添加指定SNR的噪声
                batch_x, _ = add_advanced_noise_to_batch(
                    batch_x, 
                    min_snr=snr_db, max_snr=snr_db  # 固定SNR值
                )
                
                # 转移到设备
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                # 前向传播
                final_logits, snr, weights = model(batch_x)
                
                # 统计预测结果
                _, predicted = torch.max(final_logits, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
                # 收集权重
                all_weights.append(weights.cpu())
        
        # 计算准确率和平均权重
        accuracy = correct / total
        all_weights = torch.cat(all_weights, dim=0)
        avg_weights = torch.mean(all_weights, dim=0).numpy()
        
        # 保存结果
        results['accuracy'].append(accuracy)
        results['cnn_weight'].append(avg_weights[0])
        results['lnn_weight'].append(avg_weights[1])
        
        # 打印结果
        print(f"SNR {snr_db} dB - 准确率: {accuracy:.4f}, 权重 - CNN: {avg_weights[0]:.4f}, LNN: {avg_weights[1]:.4f}")
     
    # 保存结果到CSV
    import pandas as pd
    pd.DataFrame(results).to_csv(os.path.join(save_dir, 'improved_snr_results.csv'), index=False)
    
    # 绘制SNR性能曲线
    plt.figure(figsize=(12, 10))
    
    # 准确率曲线
    plt.subplot(2, 1, 1)
    plt.plot(snr_range, results['accuracy'], 'o-', markersize=8, linewidth=2)
    plt.axhline(y=0.95, color='r', linestyle='--', linewidth=2, label='95% 目标线')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Accuracy')
    plt.title('Model Performance vs. Signal-to-Noise Ratio')
    plt.grid(True)
    plt.legend()
    plt.ylim(0, 1.05)
    
    # 权重曲线
    plt.subplot(2, 1, 2)
    plt.plot(snr_range, results['cnn_weight'], 'o-', label='CNN Weight', linewidth=2)
    plt.plot(snr_range, results['lnn_weight'], 's-', label='LNN Weight', linewidth=2)
    plt.xlabel('SNR (dB)')
    plt.ylabel('Weight')
    plt.title('Model Weights vs. Signal-to-Noise Ratio')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'improved_snr_performance.png'), dpi=300)
    plt.close()
    
    return results


def extract_features_for_visualization(model, dataset, device, max_samples=2000):
    """
    提取模型中间层特征用于可视化
    """
    model.eval()
    
    # 注册hook来提取特征
    features_list = []
    def hook_fn(module, input, output):
        features_list.append(output.detach().cpu())
    
    # 在融合层提取特征
    hook = model.fused_extractor.register_forward_hook(hook_fn)
    
    # 准备数据
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    all_labels = []
    sample_count = 0
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            if sample_count >= max_samples:
                break
                
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # 前向传播以触发hook
            _ = model(batch_x)
            
            # 收集标签
            all_labels.extend(batch_y.cpu().numpy())
            sample_count += len(batch_y)
    
    # 移除hook
    hook.remove()
    
    # 合并特征
    if features_list:
        features = torch.cat(features_list, dim=0)[:max_samples]
        labels = np.array(all_labels)[:max_samples]
        return features.numpy(), labels
    else:
        return None, None

def create_individual_identification_plot(model, test_dataset, save_dir, device, 
                                        method='tsne', max_samples=2000):
    """
    创建个体识别可视化图
    """
    print(f"🎨 生成个体识别可视化图 (方法: {method.upper()})...")
    
    # 提取特征
    features, labels = extract_features_for_visualization(model, test_dataset, device, max_samples)
    
    if features is None:
        print("❌ 特征提取失败")
        return
    
    print(f"提取特征维度: {features.shape}")
    print(f"标签数量: {len(np.unique(labels))}")
    
    # 降维处理
    if method == 'tsne':
        print("执行t-SNE降维...")
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)//4))
        features_2d = reducer.fit_transform(features)
    
    # 创建颜色映射
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    
    # 使用更好的颜色映射
    if n_classes <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    elif n_classes <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, n_classes))
    else:
        colors = plt.cm.nipy_spectral(np.linspace(0, 1, n_classes))
    
    # 创建图像
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # 为每个类别绘制点
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   c=[colors[i]], label=f'Aircraft {label:03d}', 
                   s=30, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    ax.set_title(f'Individual Aircraft Identification Visualization ({method.upper()})', 
              fontsize=16, fontweight='bold')
    ax.set_xlabel(f'{method.upper()} Component 1', fontsize=12)
    ax.set_ylabel(f'{method.upper()} Component 2', fontsize=12)
    
    # 添加colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.nipy_spectral, norm=plt.Normalize(vmin=unique_labels.min(), vmax=unique_labels.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Aircraft ID', rotation=270, labelpad=15)
    plt.tight_layout()
    
    # 保存图像
    save_path = os.path.join(save_dir, f'individual_identification_{method}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 个体识别图已保存: {save_path}")
    
    return features_2d, labels

def create_confusion_matrix_plot(model, test_dataset, device, save_dir, batch_size=128):
    """
    生成并保存个体标签识别的混淆矩阵（图片+CSV）
    """
    print("\n📊 正在生成混淆矩阵...")
    model.eval()
    all_preds = []
    all_labels = []
    from torch.utils.data import DataLoader
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            final_logits, _, _ = model(batch_x)
            preds = torch.argmax(final_logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    # 保存为CSV
    cm_csv_path = os.path.join(save_dir, 'confusion_matrix.csv')
    pd.DataFrame(cm).to_csv(cm_csv_path, index=True, header=True)
    print(f"混淆矩阵CSV已保存: {cm_csv_path}")
    # 绘制热力图
    plt.figure(figsize=(min(20, 0.5*cm.shape[0]), min(20, 0.5*cm.shape[1])))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', cbar=True)
    plt.title('Individual Identification Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    cm_img_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(cm_img_path, dpi=300)
    plt.close()
    print(f"混淆矩阵图片已保存: {cm_img_path}")
    return cm

def create_all_visualizations(model, test_dataset, device, save_dir):
    """
    创建所有可视化图表
    """
    print("\n🎨 开始生成可视化图表...")
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    # 1. t-SNE可视化
    create_individual_identification_plot(
        model, test_dataset, save_dir, device, method='tsne'
    )
    # 2. 混淆矩阵
    create_confusion_matrix_plot(model, test_dataset, device, save_dir)
    print("可视化图表生成完成！")

def main(data_path, batch_size=32, save_dir='./improved_results', subset_ratio=0.5, 
         snr_high=0.90, snr_mid=0.09, snr_low=0.01, min_snr=-10, max_snr=20, 
         finetune_epochs=15, num_epochs=30):
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # 创建结果目录
    os.makedirs(save_dir, exist_ok=True)
    finetune_save_dir = os.path.join(save_dir, 'finetune_high_snr')
    os.makedirs(finetune_save_dir, exist_ok=True)

    # 加载和预处理数据
    print(f"加载分组数据: {data_path}")
    train_dataset, val_dataset, test_dataset, num_classes, label_map, scaler = load_and_preprocess_grouped_data(data_path)
    print(f"类别数量: {num_classes}")

    # 创建模型 - input_dim修改为30 (15瞬态 + 15稳态)
    model = ImprovedEnsembleADSBNetwork(
        input_dim=30,  # 修改为30维输入
        num_classes=num_classes,
        hidden_dim=128
    )

    # 训练模型
    finetune_lr = 0.0002
    model, _ = train_improved_ensemble(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=num_epochs,
        batch_size=256,
        subset_ratio=1.0,
        change_epochs=2,
        noise_prob=0.3,
        min_snr=-10,
        max_snr=20,
        lr=finetune_lr,
        save_dir=finetune_save_dir,
        snr_high=0.9,
        snr_mid=0.05,
        snr_low=0.05
    )

    # 加载最佳模型
    model.load_state_dict(torch.load(os.path.join(finetune_save_dir, 'best_improved_model.pth')))

    # 测试模型在不同SNR下的表现
    print("\n====== 模型在不同SNR下的表现 ======")
    snr_range = [20, 15, 10, 5, 0, -5, -10]
    results = {'SNR (dB)': [], 'Accuracy': []}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    create_all_visualizations(model, test_dataset, device, save_dir)
    model.eval()
    
    for snr_db in snr_range:
        correct = 0
        total = 0
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, _ = add_advanced_noise_to_batch(batch_x, min_snr=snr_db, max_snr=snr_db)
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                final_logits, _, _ = model(batch_x)
                _, predicted = torch.max(final_logits, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        acc = correct / total
        results['SNR (dB)'].append(snr_db)
        results['Accuracy'].append(acc)
        print(f"SNR {snr_db} dB - Accuracy: {acc:.4f}")

    # 保存结果
    pd.DataFrame(results).to_csv(os.path.join(finetune_save_dir, 'final_snr_performance.csv'), index=False)

    # 绘制准确率-SNR曲线
    plt.figure(figsize=(7,5))
    plt.plot(results['SNR (dB)'], results['Accuracy'], 'o-', linewidth=2)
    plt.xlabel('SNR (dB)')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy vs SNR (Grouped Features)')
    plt.grid(True)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(finetune_save_dir, 'final_snr_performance.png'), dpi=300)
    plt.close()
    print(f"结果已保存到: {finetune_save_dir}")

# 如果作为脚本执行，运行主函数
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="训练改进的ADS-B信号识别集成模型(分组特征版本)")
    parser.add_argument('--data', type=str, required=True, help='分组降维后的数据文件路径')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--save_dir', type=str, default='./improved_results', help='保存结果的目录')
    parser.add_argument('--subset_ratio', type=float, default=0.5, help='训练集采样比例，0~1之间')
    parser.add_argument('--snr_high', type=float, default=0.90, help='极高SNR采样权重(15~20dB)')
    parser.add_argument('--snr_mid', type=float, default=0.09, help='高SNR采样权重(5~15dB)')
    parser.add_argument('--snr_low', type=float, default=0.01, help='低/中SNR采样权重(-10~5dB)')
    parser.add_argument('--min_snr', type=int, default=-10, help='最小SNR(dB)')
    parser.add_argument('--max_snr', type=int, default=20, help='最大SNR(dB)')
    parser.add_argument('--finetune_epochs', type=int, default=15, help='Fine-tune阶段的训练轮数')
    parser.add_argument('--epochs', type=int, default=30, help='训练总轮数')
    
    args = parser.parse_args()
    
    # 运行主函数
    main(args.data, args.batch_size, args.save_dir, args.subset_ratio, 
         args.snr_high, args.snr_mid, args.snr_low, args.min_snr, args.max_snr, 
         args.finetune_epochs, args.epochs)