import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import random
from sklearn.manifold import TSNE

# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

# 网络定义部分
class TemporalBlock(nn.Module):
    """时间卷积网络的基本块"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(self.conv1, self.bn1, nn.GELU(), self.dropout1,
                               self.conv2, self.bn2, nn.GELU(), self.dropout2)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.GELU()
        
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        if out.size() != res.size():
            res = F.interpolate(res, size=out.size()[2:], mode='linear', align_corners=False)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    """时间卷积网络"""
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size,
                                   stride=1, dilation=dilation,
                                   padding=(kernel_size-1) * dilation,
                                   dropout=dropout)]
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class GraphConvLayer(nn.Module):
    """图卷积层"""
    def __init__(self, in_channels, out_channels):
        super(GraphConvLayer, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        
    def forward(self, x, edge_index=None):
        if edge_index is None:
            batch_size = x.size(0)
            num_nodes = x.size(1)
            edge_index = torch.ones(batch_size, num_nodes, num_nodes).to(x.device)
        
        x = self.linear(x)
        x = self.norm(x)
        return F.gelu(x)

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, context=None):
        B, N, C = x.shape
        
        if context is None:
            # 自注意力
            context = x
            
        # 投影查询、键和值
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(context).reshape(B, context.size(1), self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(context).reshape(B, context.size(1), self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 计算注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # 应用注意力
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        return x

class TransformerEncoder(nn.Module):
    """Transformer编码器"""
    def __init__(self, d_model, nhead, dim_feedforward, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x):
        x = x + self.dropout1(self.self_attn(self.norm1(x)))
        x = x + self.dropout2(self.linear2(self.dropout(F.gelu(self.linear1(self.norm2(x))))))
        return x

class EnhancedTransientPath(nn.Module):
    """增强的瞬态特征处理路径"""
    def __init__(self, input_dim=10, hidden_dim=128):
        super(EnhancedTransientPath, self).__init__()
        
        self.feature_expansion = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # TCN处理时序特征
        self.tcn = TemporalConvNet(
            num_inputs=hidden_dim,
            num_channels=[hidden_dim, hidden_dim, hidden_dim],
            kernel_size=3,
            dropout=0.1
        )
        
        # Transformer处理全局关系
        self.transformer = TransformerEncoder(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            num_layers=2,
            dropout=0.1
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
    def forward(self, x):
        # 特征扩展
        x = self.feature_expansion(x)  # [batch_size, hidden_dim]
        
        # TCN处理
        x_tcn = x.unsqueeze(-1)  # [batch_size, hidden_dim, 1]
        x_tcn = self.tcn(x_tcn)  # [batch_size, hidden_dim, 1]
        x_tcn = x_tcn.mean(dim=-1)  # [batch_size, hidden_dim]
        
        # Transformer处理
        x_trans = x.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        x_trans = self.transformer(x_trans)  # [batch_size, 1, hidden_dim]
        x_trans = x_trans.squeeze(1)  # [batch_size, hidden_dim]
        
        # 特征融合
        x = torch.cat([x_tcn, x_trans], dim=-1)  # [batch_size, hidden_dim*2]
        x = self.fusion(x)  # [batch_size, hidden_dim]
        return x

class EnhancedSteadyPath(nn.Module):
    """增强的稳态特征处理路径"""
    def __init__(self, input_dim=10, hidden_dim=128):
        super(EnhancedSteadyPath, self).__init__()
        
        self.feature_expansion = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # GNN处理特征关系
        self.gnn = GraphConvLayer(
            in_channels=hidden_dim,
            out_channels=hidden_dim
        )
        
        # Transformer处理全局关系
        self.transformer = TransformerEncoder(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            num_layers=2,
            dropout=0.1
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
    def forward(self, x):
        # 特征扩展
        x = self.feature_expansion(x)  # [batch_size, hidden_dim]
        
        # GNN处理
        x_gnn = x.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        x_gnn = self.gnn(x_gnn)  # [batch_size, 1, hidden_dim]
        x_gnn = x_gnn.squeeze(1)  # [batch_size, hidden_dim]
        
        # Transformer处理
        x_trans = x.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        x_trans = self.transformer(x_trans)  # [batch_size, 1, hidden_dim]
        x_trans = x_trans.squeeze(1)  # [batch_size, hidden_dim]
        
        # 特征融合
        x = torch.cat([x_gnn, x_trans], dim=-1)  # [batch_size, hidden_dim*2]
        x = self.fusion(x)  # [batch_size, hidden_dim]
        return x

class CrossAttentionFusion(nn.Module):
    """交叉注意力融合模块"""
    def __init__(self, dim, num_heads=4):
        super(CrossAttentionFusion, self).__init__()
        self.attention1 = MultiHeadAttention(dim, num_heads)
        self.attention2 = MultiHeadAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, x1, x2):
        if len(x1.shape) == 2:
            x1 = x1.unsqueeze(1)  # [batch_size, 1, dim]
        if len(x2.shape) == 2:
            x2 = x2.unsqueeze(1)  # [batch_size, 1, dim]
            
        # 交叉注意力
        x1 = x1 + self.attention1(self.norm1(x1), self.norm2(x2))
        x2 = x2 + self.attention2(self.norm2(x2), self.norm1(x1))
        
        # 压缩维度并连接
        x1 = x1.squeeze(1)  # [batch_size, dim]
        x2 = x2.squeeze(1)  # [batch_size, dim]
        return torch.cat([x1, x2], dim=-1)  # [batch_size, dim*2]

class EnhancedADSBNetwork(nn.Module):
    """改进的ADS-B信号识别网络"""
    def __init__(self, input_dim=20, num_classes=107, hidden_dim=128):
        super(EnhancedADSBNetwork, self).__init__()
        
        # 增强特征提取
        self.feature_processor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # 添加噪声抑制层
        self.noise_suppression = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.transient_path = EnhancedTransientPath(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim
        )
        
        self.steady_path = EnhancedSteadyPath(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim
        )
        
        self.fusion = CrossAttentionFusion(
            dim=hidden_dim,
            num_heads=4
        )
        
        # 增强分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # 改进SNR估计器
        self.snr_estimator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # 特征处理
        x = self.feature_processor(x)
        
        # 噪声抑制
        x = self.noise_suppression(x)
        
        trans_features = x[:, :x.size(1)//2]
        steady_features = x[:, x.size(1)//2:]
        
        trans_out = self.transient_path(trans_features)
        steady_out = self.steady_path(steady_features)
        
        fused_features = self.fusion(trans_out, steady_out)
        
        # SNR估计（范围扩大到[-40, 40]）
        snr = self.snr_estimator(fused_features) * 40
        
        logits = self.classifier(fused_features)
        
        return logits, snr

class SNRWeightedDataset(Dataset):
    """基于SNR加权的数据集"""
    def __init__(self, features, labels, snr_high_range=(15, 20), snr_low_range=(-10, 15), high_snr_ratio=0.9):
        self.features = features
        self.labels = labels
        self.snr_high_range = snr_high_range
        self.snr_low_range = snr_low_range
        self.high_snr_ratio = high_snr_ratio
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # 根据概率选择SNR范围
        if random.random() < self.high_snr_ratio:
            # 90%的概率使用高SNR (15-20dB)
            snr = random.uniform(self.snr_high_range[0], self.snr_high_range[1])
        else:
            # 10%的概率使用低SNR (-10-15dB)
            snr = random.uniform(self.snr_low_range[0], self.snr_low_range[1])
        
        # 添加噪声
        noise_power = 10 ** (-snr / 10)
        noise = torch.randn_like(self.features[idx]) * torch.sqrt(torch.tensor(noise_power))
        noisy_features = self.features[idx] + noise
        
        return noisy_features, self.labels[idx], snr

# 数据加载和预处理函数
def load_and_preprocess_data(data_path, test_size=0.2, val_size=0.1):
    """加载和预处理分组降维后的数据（适配新结构）"""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    # 新结构：瞬态特征+稳态特征+target+individual_id+route
    transient_cols = [f'transient_component_{i}' for i in range(1, 16)]
    steady_cols = [f'steady_component_{i}' for i in range(1, 16)]
    feature_cols = transient_cols + steady_cols
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少特征列: {missing_cols}")
    X = df[feature_cols].values.astype(np.float32)
    if 'target' in df.columns:
        label_col = 'target'
    elif 'individual_id' in df.columns:
        label_col = 'individual_id'
    else:
        raise ValueError("找不到标签列，期望 'target' 或 'individual_id'")
    unique_labels = sorted(df[label_col].unique())
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    y = df[label_col].map(label_map).values.astype(np.int64)
    print(f"特征维度: {X.shape}")
    print(f"标签数量: {len(unique_labels)}")
    print(f"标签分布: {np.bincount(y)}")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
    )
    print(f"数据分割完成. 训练集: {len(y_train)}, 验证集: {len(y_val)}, 测试集: {len(y_test)}")
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
    return train_dataset, val_dataset, test_dataset, len(unique_labels), label_map, scaler

# 训练函数
def train_model(model, train_loader, val_loader, num_epochs=100, lr=0.001, weight_decay=1e-5,
               patience=10, checkpoint_path='best_model.pth', logger=None):
    """训练模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=lr/20
    )
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    logger.info("Starting training...")
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y, batch_snr in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            logits, estimated_snr = model(batch_x)
            
            # 计算分类损失
            classification_loss = criterion(logits, batch_y)
            
            # 计算SNR估计损失
            snr_loss = F.mse_loss(estimated_snr.squeeze(), torch.tensor(batch_snr, dtype=torch.float32).to(device))
            
            # 总损失
            loss = classification_loss + 0.1 * snr_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(logits.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        train_loss = train_loss / train_total
        train_accuracy = train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_accuracy)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                logits, snr = model(batch_x)
                loss = criterion(logits, batch_y)
                
                val_loss += loss.item() * batch_x.size(0)
                _, predicted = torch.max(logits.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        val_loss = val_loss / val_total
        val_accuracy = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_accuracy)
        
        scheduler.step()
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                   f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, "
                   f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Saved new best model with validation loss: {val_loss:.4f}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    model.load_state_dict(torch.load(checkpoint_path))
    
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accs,
        'val_acc': val_accs
    }
    
    return model, history

# 评估函数
def evaluate_model(model, test_loader, label_map=None, logger=None):
    """评估模型性能"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_snrs = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            logits, snr = model(batch_x)
            _, predictions = torch.max(logits, 1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
            all_snrs.extend(snr.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_snrs = np.array(all_snrs)
    
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    accuracy = accuracy_score(all_targets, all_predictions)
    cm = confusion_matrix(all_targets, all_predictions)
    
    logger.info(f"测试准确率: {accuracy:.4f}")
    logger.info("\n分类报告:")
    logger.info(classification_report(all_targets, all_predictions))
    
    return accuracy, cm, all_predictions, all_snrs

def test_with_different_snr(model, test_loader, snr_range=[20, 15, 10, 5, 0, -5, -10], logger=None):
    """在不同SNR下测试模型性能"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    results = {}
    
    for snr in snr_range:
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                # 添加高斯噪声
                noise_power = 10 ** (-snr / 10)
                noise = torch.randn_like(batch_x) * torch.sqrt(torch.tensor(noise_power))
                noisy_batch_x = batch_x + noise
                
                logits, _ = model(noisy_batch_x)
                _, predictions = torch.max(logits, 1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        accuracy = accuracy_score(all_targets, all_predictions)
        results[snr] = {'accuracy': accuracy}
        
        # 只输出SNR和准确率
        print(f"SNR {snr:3d} dB - Accuracy: {accuracy:.4f}")
    
    return results

def plot_snr_test_results(snr_results, save_dir):
    """绘制SNR测试结果"""
    snrs = list(snr_results.keys())
    accuracies = [results['accuracy'] for results in snr_results.values()]
    f1_scores = [results['f1'] for results in snr_results.values()]
    auc_scores = [results['auc'] for results in snr_results.values()]
    estimated_snrs = [results['mean_estimated_snr'] for results in snr_results.values()]
    
    plt.figure(figsize=(15, 10))
    
    # 准确率、F1分数和AUC
    plt.subplot(2, 1, 1)
    plt.plot(snrs, accuracies, 'b-', label='Accuracy', linewidth=2)
    plt.plot(snrs, f1_scores, 'r-', label='F1 Score', linewidth=2)
    plt.plot(snrs, auc_scores, 'g-', label='AUC Score', linewidth=2)
    plt.xlabel('SNR (dB)')
    plt.ylabel('Performance Metrics')
    plt.title('Model Performance vs SNR')
    plt.legend()
    plt.grid(True)
    plt.axhline(y=0.95, color='k', linestyle='--', label='95% Target')
    
    # 估计SNR
    plt.subplot(2, 1, 2)
    plt.plot(snrs, estimated_snrs, 'g-', label='Estimated SNR', linewidth=2)
    plt.plot(snrs, snrs, 'k--', label='Actual SNR', linewidth=2)
    plt.xlabel('Actual SNR (dB)')
    plt.ylabel('Estimated SNR (dB)')
    plt.title('SNR Estimation Performance')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'snr_test_results.png'), dpi=300, bbox_inches='tight')
    plt.close()

def extract_features_for_visualization(model, dataset, device, max_samples=2000):
    model.eval()
    features_list = []
    all_labels = []
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    sample_count = 0
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            if sample_count >= max_samples:
                break
            batch_x = batch_x.to(device)
            features_list.append(batch_x.cpu())  # 默认用输入特征
            all_labels.extend(batch_y.cpu().numpy())
            sample_count += len(batch_y)
    features = torch.cat(features_list, dim=0)[:max_samples]
    labels = np.array(all_labels)[:max_samples]
    return features.numpy(), labels

def create_individual_identification_plot(model, test_dataset, save_dir, device, max_samples=2000):
    print("🎨 生成个体识别可视化图 (t-SNE)...")
    features, labels = extract_features_for_visualization(model, test_dataset, device, max_samples)
    if features is None:
        print("❌ 特征提取失败")
        return
    print(f"特征shape: {features.shape}, 标签数量: {len(np.unique(labels))}")
    reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)//4))
    features_2d = reducer.fit_transform(features)
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    if n_classes <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    elif n_classes <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, n_classes))
    else:
        colors = plt.cm.nipy_spectral(np.linspace(0, 1, n_classes))
    plt.figure(figsize=(14, 12))
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], c=[colors[i]], label=f'Aircraft {label:03d}', s=30, alpha=0.7, edgecolors='black', linewidth=0.5)
    plt.title('Individual Aircraft Identification Visualization (TSNE)', fontsize=16, fontweight='bold')
    plt.xlabel('TSNE Component 1', fontsize=12)
    plt.ylabel('TSNE Component 2', fontsize=12)
    if n_classes <= 107:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=max(1, n_classes//20))
    else:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.nipy_spectral, norm=plt.Normalize(vmin=unique_labels.min(), vmax=unique_labels.max()))
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label('Aircraft ID', rotation=270, labelpad=15)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'individual_identification_tsne.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 个体识别图已保存: {save_path}")
    return features_2d, labels

def save_snr_performance_plot_and_csv(snr_results, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    snrs = list(snr_results.keys())
    accuracies = [snr_results[snr]['accuracy'] for snr in snrs]
    # 保存csv
    df = pd.DataFrame({'SNR (dB)': snrs, 'Accuracy': accuracies})
    csv_path = os.path.join(save_dir, 'final_snr_performance.csv')
    df.to_csv(csv_path, index=False)
    # 绘图
    plt.figure(figsize=(7,5))
    plt.plot(snrs, accuracies, 'o-', linewidth=2)
    plt.xlabel('SNR (dB)')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy vs SNR (Grouped Features)')
    plt.grid(True)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    png_path = os.path.join(save_dir, 'final_snr_performance.png')
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"SNR性能曲线和CSV已保存: {png_path}, {csv_path}")

def save_training_history_plot(history, save_dir):
    os.makedirs(save_dir, exist_ok=True)
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
    plt.tight_layout()
    png_path = os.path.join(save_dir, 'improved_training_history.png')
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"训练历史曲线已保存: {png_path}")

def main():
    parser = argparse.ArgumentParser(description="Train Enhanced ADS-B Signal Recognition Network")
    parser.add_argument('--data', type=str, required=True, help='Data file path')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--save_dir', type=str, default='./enhanced_results', help='Save directory')
    parser.add_argument('--test_snr', action='store_true', help='Whether to perform SNR testing')
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 设置日志记录器
    logger = setup_logger(args.save_dir)
    
    # 设置随机种子
    set_seed(42)
    
    # 加载数据
    train_dataset, val_dataset, test_dataset, num_classes, label_map, scaler = load_and_preprocess_data(args.data)
    
    # 创建SNR加权的训练数据集
    train_features = train_dataset.tensors[0]
    train_labels = train_dataset.tensors[1]
    weighted_train_dataset = SNRWeightedDataset(train_features, train_labels)
    
    # 创建数据加载器
    train_loader = DataLoader(weighted_train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnhancedADSBNetwork(
        input_dim=20,
        num_classes=num_classes,
        hidden_dim=args.hidden_dim
    )
    
    # 训练模型
    checkpoint_path = os.path.join(args.save_dir, 'best_model.pth')
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        checkpoint_path=checkpoint_path,
        logger=logger
    )
    
    # 评估模型
    accuracy, cm, predictions, snrs = evaluate_model(
        model=model,
        test_loader=test_loader,
        label_map=label_map,
        logger=logger
    )
    
    # 保存结果
    results = {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'predictions': predictions,
        'snrs': snrs,
        'history': history
    }
    
    # 进行SNR测试
    logger.info("Starting SNR test...")
    snr_results = test_with_different_snr(
        model=model,
        test_loader=test_loader,
        logger=logger
    )
    results['snr_test'] = snr_results
    
    # 绘制SNR测试结果
    plot_snr_test_results(snr_results, args.save_dir)
    
    # 保存SNR性能曲线和CSV
    save_snr_performance_plot_and_csv(snr_results, args.save_dir)
    # 保存训练历史曲线
    save_training_history_plot(history, args.save_dir)
    # 保存t-SNE可视化
    create_individual_identification_plot(model, test_dataset, args.save_dir, device)
    
    # 保存详细结果
    np.save(os.path.join(args.save_dir, 'results.npy'), results)
    logger.info(f"Results saved to {args.save_dir}")
    
    # 输出最佳性能
    best_snr = max(snr_results.items(), key=lambda x: x[1]['accuracy'])
    logger.info(f"\nBest performance at SNR = {best_snr[0]}dB:")
    logger.info(f"Accuracy: {best_snr[1]['accuracy']:.4f}")
    logger.info(f"F1 Score: {best_snr[1]['f1']:.4f}")
    logger.info(f"AUC Score: {best_snr[1]['auc']:.4f}")

if __name__ == "__main__":
    main() 