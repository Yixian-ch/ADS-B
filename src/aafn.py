import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import math
from sklearn.manifold import TSNE

torch.manual_seed(42)
np.random.seed(42)

def load_and_preprocess_data(data_path, test_size=0.2, val_size=0.1):
    """加载和预处理分组降维后的数据（适配新结构）"""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # 新结构：瞬态特征+稳态特征+target+individual_id+route
    transient_cols = [f'transient_component_{i}' for i in range(1, 16)]
    steady_cols = [f'steady_component_{i}' for i in range(1, 16)]
    feature_cols = transient_cols + steady_cols
    X = df[feature_cols].values.astype(np.float32)
    
    if 'target' in df.columns:
        label_col = 'target'
    elif 'individual_id' in df.columns:
        label_col = 'individual_id'
    else:
        raise ValueError("找不到标签列")
    
    unique_labels = sorted(df[label_col].unique())
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    y = df[label_col].map(label_map).values.astype(np.int64)
    
    print(f"特征维度: {X.shape}, 标签数量: {len(unique_labels)}")
    
    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = np.clip(X, -3, 3)
    
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp)
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    return train_dataset, val_dataset, test_dataset, len(unique_labels), label_map, scaler

def add_advanced_noise_to_batch(batch_x, min_snr=-10, max_snr=20, 
                               snr_high=0.85, snr_mid=0.14, snr_low=0.01):
    """更智能的噪声添加，重点训练高SNR"""
    batch_size = batch_x.size(0)
    noisy_x = torch.zeros_like(batch_x)
    
    # SNR区间权重（极度偏向高SNR，模仿build_network成功经验）
    snr_weights = {
        (15, 20): snr_high,   # 高SNR - 85%
        (5, 15): snr_mid,     # 中SNR - 14%
        (min_snr, 5): snr_low # 低SNR - 1%
    }
    
    def add_complex_noise(signal, snr_db):
        """添加复杂噪声模型"""
        signal_numpy = signal.detach().cpu().numpy()
        signal_power = np.mean(signal_numpy ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), signal_numpy.shape)
        
        # 对于低SNR，添加额外的噪声类型
        if snr_db < 0:
            # 脉冲噪声
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

class BuildInspiredAAFN(nn.Module):
    """完全参考build_network成功架构的AAFN"""
    def __init__(self, input_dim, num_classes, hidden_dim=None):
        super(BuildInspiredAAFN, self).__init__()
        self.num_classes = num_classes
        
        # 自动计算隐藏层维度
        if hidden_dim is None:
            # 根据输入维度和类别数自动计算合适的隐藏层维度
            hidden_dim = max(128, min(512, input_dim * 4))
        
        # 自动计算dropout率
        dropout_rate = min(0.3, max(0.1, 1.0 / (input_dim ** 0.5)))
        
        # 自动计算注意力头数，确保能被embed_dim整除
        num_heads = 1
        for i in range(8, 0, -1):
            if num_classes % i == 0:
                num_heads = i
                break
        
        # 自动计算SNR缩放因子
        self.snr_scale = min(50.0, max(20.0, num_classes * 0.3))
        
        print(f"自动适配参数:")
        print(f"- 隐藏层维度: {hidden_dim}")
        print(f"- Dropout率: {dropout_rate:.3f}")
        print(f"- 注意力头数: {num_heads}")
        print(f"- SNR缩放因子: {self.snr_scale:.1f}")
        
        # 完全参考build_network的特征处理方式
        self.feature_processor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # 瞬态和稳态特征分离（build_network的核心思想）
        self.transient_extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        self.steady_extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # 融合特征处理
        self.fused_extractor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 参考build_network的两个主要分支 - 使用更先进的架构
        self.vision_transformer = self._build_vision_transformer_path(hidden_dim * 2, num_classes, dropout_rate)
        self.lightweight_network = self._build_lightweight_path(hidden_dim * 2, num_classes, dropout_rate)
        
        # 简化的SNR估计器（参考build_network）
        self.snr_estimator_freq = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LeakyReLU()
        )
        
        self.snr_estimator_energy = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU()
        )
        
        self.snr_fusion = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim // 8, 1),
            nn.Tanh()
        )
        
        # 权重分配网络：只输出2个分支权重
        self.weight_net_fc1 = nn.Linear(1, hidden_dim // 4)
        self.weight_net_fc2 = nn.Linear(hidden_dim // 4, hidden_dim // 8)
        self.weight_net_out = nn.Linear(hidden_dim // 8, 2)
        
        # 融合层（使用更先进的Cross-Attention）
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=num_classes, num_heads=num_heads, dropout=dropout_rate, batch_first=True
        )
        self.fusion_fc = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self._initialize_weights()
    
    def _build_vision_transformer_path(self, input_dim, num_classes, dropout_rate):
        """构建Vision Transformer风格的路径"""
        # 自动计算中间层维度
        mid_dim = max(256, min(512, input_dim * 2))
        
        return nn.Sequential(
            # 输入投影
            nn.Linear(input_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            # 多头自注意力层
            nn.TransformerEncoderLayer(
                d_model=mid_dim, nhead=8, dim_feedforward=mid_dim * 2,
                dropout=dropout_rate, activation='gelu', batch_first=True,
                norm_first=True  # Pre-LN for better stability
            ),
            
            # 降维和输出
            nn.Linear(mid_dim, mid_dim // 2),
            nn.LayerNorm(mid_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mid_dim // 2, num_classes)
        )
    
    def _build_lightweight_path(self, input_dim, num_classes, dropout_rate):
        """构建增强版轻量级注意力网络路径"""
        class EnhancedAttention(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.norm1 = nn.LayerNorm(dim)
                self.norm2 = nn.LayerNorm(dim)
                
                # 多头自注意力
                self.attention = nn.MultiheadAttention(
                    embed_dim=dim,
                    num_heads=4,
                    dropout=dropout_rate,
                    batch_first=True
                )
                
                # 前馈网络
                self.ffn = nn.Sequential(
                    nn.Linear(dim, dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(dim * 2, dim),
                    nn.Dropout(dropout_rate)
                )
            
            def forward(self, x):
                # 自注意力
                attn_output, _ = self.attention(
                    self.norm1(x).unsqueeze(1),
                    self.norm1(x).unsqueeze(1),
                    self.norm1(x).unsqueeze(1)
                )
                x = x + attn_output.squeeze(1)
                
                # 前馈网络
                x = x + self.ffn(self.norm2(x))
                return x
        
        return nn.Sequential(
            # 特征提取
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            # 增强版注意力层
            EnhancedAttention(input_dim // 2),
            
            # 特征增强
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.LayerNorm(input_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            # 残差连接
            nn.Sequential(
                nn.Linear(input_dim // 4, input_dim // 4),
                nn.LayerNorm(input_dim // 4),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(input_dim // 4, input_dim // 4)
            ),
            
            # 输出层
            nn.Linear(input_dim // 4, num_classes)
        )
    
    def _initialize_weights(self):
        """初始化权重"""
        with torch.no_grad():
            nn.init.xavier_normal_(self.weight_net_fc1.weight)
            nn.init.xavier_normal_(self.weight_net_fc2.weight)
            nn.init.xavier_normal_(self.weight_net_out.weight)
            self.weight_net_out.bias.fill_(0.0)
    
    def _compute_adaptive_weights(self, snr):
        """计算自适应权重"""
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
        processed_features = self.feature_processor(x)  # [batch_size, hidden_dim * 2]
        
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
        snr_combined = torch.cat([snr_freq, snr_energy], dim=1)
        snr = self.snr_fusion(snr_combined) * self.snr_scale
        
        # 获取两个先进模型的输出
        # Vision Transformer路径
        vit_input = features.unsqueeze(1)  # 添加序列维度 [batch, 1, dim]
        vit_features = vit_input
        # 通过TransformerEncoderLayer（它在Sequential中）
        for layer in self.vision_transformer:
            if isinstance(layer, nn.TransformerEncoderLayer):
                vit_features = layer(vit_features)
            else:
                vit_features = layer(vit_features.squeeze(1) if vit_features.dim() > 2 else vit_features)
                if isinstance(layer, nn.Linear) and layer.out_features == self.num_classes:
                    break
                if vit_features.dim() == 2:
                    vit_features = vit_features.unsqueeze(1)
        
        vit_logits = vit_features.squeeze(1) if vit_features.dim() > 2 else vit_features
        
        # 轻量级网络路径
        lw_logits = self.lightweight_network(features)
        
        # 计算自适应权重
        weights = self._compute_adaptive_weights(snr)
        
        # 加权融合
        weighted_vit = vit_logits * weights[:, 0].unsqueeze(1)
        weighted_lw = lw_logits * weights[:, 1].unsqueeze(1)
        
        # Cross-Attention融合（更先进的融合方式）
        # 准备输入：[batch, 2, num_classes]
        fusion_input = torch.stack([weighted_vit, weighted_lw], dim=1)
        
        # Cross-attention
        attn_output, _ = self.cross_attention(fusion_input, fusion_input, fusion_input)
        
        # 池化并输出
        fusion_pooled = attn_output.mean(dim=1)  # [batch, num_classes]
        final_logits = self.fusion_fc(fusion_pooled)
        
        return final_logits, snr, weights

class ImprovedAdaptiveWeightedLoss(nn.Module):
    """改进的自适应加权损失函数（适配先进网络）"""
    def __init__(self, num_classes, snr_guidance_weight=0.05):
        super(ImprovedAdaptiveWeightedLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction='none', label_smoothing=0.05)
        self.num_classes = num_classes
        self.snr_guidance_weight = snr_guidance_weight
    
    def forward(self, final_logits, vit_logits, lw_logits, targets, snr, model_weights):
        vit_loss = self.ce_loss(vit_logits, targets).mean()
        lw_loss = self.ce_loss(lw_logits, targets).mean()
        ensemble_loss = self.ce_loss(final_logits, targets).mean()
        
        normalized_snr = torch.clamp(snr / 30.0, -1.0, 1.0)
        
        # 对高SNR引导ViT分支（ViT在高SNR下通常表现更好）
        high_snr_mask = (normalized_snr > 0.3).squeeze()
        high_snr_guidance = torch.zeros_like(ensemble_loss)
        if high_snr_mask.any():
            vit_weight_loss = F.relu(0.7 - model_weights[high_snr_mask, 0]).mean()
            high_snr_guidance = vit_weight_loss * self.snr_guidance_weight
        
        # 对低SNR引导轻量级网络分支（轻量级网络在复杂非线性场景下表现更好）
        low_snr_mask = (normalized_snr < -0.3).squeeze()
        low_snr_guidance = torch.zeros_like(ensemble_loss)
        if low_snr_mask.any():
            lw_weight_loss = F.relu(0.7 - model_weights[low_snr_mask, 1]).mean()
            low_snr_guidance = lw_weight_loss * self.snr_guidance_weight
        
        total_loss = ensemble_loss + 0.1 * (vit_loss + lw_loss) + high_snr_guidance + low_snr_guidance
        
        component_losses = {
            'vit_loss': vit_loss.item(),
            'lw_loss': lw_loss.item(),
            'ensemble_loss': ensemble_loss.item(),
            'weighted_loss': total_loss.item()
        }
        
        return total_loss, component_losses

class NoiseDataLoader:
    """添加噪声的数据加载器（完全参考build_network策略）"""
    def __init__(self, dataset, batch_size=32, shuffle=True, 
                noise_prob=0.5, min_snr=-10, max_snr=20, 
                snr_high=0.90, snr_mid=0.08, snr_low=0.02):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.noise_prob = noise_prob
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.snr_high = snr_high
        self.snr_mid = snr_mid
        self.snr_low = snr_low
        
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
                min_snr=self.min_snr, max_snr=self.max_snr,
                snr_high=self.snr_high, snr_mid=self.snr_mid, snr_low=self.snr_low
            )
            batch_snr = torch.tensor(actual_snrs).float().unsqueeze(1)
        else:
            # 如果不添加噪声，假设SNR为最大值
            batch_snr.fill_(self.max_snr)
        
        return batch_x, batch_y, batch_snr
    
    def __len__(self):
        return len(self.dataloader)

def train_build_inspired_aafn(model, train_dataset, val_dataset, num_epochs=40, 
                            batch_size=None, lr=None, save_dir='./build_inspired_results'):
    """训练build_network风格的AAFN模型"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    model = model.to(device)
    
    # 自动计算批次大小
    if batch_size is None:
        batch_size = min(32, max(16, len(train_dataset) // 100))  # 减小批次大小
    
    # 自动计算学习率
    if lr is None:
        lr = 0.0005 * (batch_size / 32) ** 0.5  # 降低基础学习率
    
    # 自动计算早停耐心值
    early_stop_patience = max(10, min(20, num_epochs // 3))
    
    # 自动计算最小学习率比例
    min_lr_ratio = max(0.01, min(0.1, 1.0 / (num_epochs ** 0.5)))
    
    print(f"自动适配训练参数:")
    print(f"- 批次大小: {batch_size}")
    print(f"- 学习率: {lr:.6f}")
    print(f"- 早停耐心值: {early_stop_patience}")
    print(f"- 最小学习率比例: {min_lr_ratio:.3f}")
    
    # 定义损失函数和优化器
    criterion = ImprovedAdaptiveWeightedLoss(num_classes=model.num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-4)  # 增加权重衰减
    
    # 使用带预热的余弦退火调度器
    warmup_epochs = 5
    total_steps = len(train_dataset) // batch_size * num_epochs
    warmup_steps = len(train_dataset) // batch_size * warmup_epochs
    
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 创建带噪声的数据加载器
    train_loader = NoiseDataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        noise_prob=0.5,  # 降低噪声概率
        min_snr=-10,     # 调整SNR范围
        max_snr=20,
        snr_high=0.90,   # 增加高SNR样本比例
        snr_mid=0.08,
        snr_low=0.02
    )
    
    # 创建验证数据加载器
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # 最佳模型追踪
    best_val_acc = 0.0
    best_val_loss = float('inf')  # 添加最佳验证损失追踪
    patience = early_stop_patience
    early_stop_counter = 0
    
    print(f"开始训练build_network风格的AAFN模型...")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_x, batch_y, batch_snr in pbar:
            batch_x, batch_y, batch_snr = (
                batch_x.to(device), 
                batch_y.to(device),
                batch_snr.to(device)
            )
            
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
                
                # ViT分支输出
                vit_input = features.unsqueeze(1)
                vit_features = vit_input
                for layer in model.vision_transformer:
                    if isinstance(layer, nn.TransformerEncoderLayer):
                        vit_features = layer(vit_features)
                    else:
                        vit_features = layer(vit_features.squeeze(1) if vit_features.dim() > 2 else vit_features)
                        if isinstance(layer, nn.Linear) and layer.out_features == model.num_classes:
                            break
                        if vit_features.dim() == 2:
                            vit_features = vit_features.unsqueeze(1)
                vit_logits = vit_features.squeeze(1) if vit_features.dim() > 2 else vit_features
                
                # 轻量级网络输出
                lw_logits = model.lightweight_network(features)
            
            # 损失函数调用
            loss, component_losses = criterion(final_logits, vit_logits, lw_logits, batch_y, snr, model_weights)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # 减小梯度裁剪阈值
            
            # 优化步骤
            optimizer.step()
            scheduler.step()
            
            # 统计
            train_loss += loss.item() * batch_y.size(0)
            _, predicted = torch.max(final_logits, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{train_correct/train_total:.4f}'
            })
        
        # 计算平均训练损失和准确率
        train_loss = train_loss / train_total
        train_accuracy = train_correct / train_total
        
        # 验证阶段
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
                
                # ViT分支输出
                vit_input = features.unsqueeze(1)
                vit_features = vit_input
                for layer in model.vision_transformer:
                    if isinstance(layer, nn.TransformerEncoderLayer):
                        vit_features = layer(vit_features)
                    else:
                        vit_features = layer(vit_features.squeeze(1) if vit_features.dim() > 2 else vit_features)
                        if isinstance(layer, nn.Linear) and layer.out_features == model.num_classes:
                            break
                        if vit_features.dim() == 2:
                            vit_features = vit_features.unsqueeze(1)
                vit_logits = vit_features.squeeze(1) if vit_features.dim() > 2 else vit_features
                
                # 轻量级网络输出
                lw_logits = model.lightweight_network(features)
                
                # 损失函数调用
                loss, _ = criterion(final_logits, vit_logits, lw_logits, batch_y, snr, model_weights)
                
                # 统计
                val_loss += loss.item() * batch_y.size(0)
                _, predicted = torch.max(final_logits, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        # 计算平均验证损失和准确率
        val_loss = val_loss / val_total
        val_accuracy = val_correct / val_total
        
        # 保存训练历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_accuracy)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_accuracy)
        
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        
        # 保存最佳模型（同时考虑准确率和损失）
        if val_accuracy > best_val_acc or (val_accuracy == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_accuracy
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_build_inspired_aafn.pth'))
            print(f"保存新的最佳模型，验证准确率: {val_accuracy:.4f}, 验证损失: {val_loss:.4f}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"早停触发！验证集性能连续{patience}次未提升，提前终止训练。")
                break
    
    # 加载最佳模型
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_build_inspired_aafn.pth')))
    
    return model, best_val_acc, history

def test_snr_performance(model, test_dataset, batch_size=64, device='cuda'):
    """测试不同SNR下的性能"""
    
    snr_range = [20, 15, 10, 5, 0, -5, -10]
    
    model.eval()
    results = {'snr_db': snr_range, 'accuracy': [], 'weights': []}
    
    print("🎯 测试build_network风格AAFN性能...")
    
    for snr_db in snr_range:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        correct = 0
        total = 0
        all_weights = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                # 确保输入维度正确
                if batch_x.dim() == 1:
                    batch_x = batch_x.unsqueeze(0)
                if batch_y.dim() == 0:
                    batch_y = batch_y.unsqueeze(0)
                
                # 只在测试时添加指定SNR的噪声
                if snr_db < 20:
                    batch_x, _ = add_advanced_noise_to_batch(
                        batch_x, min_snr=snr_db, max_snr=snr_db
                    )
                
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                logits, _, weights = model(batch_x)
                _, predicted = torch.max(logits, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
                all_weights.append(weights.cpu().numpy())
        
        accuracy = correct / total
        avg_weights = np.mean(np.concatenate(all_weights), axis=0)
        
        results['accuracy'].append(accuracy)
        results['weights'].append(avg_weights.tolist())
        
        print(f"SNR {snr_db:3d} dB: {accuracy:.4f}")
        print(f"   权重 - ViT: {avg_weights[0]:.3f}, 轻量级网络: {avg_weights[1]:.3f}")
    
    return results

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
            _ = model(batch_x)
            features_list.append(batch_x.cpu())  # 这里假设直接用输入特征，如需中间层可改
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
    import matplotlib.pyplot as plt
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
    if n_classes <= 30:
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

def save_snr_performance_plot_and_csv(results, save_dir):
    import matplotlib.pyplot as plt
    import pandas as pd
    os.makedirs(save_dir, exist_ok=True)
    # 保存csv
    df = pd.DataFrame({'SNR (dB)': results['snr_db'], 'Accuracy': results['accuracy']})
    csv_path = os.path.join(save_dir, 'final_snr_performance.csv')
    df.to_csv(csv_path, index=False)
    # 绘图
    plt.figure(figsize=(7,5))
    plt.plot(results['snr_db'], results['accuracy'], 'o-', linewidth=2)
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
    import matplotlib.pyplot as plt
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

def create_confusion_matrix(model, test_dataset, save_dir, device):
    """生成并保存混淆矩阵"""
    print("🎯 生成混淆矩阵...")
    model.eval()
    all_predictions = []
    all_labels = []
    
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            logits, _, _ = model(batch_x)
            _, predicted = torch.max(logits, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.numpy())
    
    # 计算混淆矩阵
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(all_labels, all_predictions)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(len(np.unique(all_labels))),
                yticklabels=range(len(np.unique(all_labels))))
    plt.title('Confusion Matrix for Individual Aircraft Identification', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    # 保存混淆矩阵图
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 混淆矩阵已保存: {save_path}")
    
    # 计算并打印一些评估指标
    from sklearn.metrics import classification_report
    report = classification_report(all_labels, all_predictions, output_dict=True)
    
    # 保存评估报告
    import json
    report_path = os.path.join(save_dir, 'classification_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ 分类报告已保存: {report_path}")
    
    return cm, report

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="训练build_network风格的AAFN模型")
    parser.add_argument('--data', type=str, required=True, help='数据文件路径')
    parser.add_argument('--epochs', type=int, default=40, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=None, help='批次大小（None表示自动计算）')
    parser.add_argument('--lr', type=float, default=None, help='学习率（None表示自动计算）')
    parser.add_argument('--save_dir', type=str, default='./build_inspired_aafn_results', help='保存目录')
    parser.add_argument('--hidden_dim', type=int, default=None, help='隐藏层维度（None表示自动计算）')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 使用设备: {device}")
    
    # 加载数据
    train_dataset, val_dataset, test_dataset, num_classes, label_map, scaler = load_and_preprocess_data(args.data)
    
    sample_x, _ = train_dataset[0]
    input_dim = sample_x.size(0)
    
    print(f"输入维度: {input_dim}, 类别数量: {num_classes}")
    
    # 创建build_network风格的模型
    model = BuildInspiredAAFN(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=args.hidden_dim
    )
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 训练模型
    model, best_val_acc, history = train_build_inspired_aafn(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.save_dir
    )
    
    # 测试性能
    results = test_snr_performance(model, test_dataset, batch_size=args.batch_size, device=device)
    
    # 保存SNR性能曲线和CSV
    save_snr_performance_plot_and_csv(results, args.save_dir)
    # 保存训练历史曲线
    save_training_history_plot(history, args.save_dir)
    # 保存t-SNE可视化
    create_individual_identification_plot(model, test_dataset, args.save_dir, device)
    # 生成并保存混淆矩阵
    create_confusion_matrix(model, test_dataset, args.save_dir, device)
    
    # 绘制结果对比图
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(results['snr_db'], results['accuracy'], 'ro-', linewidth=3, markersize=8, label='Build-style AAFN')
    plt.axhline(y=0.95, color='g', linestyle='--', linewidth=2, label='95% Target Line')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Accuracy')
    plt.title('Build-style AAFN Performance')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0.5, 1.05)
    
    plt.subplot(2, 1, 2)
    weights_array = np.array(results['weights'])
    plt.plot(results['snr_db'], weights_array[:, 0], 'ro-', label='ViT Weight', linewidth=2)
    plt.plot(results['snr_db'], weights_array[:, 1], 'bo-', label='Lightweight Network Weight', linewidth=2)
    plt.xlabel('SNR (dB)')
    plt.ylabel('Weight')
    plt.title('Advanced Network Branch Weights Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'build_inspired_aafn_performance.png'), dpi=300)
    plt.close()
    
    # 保存结果
    import json
    with open(os.path.join(args.save_dir, 'build_inspired_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n结果已保存到: {args.save_dir}")

if __name__ == "__main__":
    main()