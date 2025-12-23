import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinuousArcFaceLoss(nn.Module):
    
    def __init__(self, in_features_dim, num_proxies=64, s=30.0,use_uniformity_reg=True,use_contrastive=True,device=None):
        super().__init__()
        self.in_features_dim = in_features_dim
        self.num_proxies = num_proxies  # 代理数量
        self.s = s
        self.device = device
        self.use_uniformity_reg = use_uniformity_reg
        self.use_contrastive = use_contrastive

        if use_uniformity_reg:
            self.uniformity_reg = UniformityRegularizer(weight=0.1)
        if use_contrastive:
            self.contrastive_loss = ContrastiveAugmentation(temperature=0.1)
                    
        # 代理对应的基准角度（均匀分布在[0, 2π]）
        self.proxy_angles = torch.linspace(0, 2*torch.pi, num_proxies).to(device)
        
    def forward(self, features,cos_theta, targets=None):
        
        loss_dict = {}
        
        if targets is not None:
            # 将连续目标值映射到角度
            # 假设targets在[0, 1]，映射到[0, 2π]
            target_angles = targets * 2 * torch.pi
            
            # 为每个目标找到最近的两个代理（用于插值）
            batch_size = cos_theta.size(0)
            losses = []
            
            for i in range(batch_size):
                # 计算目标角度与所有代理角度的距离
                angle_diff = torch.abs(self.proxy_angles - target_angles[i])
                
                # 找到最近的两个代理
                k = 2  # 使用最近邻插值
                _, indices = torch.topk(-angle_diff, k)  # 距离最小的k个
                
                # 计算与这两个代理的相似度（带温度系数）
                cos_sim = cos_theta[i, indices]
                
                # 使用距离作为权重
                weights = 1.0 / (angle_diff[indices] + 1e-8)
                weights = weights / weights.sum()
                
                # 目标：鼓励特征靠近加权代理
                # 使用softmax交叉熵，但目标是连续的权重
                target_dist = F.softmax(weights * 10.0, dim=0)  # 温度系数
                
                # 计算KL散度损失
                pred_dist = F.softmax(self.s * cos_sim, dim=0)
                loss = F.kl_div(pred_dist.log(), target_dist, reduction='sum')
                losses.append(loss)
            
            total_loss = torch.stack(losses).mean()

        if self.use_uniformity_reg:
            uniform_loss = self.uniformity_reg(features)
            total_loss += uniform_loss
            loss_dict['uniform_loss'] = uniform_loss.item()
        
        if self.use_contrastive:
            contrastive_loss = self.contrastive_loss(features, targets)
            total_loss += contrastive_loss
            loss_dict['contrastive_loss'] = contrastive_loss.item()       
                 
        return total_loss


class RobustArcFaceRegression(nn.Module):
    """具有多重防坍塌机制的ArcFace回归模型"""
    
    def __init__(self, in_features, num_proxies=36, out_dim=16,
                 use_uniformity_reg=True,
                 use_contrastive=True,
                 use_spectral_reg=False,
                 device=None):
        super().__init__()
        self.device = device
        self.out_dim = out_dim
        
        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, out_dim)  # 嵌入维度
        )
        
        # 代理向量（均匀初始化）
        self.proxies = nn.Parameter(torch.randn(num_proxies, out_dim)).to(device).double()
        self.proxy_angles = torch.linspace(0, 2*torch.pi, num_proxies).to(device).double()
        
        # 正则化器
        self.use_uniformity_reg = use_uniformity_reg
        self.use_contrastive = use_contrastive
        self.use_spectral_reg = use_spectral_reg
        
        if use_uniformity_reg:
            self.uniformity_reg = UniformityRegularizer(weight=0.1)
        if use_contrastive:
            self.contrastive_loss = ContrastiveAugmentation(temperature=0.1)
        if use_spectral_reg:
            self.spectral_reg = SpectralRegularizer(weight=0.01)
        
        # 自适应边际
        self.adaptive_margin = AdaptiveMarginArcFace(in_features)
        
        # 监控器
        self.monitor = FeatureCollapseMonitor()
        
    def forward(self, x, targets=None, epoch=0, batch_idx=0):
        # 提取特征
        features = self.feature_extractor(x)
        
        # 监控特征分布
        if self.training:
            self.monitor.monitor_step(features, epoch, batch_idx)
            
            # 动态调整边际
            margin = torch.tensor(self.adaptive_margin.update_margin_based_on_diversity(features)).to(self.device)
        else:
            margin = torch.tensor(0.1).to(self.device)
        
        # 归一化
        features_norm = F.normalize(features, p=2, dim=1)
        proxies_norm = F.normalize(self.proxies, p=2, dim=1).to(self.device)
        
        # 计算相似度
        cos_theta = F.linear(features_norm, proxies_norm)  # [batch, num_proxies]
        
        total_loss = 0
        loss_dict = {}
        
        if targets is not None:
            # 基础回归损失
            target_angles = targets * 2 * torch.pi
            
            # 找到最近代理的加权组合
            batch_size = x.size(0)
            reg_loss = 0
            
            for i in range(batch_size):
                # 计算角度距离
                angle_diff = torch.abs(self.proxy_angles.to(x.device) - target_angles[i])
                
                # 使用高斯核作为目标分布
                weights = torch.exp(-angle_diff**2 / (2 * 0.2**2))  # sigma=0.2
                weights = weights / weights.sum()
                
                # 带边际的相似度
                cos_with_margin = cos_theta[i] * torch.cos(margin) - torch.sqrt(1 - cos_theta[i]**2) * torch.sin(margin)
                
                # 损失：KL散度
                pred_dist = F.softmax(cos_with_margin * 30.0, dim=0)  # scale=30
                reg_loss += F.kl_div(pred_dist.log(), weights, reduction='sum')
            
            base_loss = reg_loss / batch_size
            total_loss += base_loss
            loss_dict['base_loss'] = base_loss.item()
            
            # 添加各种正则化项
            if self.use_uniformity_reg:
                uniform_loss = self.uniformity_reg(features)
                total_loss += uniform_loss
                loss_dict['uniform_loss'] = uniform_loss.item()
            
            if self.use_contrastive:
                contrastive_loss = self.contrastive_loss(features, targets)
                total_loss += contrastive_loss
                loss_dict['contrastive_loss'] = contrastive_loss.item()
            
            if self.use_spectral_reg:
                spectral_loss = self.spectral_reg(features)
                total_loss += spectral_loss
                loss_dict['spectral_loss'] = spectral_loss.item()
            
            # 预测值
            with torch.no_grad():
                proxy_weights = F.softmax(cos_theta * 30.0, dim=1)
                pred_angles = (proxy_weights * self.proxy_angles.to(x.device)).sum(dim=1)
                predictions = pred_angles / (2 * torch.pi)
            
            return predictions, total_loss, loss_dict
        
        # 推理模式
        proxy_weights = F.softmax(cos_theta * 30.0, dim=1)
        pred_angles = (proxy_weights * self.proxy_angles.to(x.device)).sum(dim=1)
        return pred_angles / (2 * torch.pi)
    
        
class FeatureCollapseMonitor:
    def __init__(self, check_interval=100):
        self.check_interval = check_interval
        self.angle_histories = []
        self.variance_histories = []
        
    def monitor_step(self, features, epoch, batch_idx):
        if batch_idx % self.check_interval == 0:
            features_norm = F.normalize(features, p=2, dim=1)
            
            # 计算主方向
            U, S, V = torch.svd(features_norm.T)
            explained_variance = S[0] / S.sum()
            
            # 角度方差
            angles = torch.atan2(features_norm[:, 1], features_norm[:, 0])
            angle_var = torch.var(angles)
            
            self.angle_histories.append(angle_var.item())
            self.variance_histories.append(explained_variance.item())
            
            # 警告信号
            if explained_variance > 0.8:  # 80%方差由第一主成分解释
                print(f"⚠️  Warning: High explained variance {explained_variance:.3f} at epoch {epoch}")
            if angle_var < 0.1:  # 角度方差太小
                print(f"⚠️  Warning: Low angle variance {angle_var:.3f} at epoch {epoch}")
                
class AdaptiveMarginArcFace(nn.Module):
    def __init__(self, in_features, margin_range=(0.1, 0.8)):
        super().__init__()
        self.margin_min, self.margin_max = margin_range
        self.current_margin = margin_range[0]
        
        # 监控角度方差
        self.angle_variance_history = []
        
    def update_margin_based_on_diversity(self, features, threshold=0.3):
        """根据特征多样性动态调整边际"""
        features_norm = F.normalize(features, p=2, dim=1)
        
        # 计算角度多样性
        angles = torch.atan2(features_norm[:, 1], features_norm[:, 0])
        angle_var = torch.var(angles).item()
        
        # 根据方差调整边际
        if angle_var < threshold:
            # 多样性不足，增加边际以促进分离
            self.current_margin = min(self.margin_max, self.current_margin + 0.05)
        else:
            # 多样性足够，减小边际以避免过分离
            self.current_margin = max(self.margin_min, self.current_margin - 0.02)
        
        return self.current_margin
        
class UniformityRegularizer:
    def __init__(self, weight=0.1, mode='orthogonal'):
        self.weight = weight
        self.mode = mode
        
    def __call__(self, features):
        """添加均匀性正则化损失"""
        features_norm = F.normalize(features, p=2, dim=1)
        batch_size = features.size(0)
        
        if self.mode == 'orthogonal':
            # 鼓励特征向量正交
            gram_matrix = torch.mm(features_norm, features_norm.T)
            identity = torch.eye(batch_size, device=features.device)
            orth_loss = torch.norm(gram_matrix - identity, p='fro') ** 2
            
            return self.weight * orth_loss / (batch_size ** 2)
            
        elif self.mode == 'uniform':
            # 使用均匀分布参考
            angles = torch.atan2(features_norm[:, 1], features_norm[:, 0])
            
            # 计算与均匀分布的KL散度
            # 将角度离散化到bins
            num_bins = 36
            bins = torch.linspace(-torch.pi, torch.pi, num_bins + 1)
            
            hist = torch.histc(angles, bins=num_bins, min=-torch.pi, max=torch.pi)
            hist = hist / hist.sum()  # 归一化为概率分布
            
            # 目标均匀分布
            uniform = torch.ones(num_bins, device=features.device) / num_bins
            
            # KL散度
            kl_loss = F.kl_div(hist.log(), uniform, reduction='sum')
            
            return self.weight * kl_loss
            
class AdversarialSeparationLoss(nn.Module):
    """使用对抗学习鼓励特征分离"""
    def __init__(self, feature_dim, discriminator_hidden=128):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, discriminator_hidden),
            nn.ReLU(),
            nn.Linear(discriminator_hidden, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features, targets):
        """
        特征: [batch, feature_dim]
        目标: [batch] 连续值
        """
        # 1. 生成"假"特征（通过扰动）
        noise = torch.randn_like(features) * 0.1
        fake_features = features + noise
        
        # 2. 训练判别器
        real_labels = torch.ones(features.size(0), 1, device=features.device)
        fake_labels = torch.zeros(fake_features.size(0), 1, device=features.device)
        
        real_pred = self.discriminator(features)
        fake_pred = self.discriminator(fake_features.detach())
        
        d_loss_real = F.binary_cross_entropy(real_pred, real_labels)
        d_loss_fake = F.binary_cross_entropy(fake_pred, fake_labels)
        d_loss = d_loss_real + d_loss_fake
        
        # 3. 生成器损失（鼓励特征多样性）
        # 让判别器难以区分真实特征和扰动特征
        gen_pred = self.discriminator(fake_features)
        g_loss = F.binary_cross_entropy(gen_pred, real_labels)
        
        return {
            'discriminator_loss': d_loss,
            'generator_loss': g_loss,
            'separation_loss': g_loss  # 用于特征提取器的损失
        }
        
class ContrastiveAugmentation:
    """通过对比学习防止特征坍塌"""
    def __init__(self, temperature=0.1):
        self.temperature = temperature
        
    def __call__(self, features, targets):
        """
        对比损失：鼓励相似目标有相似特征，不同目标有不同特征
        """
        batch_size = features.size(0)
        
        # 归一化特征
        features_norm = F.normalize(features, p=2, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.mm(features_norm, features_norm.T)  # [batch, batch]
        
        # 创建目标相似度矩阵（基于连续目标的相似度）
        targets_expanded = targets.unsqueeze(1)  # [batch, 1]
        target_diff = torch.abs(targets_expanded - targets_expanded.T)  # [batch, batch]
        
        # 使用高斯核将差异转换为相似度
        # 相似目标 → 高相似度，不同目标 → 低相似度
        target_similarity = torch.exp(-target_diff ** 2 / (2 * 0.1 ** 2))  # sigma=0.1
        
        # 对角线置零（自己和自己不算）
        mask = torch.eye(batch_size, device=features.device).bool()
        target_similarity = target_similarity.masked_fill(mask, 0)
        
        # 计算对比损失
        # 鼓励特征相似度匹配目标相似度
        pos_weight = target_similarity / target_similarity.sum(dim=1, keepdim=True)
        
        # InfoNCE风格损失
        logits = similarity_matrix / self.temperature
        logits = logits.masked_fill(mask, -1e9)  # 屏蔽对角线
        
        loss = -torch.sum(pos_weight * F.log_softmax(logits, dim=1), dim=1).mean()
        
        return loss
        
class SpectralRegularizer:
    """通过控制特征矩阵的奇异值分布防止坍塌"""
    def __init__(self, weight=0.01, target_rank=None):
        self.weight = weight
        self.target_rank = target_rank
        
    def __call__(self, features):
        """
        特征: [batch_size, feature_dim]
        鼓励特征矩阵的奇异值分布均匀，避免少数奇异值主导
        """
        # 计算奇异值
        U, S, V = torch.svd(features)
        
        # 归一化奇异值
        S_normalized = S / S.sum()
        
        if self.target_rank is not None:
            # 鼓励达到目标秩
            rank_loss = torch.relu(self.target_rank - S_normalized.nonzero().size(0))
        else:
            # 鼓励奇异值分布均匀（最大化熵）
            # 避免某些奇异值为0或过大
            epsilon = 1e-8
            entropy = -torch.sum(S_normalized * torch.log(S_normalized + epsilon))
            max_entropy = torch.log(torch.tensor(S.size(0), dtype=torch.float32))
            
            # 损失 = 1 - 归一化熵（最小化这个损失）
            rank_loss = 1 - entropy / max_entropy
        
        return self.weight * rank_loss                                    