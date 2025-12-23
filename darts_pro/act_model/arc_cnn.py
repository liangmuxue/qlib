import torch
import torch.nn as nn
import torch.nn.functional as F

class ContinuousArcFace(nn.Module):
    def __init__(self, in_features_dim, num_proxies=64, s=30.0):
        super().__init__()
        
        self.in_features_dim = in_features_dim
        self.num_proxies = num_proxies  # 代理数量
        self.s = s
        
        # 特征提取
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # 创建连续分布的代理向量
        # 每个代理对应圆周上的一个特定角度
        self.proxies = nn.Parameter(torch.randn(num_proxies, 128))
        nn.init.xavier_normal_(self.proxies)
        
        # 代理对应的基准角度（均匀分布在[0, 2π]）
        self.proxy_angles = torch.linspace(0, 2*torch.pi, num_proxies)
        
    def forward(self, x):
        # 提取并归一化特征
        features = self.feature_extractor(x)
        features = F.normalize(features, p=2, dim=1)
        proxies = F.normalize(self.proxies, p=2, dim=1)
        
        # 计算特征与所有代理的余弦相似度
        cos_theta = F.linear(features, proxies)  # [batch, num_proxies]
        return cos_theta
        
        