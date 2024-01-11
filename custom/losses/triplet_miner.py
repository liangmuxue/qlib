import torch

from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
import pytorch_metric_learning.utils.common_functions as c_f
from pytorch_metric_learning.miners.triplet_margin_miner import TripletMarginMiner

from cus_utils.common_compute import pairwise_distances

class TripletTargetMiner(TripletMarginMiner):
    """自定义挖掘类，实现三元组数据挖掘"""
    
    def __init__(self, margin=0.2, type_of_triplets="all", **kwargs):
        super().__init__(margin=margin,type_of_triplets=type_of_triplets,**kwargs)
        
    def mine(self, embeddings, labels, ref_emb, ref_labels, ref_target=None):
        """修改原流程，不使用矩阵统一比较,而是使用目标值作为锚点进行比较"""
        
        # 获取所有三元组组合
        anchor_idx, positive_idx, negative_idx = lmu.get_all_triplets_indices(
            labels, ref_labels
        )
        # 使用自定义配对比较函数生成比较矩阵
        mat = pairwise_distances(embeddings,distance_func=self.distance)
        # mat = self.distance(embeddings, ref_emb)
        ap_dist = mat[anchor_idx, positive_idx]
        an_dist = mat[anchor_idx, negative_idx]
        triplet_margin = (
            an_dist - ap_dist
        )

        self.set_stats(ap_dist, an_dist, triplet_margin)

        if self.type_of_triplets == "easy":
            threshold_condition = triplet_margin > self.margin
        else:
            threshold_condition = triplet_margin <= self.margin
            if self.type_of_triplets == "hard":
                threshold_condition &= triplet_margin <= 0
            elif self.type_of_triplets == "semihard":
                threshold_condition &= triplet_margin > 0

        return (
            anchor_idx[threshold_condition],
            positive_idx[threshold_condition],
            negative_idx[threshold_condition],
        )          
        
        
    def forward(self, embeddings, labels, ref_emb=None, ref_labels=None, ref_target=None):
        """重载原方法，加入新目标参数"""
        
        self.reset_stats()
        with torch.no_grad():
            c_f.check_shapes(embeddings, labels)
            labels = c_f.to_device(labels, embeddings)
            ref_emb, ref_labels = c_f.set_ref_emb(
                embeddings, labels, ref_emb, ref_labels
            )
            mining_output = self.mine(embeddings, labels, ref_emb, ref_labels,ref_target=ref_target)
        self.output_assertion(mining_output)
        return mining_output
            