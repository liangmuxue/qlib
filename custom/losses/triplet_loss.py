import torch

from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from pytorch_metric_learning.losses import TripletMarginLoss
import pytorch_metric_learning.utils.common_functions as c_f

from cus_utils.common_compute import pairwise_distances

class TripletTargetLoss(TripletMarginLoss):
    
    def __init__(
        self,
        margin=0.3,
        swap=False,
        smooth_loss=False,
        triplets_per_anchor="all",
        **kwargs
    ):
        super().__init__(margin=margin,swap=swap,smooth_loss=smooth_loss,triplets_per_anchor=triplets_per_anchor,**kwargs)
 
    def forward(
        self, embeddings, labels=None, indices_tuple=None, ref_emb=None, ref_labels=None, ref_target=None
    ):
        return super().forward(embeddings,labels=labels,indices_tuple=indices_tuple,ref_emb=ref_emb,ref_labels=ref_labels)
        # self.reset_stats()
        # c_f.check_shapes(embeddings, labels)
        # if labels is not None:
        #     labels = c_f.to_device(labels, embeddings)
        # ref_emb, ref_labels = c_f.set_ref_emb(embeddings, labels, ref_emb, ref_labels)
        # loss_dict = self.compute_loss(
        #     embeddings, labels, indices_tuple, ref_emb, ref_labels,ref_target=ref_target
        # )
        # self.add_embedding_regularization_to_loss_dict(loss_dict, embeddings)
        # return self.reducer(loss_dict, embeddings, labels)
               
    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels, ref_target=None):
        return super().compute_loss(embeddings,labels, indices_tuple, ref_emb, ref_labels)
        # c_f.labels_or_indices_tuple_required(labels, indices_tuple)
        # indices_tuple = lmu.convert_to_triplets(
        #     indices_tuple, labels, ref_labels, t_per_anchor=self.triplets_per_anchor
        # )
        # anchor_idx, positive_idx, negative_idx = indices_tuple
        # if len(anchor_idx) == 0:
        #     return self.zero_losses()
        # # 使用自定义配对比较函数生成比较矩阵
        # mat = pairwise_distances(embeddings,distance_func=self.distance)
        # ap_dists = mat[anchor_idx, positive_idx]
        # an_dists = mat[anchor_idx, negative_idx]
        #
        # current_margins = ap_dists - an_dists
        # violation = current_margins + self.margin
        # if self.smooth_loss:
        #     loss = torch.nn.functional.softplus(violation)
        # else:
        #     loss = torch.nn.functional.relu(violation)
        #
        # return {
        #     "loss": {
        #         "losses": loss,
        #         "indices": indices_tuple,
        #         "reduction_type": "triplet",
        #     }
        # }    