# encoding: utf-8
"""
Concordance-induced Triplet loss.
Author: Jason.Fang
Update time: 09/10/2022
Novelty: Our Concordance index-based Triplet Loss can avoid the dependency of margin setting of triplet loss.
Reference: https://kevinmusgrave.github.io/pytorch-metric-learning/
           https://github.com/KevinMusgrave/pytorch-metric-learning
Requirements: conda install pytorch-metric-learning -c metric-learning -c pytorch
"""

import torch
from pytorch_metric_learning.reducers import AvgNonZeroReducer
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from pytorch_metric_learning.losses.base_metric_loss_function import BaseMetricLossFunction
from pytorch_metric_learning.distances import CosineSimilarity

class CITLoss(BaseMetricLossFunction):
    """
    Args:
        margin: The desired difference between the anchor-positive distance and the
                anchor-negative distance.
        swap: Use the positive-negative distance instead of anchor-negative distance,
              if it violates the margin more.
        smooth_loss: Use the log-exp version of the triplet loss
    """

    def __init__(
        self,
        margin=0.05,
        swap=False,
        smooth_loss=False,
        triplets_per_anchor="all",
        gamma = 1.0, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.margin = margin
        self.swap = swap
        self.smooth_loss = smooth_loss
        self.triplets_per_anchor = triplets_per_anchor
        self.add_to_recordable_attributes(list_of_names=["margin"], is_stat=False)
        self.cosine = CosineSimilarity()
        self.gamma = gamma #scale factor

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        indices_tuple = lmu.convert_to_triplets(indices_tuple, labels, ref_labels, t_per_anchor=self.triplets_per_anchor)
        anchor_idx, positive_idx, negative_idx = indices_tuple
        if len(anchor_idx) == 0:
            return self.zero_losses()
        # mat1 = self.distance(embeddings, ref_emb) #distance=LpDistance(), default
        mat = self.cosine(embeddings, ref_emb) #distance=CosineSimilarity()
        ap_dists = mat[anchor_idx, positive_idx]
        an_dists = mat[anchor_idx, negative_idx]
        pn_dists = mat[positive_idx, negative_idx]
        
        if self.swap:
            pn_dists = mat[positive_idx, negative_idx]
            an_dists = self.distance.smallest_dist(an_dists, pn_dists)

        current_margins = self.distance.margin(ap_dists, an_dists)
        # current_margins = self.distance.margin(an_dists,ap_dists)
        #violation = current_margins + self.margin#d(a,p)-d(a,n) + m
        #exponential lower bound
        # violation = torch.exp(current_margins/self.gamma)
        violation = 1-torch.exp(current_margins) #cosine similarity
        if self.smooth_loss:
            le = torch.nn.functional.softplus(violation)
        else:
            le = torch.nn.functional.relu(violation)
        
        lp = -(ap_dists-torch.log(torch.exp(an_dists)+torch.exp(pn_dists)))

        loss = self.gamma*le+(1-self.gamma)*lp

        return {
            "loss": {
                "losses": loss,
                "indices": indices_tuple,
                "reduction_type": "triplet",
            }
        }

    def compute_CIScore(self, embeddings, labels, indices_tuple=None, ref_emb=None, ref_labels=None):
        indices_tuple = lmu.convert_to_triplets(indices_tuple, labels, ref_labels, t_per_anchor=self.triplets_per_anchor)
        anchor_idx, positive_idx, negative_idx = indices_tuple
        if len(anchor_idx) == 0:
            return torch.tensor(-1.0)
        mat = self.distance(embeddings, ref_emb)
        ap_dists = mat[anchor_idx, positive_idx]
        an_dists = mat[anchor_idx, negative_idx]
        if self.swap:
            pn_dists = mat[positive_idx, negative_idx]
            an_dists = self.distance.smallest_dist(an_dists, pn_dists)

        current_margins = self.distance.margin(ap_dists, an_dists)#d(a,p)-d(a,n) + m
        one_hot = torch.where(current_margins<0, 1, 0)
        ciscore = one_hot.sum()/len(one_hot)
        return ciscore

    def compute_triplet_sim(self, embeddings, labels, indices_tuple=None, ref_emb=None, ref_labels=None):
        indices_tuple = lmu.convert_to_triplets(indices_tuple, labels, ref_labels, t_per_anchor=self.triplets_per_anchor)
        anchor_idx, positive_idx, negative_idx = indices_tuple
        if len(anchor_idx) == 0:
            return torch.tensor([]), torch.tensor([])
        mat = self.distance(embeddings, ref_emb)
        ap_dists = mat[anchor_idx, positive_idx]
        an_dists = mat[anchor_idx, negative_idx]
        
        return ap_dists, an_dists

    def get_default_reducer(self):
        return AvgNonZeroReducer()

if __name__ == "__main__":
    #pdist = torch.nn.PairwiseDistance(p=2)
    #input1 = torch.randn(1, 128)
    #input2 = torch.randn(1, 128)
    #print(pdist(input1, input2))
    #print(pdist(input1, input1))
    #print(pdist(input2, input2))
    #print(torch.mul(input1,input1))
    #print(torch.mul(input1,input2))
    #input1 = torch.randn(128)
    #input2 = torch.randn(128)
    #print(input1.dot(input1))
    #print(input1.dot(input2))
    x = torch.tensor([])
    print(len(x))