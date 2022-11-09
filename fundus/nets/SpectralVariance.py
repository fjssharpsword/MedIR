from scipy.spatial import distance
from sklearn.preprocessing import normalize
import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.stats import entropy
import torch.nn as nn
import torch

#https://github.com/Confusezius/Revisiting_Deep_Metric_Learning_PyTorch/blob/master/metrics/rho_spectrum.py
"""
Include it with rho_spectrum@1. 
To exclude the k largest spectral values for a more robust estimate, simply include rho_spectrum@k+1. 
Adding rho_spectrum@0 logs the whole singular value distribution, and rho_spectrum@-1 computes KL(q,p) instead of KL(p,q).
"""

class SpectralVariance(nn.Module):
    def __init__(self, embed_dim, mode):
        super(SpectralVariance, self).__init__()
        #mode=-1, 0, k
        self.mode      = mode
        self.embed_dim = embed_dim
        self.requires = ['features']
        self.name     = 'rho_spectrum@'+str(mode)

    def forward(self, features):

        if isinstance(features, torch.Tensor):
            _,s,_ = torch.svd(features)
            s     = s.cpu().numpy()
        else:
            svd = TruncatedSVD(n_components=self.embed_dim-1, n_iter=7, random_state=42)
            svd.fit(features)
            s = svd.singular_values_

        if self.mode!=0:
            s = s[np.abs(self.mode)-1:]
        s_norm  = s/np.sum(s)
        uniform = np.ones(len(s))/(len(s))

        if self.mode<0:
            kl = entropy(s_norm, uniform)
        if self.mode>0:
            kl = entropy(uniform, s_norm)
        if self.mode==0:
            kl = s_norm

        return kl

if __name__ == '__main__':
    x = torch.rand(100, 512).cuda()
    sv = SpectralVariance(embed_dim=512, mode=1).cuda()
    out = sv(x)
    print(out)