from itertools import product as product
from math import sqrt as sqrt

import torch

# if torch.cuda.is_available():
#     torch.set_default_tensor_type('torch.cuda.FloatTensor')

class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    Note:
    This 'layer' has changed between versions of the original SSD
    paper, so we include both versions, but note v2 is the most tested and most
    recent version of the paper.

    """

    def __init__(self, 
                variance=[0.1, 0.2],
                min_dim=512,
                feature_maps=[64, 32, 16, 8], 
                min_sizes=[32, 64, 128, 256],
                max_sizes=None,
                steps=[8, 16, 32, 64], 
                aspect_ratios=[[2], [2], [2], [2]], 
                clip=True):
        super(PriorBox, self).__init__()
        self.image_size = min_dim
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(aspect_ratios)
        self.variance = variance or [0.1]
        self.feature_maps = feature_maps
        self.min_sizes = min_sizes
        self.max_sizes = max_sizes
        self.steps = steps
        self.aspect_ratios = aspect_ratios
        self.clip = clip
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def gen_base_anchors(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                s_k = self.min_sizes[k] / self.image_size
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                if self.max_sizes:
                    s_k_prime = sqrt(s_k * (self.max_sizes[k] / self.image_size))
                    mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                    mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]

        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
