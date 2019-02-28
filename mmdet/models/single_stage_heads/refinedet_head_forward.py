from __future__ import division

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from mmdet.core import (AnchorGenerator, anchor_target, multi_apply,
                        delta2bbox, weighted_smoothl1, multiclass_nms,
                        PriorBox, RefineMultiBoxLoss)
from mmdet.core.bbox import decode, center_size

class RefinedetHead(nn.Module):

    def __init__(self, 
                 num_classes,
                 feat_channel = 256,
                 arm_channels = [512, 512, 1024, 512],
                 variance=[0.1, 0.2],
                 min_dim=512,
                 feature_maps=[64, 32, 16, 8], 
                 min_sizes=[32, 64, 128, 256],
                 max_sizes=None,
                 steps=[8, 16, 32, 64],
                 aspect_ratios=[[2], [2], [2], [2]],
                 clip=True):
        super(RefinedetHead, self).__init__()
        self.num_classes = num_classes
        self.variance = variance
        self.num_anchors = 2*len(aspect_ratios[0]) + 1
        self.arm_channels = arm_channels
        self.feat_channel = feat_channel
        self.num_levels = len(arm_channels)
        self.priors = PriorBox(variance, min_dim, feature_maps, min_sizes, max_sizes, steps, aspect_ratios, clip).gen_base_anchors().cuda()
        self.arm_criterion = RefineMultiBoxLoss(2, 0.5, True, 0, True, 3, 0.5, False)
        self.odm_criterion = RefineMultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False,0.01)
        
        self.arm_loc = nn.ModuleList()
        self.arm_conf = nn.ModuleList()
        self.odm_loc = nn.ModuleList()
        self.odm_conf = nn.ModuleList()

        for i, arm_cin in enumerate(arm_channels):
            self.arm_loc.append(nn.Conv2d(arm_cin, 4 * self.num_anchors, kernel_size=3, stride=1, padding=1))
            self.arm_conf.append(nn.Conv2d(arm_cin, 2 * self.num_anchors, kernel_size=3, stride=1, padding=1))
            self.odm_loc.append(nn.Conv2d(self.feat_channel, 4 * self.num_anchors, kernel_size=3, stride=1, padding=1))
            self.odm_conf.append(nn.Conv2d(self.feat_channel, self.num_anchors*num_classes, kernel_size=3, stride=1, padding=1))

    def forward(self, feats):
        arm_fms, odm_fms = feats

        arm_loc_list = list()
        arm_conf_list = list()
        odm_loc_list = list()
        odm_conf_list = list()

        for (x, l, c) in zip(arm_fms, self.arm_loc, self.arm_conf):
            arm_loc_list.append(l(x).permute(0, 2, 3, 1).contiguous())
            arm_conf_list.append(c(x).permute(0, 2, 3, 1).contiguous())
        arm_loc = torch.cat([o.view(o.size(0), -1) for o in arm_loc_list], 1)
        arm_conf = torch.cat([o.view(o.size(0), -1) for o in arm_conf_list], 1)

        for (x, l, c) in zip(odm_fms, self.odm_loc, self.odm_conf):
            odm_loc_list.append(l(x).permute(0, 2, 3, 1).contiguous())
            odm_conf_list.append(c(x).permute(0, 2, 3, 1).contiguous())
        odm_loc = torch.cat([o.view(o.size(0), -1) for o in odm_loc_list], 1)
        odm_conf = torch.cat([o.view(o.size(0), -1) for o in odm_conf_list], 1)

        output = (
            arm_loc.view(arm_loc.size(0), -1, 4),  # loc preds
            arm_conf.view(arm_conf.size(0), -1, 2),  # conf preds
            odm_loc.view(odm_loc.size(0), -1, 4),  # loc preds
            odm_conf.view(odm_conf.size(0), -1, self.num_classes),  # conf preds
        )

        return output

    def loss(self, arm_loc, arm_conf, odm_loc, odm_conf, gt_bboxes, gt_labels, img_metas,
             cfg):
        for i in range(len(gt_bboxes)):
            gt_bboxes[i][:, 0] /= img_metas[i]['pad_shape'][1]
            gt_bboxes[i][:, 2] /= img_metas[i]['pad_shape'][1]
            gt_bboxes[i][:, 1] /= img_metas[i]['pad_shape'][0]
            gt_bboxes[i][:, 3] /= img_metas[i]['pad_shape'][0]
            gt_labels[i] = gt_labels[i].type_as(gt_bboxes[i]).unsqueeze(-1)
        targets = [torch.cat([gt_bboxes[i], gt_labels[i]], -1) for i in range(len(gt_bboxes))]
        arm_loss_l, arm_loss_c = self.arm_criterion((arm_loc, arm_conf), self.priors.to(targets[0].get_device()), targets)
        odm_loss_l, odm_loss_c = self.odm_criterion((odm_loc, odm_conf), self.priors.to(targets[0].get_device()), targets, (arm_loc,arm_conf),False)
        return dict(loss_arm_cls=arm_loss_c, loss_arm_reg=arm_loss_l, loss_odm_cls=odm_loss_c, loss_odm_reg=odm_loss_l)

    def get_det_bboxes(self,
                       arm_loc, 
                       arm_conf, 
                       loc, 
                       conf,
                       img_metas,
                       cfg,
                       rescale=False):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """

        # loc, conf = predictions
        loc_data = loc.data
        conf_data = conf.softmax(-1).data
        prior_data = self.priors.to(loc.get_device()).data
        num = loc_data.size(0)  # batch size
        
        # arm_loc, arm_conf = arm_data
        arm_loc_data = arm_loc.data
        arm_conf_data = arm_conf.softmax(-1).data
        arm_object_conf = arm_conf_data[:,:,1:]
        no_object_index = arm_object_conf <= cfg.score_thr
        conf_data[no_object_index.expand_as(conf_data)] = 0

        self.num_priors = prior_data.size(0)
        self.boxes = torch.zeros(num, self.num_priors, 4)
        self.scores = torch.zeros(num, self.num_priors, self.num_classes)

        if num == 1:
            # size batch x num_classes x num_priors
            conf_preds = conf_data.unsqueeze(0)

        else:
            conf_preds = conf_data.view(num, self.num_priors,
                                        self.num_classes)
            self.boxes.expand(num, self.num_priors, 4)
            self.scores.expand(num, self.num_priors, self.num_classes)
        # Decode predictions into bboxes.
        result_list = []
        for i in range(num):
            default = decode(arm_loc_data[i], prior_data, self.variance)
            default = center_size(default)

            decoded_boxes = decode(loc_data[i], default, self.variance)
            conf_scores = conf_preds[i].clone().squeeze(0)
            '''
            c_mask = conf_scores.gt(self.thresh)
            decoded_boxes = decoded_boxes[c_mask]
            conf_scores = conf_scores[c_mask]
            '''
            decoded_boxes[:, 0] = decoded_boxes[:, 0] * img_metas[i]['pad_shape'][1] 
            decoded_boxes[:, 2] = decoded_boxes[:, 2] * img_metas[i]['pad_shape'][1] 
            decoded_boxes[:, 1] = decoded_boxes[:, 1] * img_metas[i]['pad_shape'][0] 
            decoded_boxes[:, 3] = decoded_boxes[:, 3] * img_metas[i]['pad_shape'][0]
            decoded_boxes[:, 0].clamp_(min=0, max=img_metas[i]['pad_shape'][1] - 1)
            decoded_boxes[:, 2].clamp_(min=0, max=img_metas[i]['pad_shape'][1] - 1)
            decoded_boxes[:, 1].clamp_(min=0, max=img_metas[i]['pad_shape'][0] - 1)
            decoded_boxes[:, 3].clamp_(min=0, max=img_metas[i]['pad_shape'][0] - 1)
            if rescale:
                scales = torch.tensor(img_metas[i]['scale_factor']).to(decoded_boxes.get_device())
                decoded_boxes[:, 0] /= scales[0]
                decoded_boxes[:, 2] /= scales[2]
                decoded_boxes[:, 1] /= scales[1]
                decoded_boxes[:, 3] /= scales[3]
                    
            # For each class, perform nms
            det_bboxes, det_labels = multiclass_nms(decoded_boxes, conf_scores,
                                                cfg.score_thr, cfg.nms,
                                                cfg.max_per_img)

            result_list.append((det_bboxes, det_labels))
        return result_list

    def init_weights(self):
        return