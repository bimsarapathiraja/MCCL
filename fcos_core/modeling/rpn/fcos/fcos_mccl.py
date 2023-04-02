import math
import torch
import torch.nn.functional as F
from torch import nn
import logging

from .inference import make_fcos_postprocessor
from .loss import make_fcos_loss_evaluator

from fcos_core.layers import Scale
from fcos_core.layers import DFConv2d


class FCOSHeadMCCL(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOSHeadMCCL, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg.MODEL.FCOS.NUM_CLASSES - 1
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.norm_reg_targets = cfg.MODEL.FCOS.NORM_REG_TARGETS
        self.centerness_on_reg = cfg.MODEL.FCOS.CENTERNESS_ON_REG
        self.use_dcn_in_tower = cfg.MODEL.FCOS.USE_DCN_IN_TOWER

        self.num_mc_samples = cfg.MODEL.FCOS.NUM_MC_SAMPLES
        self.iou_dropout = cfg.MODEL.FCOS.MCCL_IOU_DROPOUT
        self.cls_dropout = cfg.MODEL.FCOS.MCCL_CLS_DROPOUT

        logger = logging.getLogger("fcos_core.fcos_mccl")
        logger.info("Inside fcos_mccl")
        logger.info(f"num_mc_samples: {self.num_mc_samples} cls_dropout: {self.cls_dropout} iou_dropout: {self.iou_dropout}")

        cls_tower = []
        bbox_tower = []
        for i in range(cfg.MODEL.FCOS.NUM_CONVS):
            if self.use_dcn_in_tower and \
                    i == cfg.MODEL.FCOS.NUM_CONVS - 1:
                conv_func = DFConv2d
            else:
                conv_func = nn.Conv2d

            cls_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1,
            padding=1
        )
        self.centerness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )

        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred,
                        self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x):
        # logits = []
        # bbox_reg = []
        cls_logits_lst = []
        bbox_reg_lst = []
        num_scales = len(x)
        for i in range(num_scales):
            cls_logits_lst.append([])
            bbox_reg_lst.append([])

        centerness = []
        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            box_tower = self.bbox_tower(feature)

            # logits.append(self.cls_logits(cls_tower))
            for i in range(self.num_mc_samples):
                cls_logits_val = self.cls_logits(F.dropout2d(cls_tower, p=self.cls_dropout, training=self.training)) 
                cls_logits_lst[l].append(cls_logits_val)

            if self.centerness_on_reg:
                centerness.append(self.centerness(box_tower))
            else:
                centerness.append(self.centerness(cls_tower))

            for i in range(self.num_mc_samples):
                bbox_pred = self.scales[l](self.bbox_pred(F.dropout2d(box_tower, p=self.iou_dropout, training=self.training))) # cert
                # bbox_pred = self.scales[l](self.bbox_pred(box_tower))
                if self.norm_reg_targets:
                    bbox_pred = F.relu(bbox_pred)
                    if self.training:
                        bbox_reg_lst[l].append(bbox_pred)
                    else:
                        bbox_reg_lst[l].append(bbox_pred * self.fpn_strides[l])
                else:
                    bbox_reg_lst[l].append(torch.exp(bbox_pred))
        return cls_logits_lst, bbox_reg_lst, centerness


class FCOSModuleMCCL(torch.nn.Module):
    """
    Module for FCOS computation. Takes feature maps from the backbone and
    FCOS outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):
        super(FCOSModuleMCCL, self).__init__()

        logger = logging.getLogger("fcos_core.fcos_mccl")
        logger.info("Inside FCOSModuleMCCL")

        head = FCOSHeadMCCL(cfg, in_channels)

        box_selector_test = make_fcos_postprocessor(cfg)

        loss_evaluator = make_fcos_loss_evaluator(cfg)
        self.head = head
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.calib_loss = "loss_" + cfg.MODEL.FCOS.LOSS_TYPE

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        box_cls_lst, box_regression_lst, centerness = self.head(features)
        locations = self.compute_locations(features)
 
        if self.training:
            return self._forward_train(
                locations, box_cls_lst, 
                box_regression_lst, 
                centerness, targets
            )
        else:
            box_cls_test = [lst[0] for lst in box_cls_lst]
            box_regression_test = [lst[0] for lst in box_regression_lst]
            return self._forward_test(
                locations, box_cls_test, box_regression_test, 
                centerness, images.image_sizes
            )

    def _forward_train(self, locations, box_cls_lst, box_regression_lst, centerness, targets):
        loss_box_cls, loss_box_reg, loss_centerness, loss_cal = self.loss_evaluator(
            locations, box_cls_lst, box_regression_lst, centerness, targets
        )
        losses = {
            "loss_cls": loss_box_cls,
            "loss_reg": loss_box_reg,
            "loss_centerness": loss_centerness,
            self.calib_loss: loss_cal
        }
        return None, losses

    def _forward_test(self, locations, box_cls, box_regression, centerness, image_sizes):
        boxes = self.box_selector_test(
            locations, box_cls, box_regression, 
            centerness, image_sizes
        )
        return boxes, {}

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations

def build_fcos_mccl(cfg, in_channels):
    return FCOSModuleMCCL(cfg, in_channels)
