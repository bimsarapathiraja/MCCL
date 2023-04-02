import torch
import logging
import os

class MCCLLoss(torch.nn.Module):
    def __init__(self, iou_weight=0.1):
        super(MCCLLoss, self).__init__()
        self.iou_weight = iou_weight
        logger = logging.getLogger("fcos_core.mccl")
        logger.info("Inside MCCL with iou_weight: {}".format(iou_weight))

    def forward(self, output, variances, target, box_reg, box_reg_var, ious):
        output = torch.softmax(output, dim=1)
        # output = torch.sigmoid(output)
        certainity = 1 - torch.tanh(variances)
        
        class_range = torch.arange(1, output.shape[1] + 1).unsqueeze(0).cuda()
        t = target.unsqueeze(1)
        one_hotT = (t == class_range)
        loss = torch.abs((torch.mean(output, 0) + torch.mean(certainity, 0)) / 2 - torch.mean(one_hotT.float(), 0)).mean()

        box_reg_mean = torch.mean(box_reg, dim=1, keepdim=True)
        joint_var = box_reg_var + torch.square(box_reg - box_reg_mean)
        joint_var_mean = torch.mean(joint_var, dim=1)

        joint_certainity = 1 - torch.tanh(joint_var_mean)

        loss_iou = torch.mean(torch.abs(ious - joint_certainity))

        return loss + self.iou_weight * loss_iou
