# coding: utf-8

import numpy as np
import torch


def weighted_binary_logistic(pred_score, gt_label, threshold=0.5):
    # Flatten to 1d-array.
    pred_score, gt_label = pred_score.contiguous().view(-1), gt_label.contiguous().view(-1)
    # print(pred_score.size(), gt_label.size())
    threshold_mask = (gt_label > threshold).float()
    num_entries = len(threshold_mask)
    num_positive = torch.sum(threshold_mask)
    ratio = num_entries / num_positive

    # For positive one(above threshold), loss = num_entries / num_positive * log(p_i)
    # For negative one(below threshold), loss = num_entries / num_negative * log(1 - p_i)
    epsilon = 1e-6
    loss_positive = 0.5 * ratio * torch.log(pred_score + epsilon) * threshold_mask
    loss_negative = 0.5 * ratio / (ratio - 1) * torch.log(1.0 - pred_score + epsilon) * (1.0 - threshold_mask)
    loss = -1.0 * torch.mean(loss_positive + loss_negative)
    return loss


def get_mask(tscale, max_len):
    """Generate mask of BM confidence map. """

    # mask.shape: Duration * Start Time
    mask = np.zeros([max_len, tscale], np.float32)

    # The proposals whose ending boundaries exceed the range of video are left(mask = 0).
    for i in range(max_len):
        for j in range(tscale - i):
            mask[i][j] = 1
    return torch.Tensor(mask)


def bmn_loss(pred_bm, pred_start, pred_end, gt_iou_map, gt_start, gt_end, gt_start_short, gt_end_short, bm_mask):
    """
    Loss of BMN, which consists of three parts: TEM, PEM-regression and PEM-classification.

    Arguements:
        1. Model output:
        pred_bm([2*D*T]): (M_c): BM confidence map which consist of 'regression' and 'binary classification'.
        pred_start([T]): Temporal boundary start probability sequence.
        pred_end([T]): Temporal boundary end probability sequence.
        2. Label from 'DataLoader':
        gt_iou_map([T*T]): (G_c): iou between certain period and all GT proposals.
        gt_start([T]): G_(S, w): score sequence which presents ioa between certain temporal moment and expanded start periods(G_S).
        gt_end([T]): G_(E, w): score sequence which presents ioa between certain temporal moment and expanded end periods(G_E).
        3. Mask.
        bm_mask([T*T]): BM confidence map's mask.
    """
    # pred_bm.real_shape: batch_size * 2 * D * T
    # print("pred_bm", pred_bm.size())
    # print("pred_bm_contiguous", pred_bm.contiguous().size())

    pred_bm_reg = pred_bm[:, 0].contiguous()
    pred_bm_cls = pred_bm[:, 1].contiguous()

    # print("gt_iou_map", gt_iou_map.size())
    # print("pred_bm_reg", pred_bm_reg.size())
    # print("pred_bm_cls", pred_bm_cls.size())
    # print("pred_bm", pred_bm.size())
    # print("pred_start", pred_start.size())
    # print("pred_end", pred_end.size())
    # print("gt_start", gt_start.size())
    # print("gt_end", gt_end.size())

    gt_iou_map = gt_iou_map * bm_mask

    tem_global_loss = tem_loss_func(pred_start[:, 0], pred_end[:, 0], gt_start, gt_end)
    tem_local_loss = tem_loss_func(pred_start[:, 1], pred_end[:, 1], gt_start_short, gt_end_short)
    tem_loss = tem_global_loss + tem_local_loss

    pem_reg_loss = pem_reg_loss_func(pred_bm_reg, gt_iou_map, bm_mask)
    pem_cls_loss = pem_cls_loss_func(pred_bm_cls, gt_iou_map, bm_mask)

    # cost = tem_loss
    cost = tem_loss + 10 * pem_reg_loss + pem_cls_loss

    loss_dict = dict(tem_global_loss=tem_global_loss, tem_local_loss=tem_local_loss, pem_reg_loss=pem_reg_loss,
                     pem_cls_loss=pem_cls_loss, cost=cost)

    return loss_dict


def tem_loss_func(pred_start, pred_end, gt_start, gt_end):
    """
    Adopt weighted binary logistic regression loss function for predicted and GT start/end score sequence.

    Arguements:
        same as 'bmn_loss'.
    """

    loss_start = weighted_binary_logistic(pred_start, gt_start)
    loss_end = weighted_binary_logistic(pred_end, gt_end)
    tem_loss = loss_start + loss_end
    return tem_loss


def pem_reg_loss_func(pred_reg_score, gt_iou_map, bm_mask, high_threshold=0.7, low_threshold=0.3):
    """
    Use MSE loss + L2 to make each proposal's regression score approxiamte to proposal's IoU between GT.

    Arguements:
        pred_reg_score([T*T]): regression part of 'BM_confidence_map'.
        gt_iou_map([T*T]): (G_c): iou between certain period and all GT proposals.
        bm_mask([T*T]): BM confidence map's mask.
        high_threshold(float[1]): high threshold of regression score.
        low_threshold(float[1]): low threshold of regression score.
    """

    gt_iou_map = gt_iou_map * bm_mask

    mask_high = (gt_iou_map > high_threshold).float()
    mask_medium = ((gt_iou_map <= high_threshold) & (gt_iou_map > low_threshold)).float()
    mask_low = ((gt_iou_map <= low_threshold) & (gt_iou_map > 0.)).float()

    num_high = torch.sum(mask_high)
    num_medium = torch.sum(mask_medium)
    num_low = torch.sum(mask_low)

    ratio_1 = num_high / num_medium
    # eg: gt_iou_map.shape: torch.Size([2,3]), then *gt_iou_map.shape: 2 3
    mask_medium_rand = torch.Tensor(np.random.rand(*gt_iou_map.shape)).cuda()
    mask_medium_rand = mask_medium_rand * mask_medium
    mask_medium_rand = (mask_medium_rand > (1 - ratio_1)).float()

    ratio_2 = num_high / num_low
    mask_low_rand = torch.Tensor(np.random.rand(*gt_iou_map.shape)).cuda()
    mask_low_rand = mask_low_rand * mask_low
    mask_low_rand = (mask_low_rand > (1 - ratio_2)).float()

    weights = mask_high + mask_medium_rand + mask_low_rand
    MSE_loss_function = torch.nn.MSELoss()
    loss = MSE_loss_function(pred_reg_score * weights, gt_iou_map * weights)
    loss = 0.5 * torch.sum(loss * torch.ones(*weights.shape).cuda()) / torch.sum(weights)

    return loss


def pem_cls_loss_func(pred_cls_score, gt_iou_map, bm_mask, threshold=0.9):
    """
    Adopt weighted binary logistic regression loss function for predicted binary classification confidence map and GT iou map.

    Arguements:
        pred_cls_score([T*T]): binary classification part of 'BM_confidence_map'.
        gt_iou_map([T*T]): (G_c): iou between certain period and all GT proposals.
        threshold(float[1]): threshold of gt_iou_map.
    """

    mask_positive = (gt_iou_map > threshold).float()
    mask_negative = (gt_iou_map <= threshold).float() * bm_mask

    num_positive = torch.sum(mask_positive)
    num_negative = torch.sum(mask_negative)
    num_entries = num_positive + num_negative

    # For positive one(above threshold), loss = num_entries / num_positive * log(p_i)
    # For negative one(below threshold), loss = num_entries / num_negative * log(1 - p_i)
    epsilon = 1e-6
    ratio = num_entries / num_positive

    loss_positive = 0.5 * ratio * torch.log(pred_cls_score + epsilon) * mask_positive
    loss_negative = 0.5 * ratio / (ratio - 1) * torch.log(1.0 - pred_cls_score + epsilon) * mask_negative
    loss = -1.0 * torch.sum(loss_positive + loss_negative) / num_entries
    return loss
