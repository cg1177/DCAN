#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-10-15 14:50:03
#   Description : pytorch_ppyolo
#
# ================================================================
import torch


def iou_with_temporal_proposals(proposals_a, proposals_b):
    proposals_a = proposals_a.unsqueeze(1)
    proposals_b = proposals_b.unsqueeze(0)
    inter_min = torch.maximum(proposals_a[:, :, 0], proposals_b[:, :, 0])
    inter_max = torch.minimum(proposals_a[:, :, 1], proposals_b[:, :, 1])
    inter_length = torch.relu(inter_max - inter_min)
    union_length = (proposals_b[:, :, 1] - proposals_b[:, :, 0]) + (
            proposals_a[:, :, 1] - proposals_a[:, :, 0]) - inter_length
    iou_matrix = inter_length / union_length
    return iou_matrix


# 相交矩形的面积
def intersect(box_a, box_b):
    """计算两组矩形两两之间相交区域的面积
    Args:
        box_a: (tensor) bounding boxes, Shape: [A, 4].
        box_b: (tensor) bounding boxes, Shape: [B, 4].
    Return:
      (tensor) intersection area, Shape: [A, B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """计算两组矩形两两之间的iou
    Args:
        box_a: (tensor) bounding boxes, Shape: [A, 4].
        box_b: (tensor) bounding boxes, Shape: [B, 4].
    Return:
        ious: (tensor) Shape: [A, B]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A, B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A, B]
    union = area_a + area_b - inter
    return inter / union  # [A, B]


def _matrix_nms(bboxes, cate_labels, cate_scores, kernel='gaussian', sigma=2.0):
    """Matrix NMS for multi-class bboxes.
    Args:
        bboxes (Tensor): shape (n, 4)
        cate_labels (Tensor): shape (n), mask labels in descending order
        cate_scores (Tensor): shape (n), mask scores in descending order
        kernel (str):  'linear' or 'gaussian'
        sigma (float): std in gaussian method
    Returns:
        Tensor: cate_scores_update, tensors of shape (n)
    """
    n_samples = len(cate_labels)
    if n_samples == 0:
        return []

    # 计算一个n×n的IOU矩阵，两组矩形两两之间的IOU
    iou_matrix = jaccard(bboxes, bboxes)  # shape: [n_samples, n_samples]
    iou_matrix = iou_matrix.triu(diagonal=1)  # 只取上三角部分

    # label_specific matrix.
    cate_labels_x = cate_labels.expand(n_samples, n_samples)  # shape: [n_samples, n_samples]
    # 第i行第j列表示的是第i个预测框和第j个预测框的类别id是否相同。我们抑制的是同类的预测框。
    label_matrix = (cate_labels_x == cate_labels_x.transpose(1, 0)).float().triu(
        diagonal=1)  # shape: [n_samples, n_samples]

    # IoU compensation
    # 非同类的iou置为0，同类的iou保留。逐列取最大iou
    compensate_iou, _ = (iou_matrix * label_matrix).max(0)  # shape: [n_samples, ]
    compensate_iou = compensate_iou.expand(n_samples, n_samples).transpose(1, 0)  # shape: [n_samples, n_samples]

    # IoU decay
    # 非同类的iou置为0，同类的iou保留。
    decay_iou = iou_matrix * label_matrix  # shape: [n_samples, n_samples]

    # matrix nms
    if kernel == 'gaussian':
        decay_matrix = torch.exp(-1 * sigma * (decay_iou ** 2))
        compensate_matrix = torch.exp(-1 * sigma * (compensate_iou ** 2))
        decay_coefficient, _ = (decay_matrix / compensate_matrix).min(0)
    elif kernel == 'linear':
        decay_matrix = (1 - decay_iou) / (1 - compensate_iou)
        decay_coefficient, _ = decay_matrix.min(0)
    else:
        raise NotImplementedError

    # 更新分数
    cate_scores_update = cate_scores * decay_coefficient
    return cate_scores_update


def matrix_nms(bboxes,
               scores,
               score_threshold,
               post_threshold,
               nms_top_k,
               keep_top_k,
               use_gaussian=False,
               gaussian_sigma=2.):
    inds = (scores > score_threshold)
    cate_scores = scores[inds]
    if len(cate_scores) == 0:
        return torch.zeros((1, 6), device=bboxes.device) - 1.0

    inds = inds.nonzero()
    cate_labels = inds[:, 1]
    bboxes = bboxes[inds[:, 0]]

    # sort and keep top nms_top_k
    sort_inds = torch.argsort(cate_scores, descending=True)
    if nms_top_k > 0 and len(sort_inds) > nms_top_k:
        sort_inds = sort_inds[:nms_top_k]
    bboxes = bboxes[sort_inds, :]
    cate_scores = cate_scores[sort_inds]
    cate_labels = cate_labels[sort_inds]

    # Matrix NMS
    kernel = 'gaussian' if use_gaussian else 'linear'
    cate_scores = _matrix_nms(bboxes, cate_labels, cate_scores, kernel=kernel, sigma=gaussian_sigma)

    # filter.
    keep = cate_scores >= post_threshold
    if keep.sum() == 0:
        return torch.zeros((1, 6), device=bboxes.device) - 1.0
    bboxes = bboxes[keep, :]
    cate_scores = cate_scores[keep]
    cate_labels = cate_labels[keep]

    # sort and keep keep_top_k
    sort_inds = torch.argsort(cate_scores, descending=True)
    if len(sort_inds) > keep_top_k:
        sort_inds = sort_inds[:keep_top_k]
    bboxes = bboxes[sort_inds, :]
    cate_scores = cate_scores[sort_inds]
    cate_labels = cate_labels[sort_inds]

    cate_scores = cate_scores.unsqueeze(1)
    cate_labels = cate_labels.unsqueeze(1).float()
    pred = torch.cat([cate_labels, cate_scores, bboxes], 1)

    return pred


# def _temporal_proposal_matrix_nms(proposals, cate_labels, cate_scores, kernel='gaussian', sigma=2.0):
#     """Matrix NMS for multi-class bboxes.
#     Args:
#         bboxes (Tensor): shape (n, 4)
#         cate_labels (Tensor): shape (n), mask labels in descending order
#         cate_scores (Tensor): shape (n), mask scores in descending order
#         kernel (str):  'linear' or 'gaussian'
#         sigma (float): std in gaussian method
#     Returns:
#         Tensor: cate_scores_update, tensors of shape (n)
#     """
#     n_samples = len(cate_labels)
#     if n_samples == 0:
#         return []
#
#     # 计算一个n×n的IOU矩阵，两组矩形两两之间的IOU
#     iou_matrix = iou_with_temporal_proposals(proposals, proposals)  # shape: [n_samples, n_samples]
#     iou_matrix = iou_matrix.triu(diagonal=1)  # 只取上三角部分
#
#     # label_specific matrix.
#     cate_labels_x = cate_labels.expand(n_samples, n_samples)  # shape: [n_samples, n_samples]
#     # 第i行第j列表示的是第i个预测框和第j个预测框的类别id是否相同。我们抑制的是同类的预测框。
#     label_matrix = (cate_labels_x == cate_labels_x.transpose(1, 0)).float().triu(
#         diagonal=1)  # shape: [n_samples, n_samples]
#
#     # IoU compensation
#     # 非同类的iou置为0，同类的iou保留。逐列取最大iou
#     compensate_iou, _ = (iou_matrix * label_matrix).max(0)  # shape: [n_samples, ]
#     compensate_iou = compensate_iou.expand(n_samples, n_samples).transpose(1, 0)  # shape: [n_samples, n_samples]
#
#     # IoU decay
#     # 非同类的iou置为0，同类的iou保留。
#     decay_iou = iou_matrix * label_matrix  # shape: [n_samples, n_samples]
#
#     # matrix nms
#     if kernel == 'gaussian':
#         decay_matrix = torch.exp(-1 * sigma * (decay_iou ** 2))
#         compensate_matrix = torch.exp(-1 * sigma * (compensate_iou ** 2))
#         decay_coefficient, _ = (decay_matrix / compensate_matrix).min(0)
#     elif kernel == 'linear':
#         decay_matrix = (1 - decay_iou) / (1 - compensate_iou)
#         decay_coefficient, _ = decay_matrix.min(0)
#     else:
#         raise NotImplementedError
#
#     # 更新分数
#     cate_scores_update = cate_scores * decay_coefficient
#     return cate_scores_update


def _temporal_proposal_matrix_nms_(proposals, scores, kernel='gaussian', sigma=2.0):
    n_samples = len(proposals)

    # 计算一个n×n的IOU矩阵，两组矩形两两之间的IOU
    iou_matrix = iou_with_temporal_proposals(proposals, proposals)  # shape: [n_samples, n_samples]
    iou_matrix = iou_matrix.triu(diagonal=1)  # 只取上三角部分

    # IoU compensation
    # 非同类的iou置为0，同类的iou保留。逐列取最大iou
    # print(iou_matrix.size())
    compensate_iou, _ = iou_matrix.max(0)  # shape: [n_samples, ]
    # print(compensate_iou.size())
    compensate_iou = compensate_iou.expand(n_samples, n_samples).transpose(1, 0)  # shape: [n_samples, n_samples]

    # IoU decay
    # 非同类的iou置为0，同类的iou保留。
    decay_iou = iou_matrix  # shape: [n_samples, n_samples]

    # matrix nms
    if kernel == 'gaussian':
        decay_matrix = torch.exp(-1 * sigma * (decay_iou ** 2))
        compensate_matrix = torch.exp(-1 * sigma * (compensate_iou ** 2))
        decay_coefficient, _ = (decay_matrix / compensate_matrix).min(0)
    elif kernel == 'linear':
        decay_matrix = (1 - decay_iou) / (1 - compensate_iou)
        decay_coefficient, _ = decay_matrix.min(0)
    else:
        raise NotImplementedError

    # 更新分数
    cate_scores_update = scores * decay_coefficient
    return cate_scores_update


def temporal_proposal_matrix_nms(proposals,
                                 scores,
                                 score_threshold=0.,
                                 post_threshold=0.2,
                                 nms_top_k=300,
                                 keep_top_k=100,
                                 use_gaussian=False,
                                 gaussian_sigma=2.):
    inds = (scores > score_threshold)
    cate_scores = scores[inds]

    inds = inds.nonzero(as_tuple=False)
    proposals = proposals[inds[:, 0]]

    # sort and keep top nms_top_k
    sort_inds = torch.argsort(cate_scores, descending=True)
    if nms_top_k > 0 and len(sort_inds) > nms_top_k:
        sort_inds = sort_inds[:nms_top_k]
    proposals = proposals[sort_inds, :]
    cate_scores = cate_scores[sort_inds]

    # Matrix NMS
    kernel = 'gaussian' if use_gaussian else 'linear'
    cate_scores = _temporal_proposal_matrix_nms_(proposals, cate_scores, kernel=kernel, sigma=gaussian_sigma)

    # filter.
    keep = cate_scores >= post_threshold
    if keep.sum() == 0:
        return torch.zeros((1, 6), device=proposals.device) - 1.0
    proposals = proposals[keep, :]
    cate_scores = cate_scores[keep]

    # sort and keep keep_top_k
    sort_inds = torch.argsort(cate_scores, descending=True)
    if len(sort_inds) > keep_top_k:
        sort_inds = sort_inds[:keep_top_k]
    proposals = proposals[sort_inds, :]
    cate_scores = cate_scores[sort_inds]

    cate_scores = cate_scores.unsqueeze(1)

    return proposals, cate_scores


if __name__ == '__main__':
    proposals = torch.randn(3000, 2)
    scores = torch.randn(3000)
    proposals, scores = temporal_proposal_matrix_nms(proposals, scores, score_threshold=-1, post_threshold=-1,
                                                     nms_top_k=1000, keep_top_k=200, use_gaussian=True)
    print(proposals.size())
    print(scores.size())
