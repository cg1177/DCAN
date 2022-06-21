# coding: utf-8

import numpy as np


def ioa_with_temporal_proposals(box_min, box_max, proposals_min, proposals_max):
    """ 
    Return ioas bewteen box and all temporal proposals(anchors). Used in action field.
        ioa = intersection(overlap) / box_length
    
    Arguements:
        box_min: (np.ndarray[1]): start time of box.
        box_max: (np.ndarray[1]): end time of box.
        proposals_min: (np.ndarray[N]): start times of temporal proposals.
        proposals_max: (np.ndarray[N]): end times of temporal proposals.

    Return Arguements:
        ioas: (np.ndarray[N]): ioas between box and all temporal proposals.
    """
    inter_min = np.maximum(box_min, proposals_min)
    inter_max = np.minimum(box_max, proposals_max)
    inter_length = np.maximum(inter_max - inter_min, 0.)
    box_length = box_max - box_min
    ioas = np.divide(inter_length, box_length)
    return ioas


def iou_with_temporal_proposals(box_min, box_max, proposals_min, proposals_max):
    """ 
    Return ious bewteen box and all temporal proposals(anchors). Used in action field.
        iou = intersection(overlap) / union
    
    Arguements:
        box_min: (np.ndarray[1]): start time of box.
        box_max: (np.ndarray[1]): end time of box.
        proposals_min: (np.ndarray[N]): start times of temporal proposals.
        proposals_max: (np.ndarray[N]): end times of temporal proposals.

    Return Arguements:
        ious: (np.ndarray[N]): ious between box and all temporal proposals.
    """
    inter_min = np.maximum(box_min, proposals_min)
    inter_max = np.minimum(box_max, proposals_max)
    inter_length = np.maximum(inter_max - inter_min, 0.)
    proposals_length = proposals_max - proposals_min
    union_length = proposals_length - inter_length + box_max - box_min
    ious = np.divide(inter_length, union_length)
    return ious
