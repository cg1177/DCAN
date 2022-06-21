import numpy as np
import pandas as pd

from utils.iou_utils import iou_with_temporal_proposals


def IOU(s1, e1, s2, e2):
    if (s2 > e1) or (s1 > e2):
        return 0
    Aor = max(e1, e2) - min(s1, s2)
    Aand = min(e1, e2) - max(s1, s2)
    return float(Aand) / Aor


# class NN(object):
#     ave = 0
#     num = 0


def Soft_NMS(df, nms_threshold=0.85, num_prop=100):
    '''
    From BSN code
    :param df:
    :param nms_threshold:
    :return:
    '''
    df = df.sort_values(by="score", ascending=False)
    import time
    start_time = time.time()

    tstart = list(df.xmin.values[:])
    tend = list(df.xmax.values[:])
    tscore = list(df.score.values[:])

    rstart = []
    rend = []
    rscore = []

    # # frost: I use a trick here, remove the detection
    # # which is longer than 300
    # for idx in range(0, len(tscore)):
    #     if tend[idx] - tstart[idx] >= 300:
    #         tscore[idx] = 0

    while len(tscore) > 1 and len(rscore) < num_prop and max(tscore) > 0:
        max_index = tscore.index(max(tscore))
        for idx in range(0, len(tscore)):
            if idx != max_index:
                tmp_iou = IOU(tstart[max_index], tend[max_index], tstart[idx], tend[idx])
                if tmp_iou > 0:
                    tscore[idx] = tscore[idx] * np.exp(-np.square(tmp_iou) / nms_threshold)

        rstart.append(tstart[max_index])
        rend.append(tend[max_index])
        rscore.append(tscore[max_index])
        tstart.pop(max_index)
        tend.pop(max_index)
        tscore.pop(max_index)

    newDf = pd.DataFrame()
    newDf['score'] = rscore
    newDf['xmin'] = rstart
    newDf['xmax'] = rend
    # NN.ave = NN.ave * NN.num
    # NN.num += 1
    # NN.ave = (NN.ave + time.time() - start_time) / NN.num
    # print(NN.ave)

    # print("nms cost:", time.time() - start_time)
    return newDf


def soft_nms_proposal(df_proposal, alpha, t1, t2, num_return_proposal=100):
    """
    Soft Non-Max-Suppression on temporal proposals in inference process. Used in video field.
        Soft NMS will suppress the score of low-score proposals which has high IoU with high-score proposals.

    Arguements:  
        df_proposal (pandas.DataFrame): a csv file which contains start/end/score of proposals generated by network.
            keys: xmin, xmax, xmin_score, xmax_score, cls_score, reg_score, score
        alpha (float[1]): alpha value for Gaussian decaying function.
        t1 (float[1]): lower threshold for soft NMS.
        t2 (float[1]): higher threshold for soft NMS.
        num_return_proposal (int[1]): number of max return proposals.

    Return Arguements:    
        return_df (pandas.DataFrame()): remaining bounding boxes after NMS, which concludes 'start', 'end' and 'score'.
    """
    proposal_information = df_proposal.sort_values(by='score', ascending=False)

    # len(t_start) == len(t_end) == num_proposals
    # Directly concate corresponding element of 't_start' and 't_end' to get initial proposals.
    t_start = list(proposal_information.xmin.values[:])
    t_end = list(proposal_information.xmax.values[:])
    score = list(proposal_information.score.values[:])

    t_start_return = []
    t_end_return = []
    score_return = []

    while len(score) > 1 and len(score_return) < num_return_proposal:
        max_index = score.index(max(score))
        ious = iou_with_temporal_proposals(np.array(t_start), np.array(t_end), t_start[max_index], t_end[max_index])
        for i in range(len(score)):
            if i != max_index:
                iou = ious[i]
                length = t_end[max_index] - t_start[max_index]
                # if iou(max_score_bbox, b_i) is above the threshold, then `score_i = score_i * function(iou(max_score_box, b_i))`.
                # Here we use Gaussian kernel as the function.
                # if iou > t1 + (t2 - t1) * length:
                #     score[i] = score[i] * np.exp(-np.square(iou) / alpha)
                if iou > 0:
                    score[i] = score[i] * np.exp(-np.square(iou) / 0.8)

        t_start_return.append(t_start[max_index])
        t_end_return.append(t_end[max_index])
        score_return.append(score[max_index])

        t_start.pop(max_index)
        t_end.pop(max_index)
        score.pop(max_index)

    return_df = pd.DataFrame()
    return_df['xmin'] = t_start_return
    return_df['xmax'] = t_end_return
    return_df['score'] = score_return

    return return_df
