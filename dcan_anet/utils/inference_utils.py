# coding: utf-8

import numpy as np
import pandas as pd
from utils.nms_utils import Soft_NMS


def proposals_select_pervideo(opt, video_list, video_dict, results, cuhk_data_score, cuhk_data_action):
    """
    Select 100 proposals in each video's proposals(soft-NMS), then save.

    Arguements:
        opt: (config): parameters.
        video_list: (list): video names.
        video_dict: (dict): video information.
        results: (multiprocessing.Manager().dict()): a dict which contains selected periods of each video.

    Return Arguements:
        video_name[2:]: (str): name of video, eg: 'c8enCfzqw'.
        proposals: (list): selected proposals of certain video.
    """
    for video_name in video_list:
        cuhk_score = cuhk_data_score[video_name[2:]]
        cuhk_class_1 = cuhk_data_action[np.argmax(cuhk_score)]
        cuhk_score_1 = max(cuhk_score)
        cuhk_score[np.argmax(cuhk_score)] = -1
        cuhk_score_2 = np.max(cuhk_score)
        cuhk_class_2 = cuhk_data_action[np.argmax(cuhk_score)]

        df = pd.read_csv("./output/BMN_results/" + video_name + ".csv")
        df['score'] = df.cls_score.values[:] * df.reg_score.values[:]

        if len(df) > 1:
            df = Soft_NMS(df, nms_threshold=opt.soft_nms_alpha)

        df = df.sort_values(by="score", ascending=False)

        video_info = video_dict[video_name]

        real_video_duration = float(
            video_info["duration_frame"] // 16 * 16) / video_info["duration_frame"] * video_info["duration_second"]

        proposals = []
        for i in range(min(100, len(df))):
            proposal = {}
            proposal["score"] = df.score.values[i] * cuhk_score_1
            proposal["segment"] = [
                max(0, df.xmin.values[i]) * real_video_duration,
                min(1, df.xmax.values[i]) * real_video_duration
            ]
            proposal["label"] = cuhk_class_1
            proposals.append(proposal)

        for i in range(min(100, len(df))):
            proposal = {}
            proposal["score"] = df.score.values[i] * cuhk_score_2
            proposal["segment"] = [
                max(0, df.xmin.values[i]) * real_video_duration,
                min(1, df.xmax.values[i]) * real_video_duration
            ]
            proposal["label"] = cuhk_class_2
            proposals.append(proposal)

        results[video_name[2:]] = proposals
