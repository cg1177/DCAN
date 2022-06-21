import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.iou_utils import iou_with_temporal_proposals, ioa_with_temporal_proposals


def load_json(file):
    with open(file) as f:
        json_data = json.load(f)
        return json_data


class MyDataset(Dataset):
    def __init__(self, opt, mode='train'):

        self.mode = mode
        self.temporal_scale = opt.temporal_scale  # 100
        self.temporal_gap = 1. / self.temporal_scale
        self.feature_path = opt.feature_path  # /data/activitynet_feature_cuhk/
        self.video_info_path = opt.video_info_path  # /data/activitynet_annotations/video_info_new.csv
        self.video_anno_path = opt.video_anno_path  # /data/activitynet_annotations/anet_anno_action.json

        self.get_all_data()

    def get_all_data(self):
        """Get all data. Save annotation in 'video_dict' and video names in 'video_list'. """

        self.video_dict = {}

        videos_info = pd.read_csv(self.video_info_path)
        videos_anno = load_json(self.video_anno_path)

        for i in range(len(videos_info)):
            video_name = videos_info.video.values[i]
            video_anno = videos_anno[video_name]
            if self.mode in videos_info.subset.values[i]:
                self.video_dict[video_name] = video_anno
        self.video_list = list(self.video_dict.keys())
        print("{} subset has {} videos.".format(self.mode, len(self.video_list)))

    def get_train_label(self, index):
        """Get confidence map and score sequences through certain video's annotation. """

        video_name = self.video_list[index]
        video_info = self.video_dict[video_name]
        video_anno = video_info['annotations']

        # The number of feature frames should be a multiple of 16, while some frames in the end are left.
        video_real_frame = video_info['duration_frame']
        video_real_second = video_info['duration_second']
        video_feature_frame = video_info['feature_frame']
        video_feature_second = float(video_feature_frame) / video_real_frame * video_real_second

        # Change the measurement from second to percentage, then save in 'gt_bbox'.
        gt_bbox = []
        for i in range(len(video_anno)):
            temp_anno = video_anno[i]
            temp_start = max((min(1, temp_anno['segment'][0] / video_feature_second)), 0)
            temp_end = max((min(1, temp_anno['segment'][1] / video_feature_second)), 0)
            gt_bbox.append([temp_start, temp_end])

        # Expand start/end moment to a period, in which the moment is at the middle.
        gt_bbox = np.array(gt_bbox)
        gt_starts = gt_bbox[:, 0]
        gt_ends = gt_bbox[:, 1]
        gt_lens = gt_ends - gt_starts
        # duration = 3 * self.temporal_gap
        duration = np.maximum(self.temporal_gap, 0.1 * gt_lens)

        gt_start_periods = np.stack((gt_starts - duration / 2, gt_starts + duration / 2), axis=1)
        gt_end_periods = np.stack((gt_ends - duration / 2, gt_ends + duration / 2), axis=1)

        duration_short = 3 * self.temporal_gap
        gt_start_periods_short = np.stack((gt_starts - duration_short / 2, gt_starts + duration_short / 2), axis=1)
        gt_end_periods_short = np.stack((gt_ends - duration_short / 2, gt_ends + duration_short / 2), axis=1)

        # NOTE: tensor or np?
        # Generate 'iou_map' which presents iou between certain period and all GT proposals.
        gt_iou_map = np.zeros((self.temporal_scale, self.temporal_scale))
        for i in range(self.temporal_scale):
            for j in range(i, self.temporal_scale):
                gt_iou_map[i][j] = np.max(
                    iou_with_temporal_proposals(i * self.temporal_gap, (j + 1) * self.temporal_gap, gt_starts, gt_ends))
        gt_iou_map = torch.Tensor(gt_iou_map)

        # Generate score sequence which presents ioa between certain temporal moment and expanded periods.
        #   ioa = overlap / fixed period length, the distribution of moments is regular while that of periods is irregular.
        anchor_xmin = [self.temporal_gap * (i - 0.5) for i in range(self.temporal_scale)]
        anchor_xmax = [self.temporal_gap * (i + 0.5) for i in range(self.temporal_scale)]
        start_score = []
        end_score = []
        start_short_score = []
        end_short_score = []
        for i in range(len(anchor_xmin)):
            start_score.append(
                np.max(
                    ioa_with_temporal_proposals(anchor_xmin[i], anchor_xmax[i], gt_start_periods[:, 0],
                                                gt_start_periods[:, 1])))
            end_score.append(
                np.max(
                    ioa_with_temporal_proposals(anchor_xmin[i], anchor_xmax[i], gt_end_periods[:, 0],
                                                gt_end_periods[:, 1])))
            start_short_score.append(
                np.max(
                    ioa_with_temporal_proposals(anchor_xmin[i], anchor_xmax[i], gt_start_periods_short[:, 0],
                                                gt_start_periods_short[:, 1])))
            end_short_score.append(
                np.max(
                    ioa_with_temporal_proposals(anchor_xmin[i], anchor_xmax[i], gt_end_periods_short[:, 0],
                                                gt_end_periods_short[:, 1])))
        start_score = torch.Tensor(start_score)
        end_score = torch.Tensor(end_score)
        start_short_score = torch.Tensor(start_short_score)
        end_short_score = torch.Tensor(end_short_score)

        return gt_iou_map, start_score, end_score, start_short_score, end_short_score

    def __getitem__(self, index):

        video_name = self.video_list[index]
        video_feature = pd.read_csv(self.feature_path + "csv_mean_" + str(self.temporal_scale) + "/" + video_name +
                                    ".csv")
        # NOTE: test how these works.
        video_feature = torch.Tensor(video_feature.values[:, :])
        video_feature = torch.transpose(video_feature, 0, 1).float()
        gt_iou_map, start_score, end_score, start_short_score, end_short_score = self.get_train_label(index)
        if self.mode == 'train':
            # the last three elements present: label_confidence, label_start, label_end
            return video_feature, gt_iou_map, start_score, end_score, start_short_score, end_short_score
        else:

            return index, video_feature, gt_iou_map, start_score, end_score, start_short_score, end_short_score

    def __len__(self):

        return len(self.video_list)
