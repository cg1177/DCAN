# coding: utf-8

import json
import os

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

matplotlib.use('Agg')
from dataset import MyDataset
from model import DCAN
from opt import MyConfig

from Eval.eval_detection import ANETdetection
from utils.opt_utils import get_cur_time_stamp
from joblib import Parallel, delayed
import math

# GPU setting.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # range GPU in order
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Basic test.
print("Pytorch's version is {}.".format(torch.__version__))
print("CUDNN's version is {}.".format(torch.backends.cudnn.version()))
print("CUDA's state is {}.".format(torch.cuda.is_available()))
print("CUDA's version is {}.".format(torch.version.cuda))
print("GPU's type is {}.".format(torch.cuda.get_device_name(0)))

# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True


thumos_class = {
    7: 'BaseballPitch',
    9: 'BasketballDunk',
    12: 'Billiards',
    21: 'CleanAndJerk',
    22: 'CliffDiving',
    23: 'CricketBowling',
    24: 'CricketShot',
    26: 'Diving',
    31: 'FrisbeeCatch',
    33: 'GolfSwing',
    36: 'HammerThrow',
    40: 'HighJump',
    45: 'JavelinThrow',
    51: 'LongJump',
    68: 'PoleVault',
    79: 'Shotput',
    85: 'SoccerPenalty',
    92: 'TennisSwing',
    93: 'ThrowDiscus',
    97: 'VolleyballSpiking',
}


def IOU(s1, e1, s2, e2):
    if (s2 > e1) or (s1 > e2):
        return 0
    Aor = max(e1, e2) - min(s1, s2)
    Aand = min(e1, e2) - max(s1, s2)
    return float(Aand) / Aor


def Soft_NMS(df, soft_nms_low_thres=0, soft_nms_high_thres=0, nms_threshold=0.85, num_prop=200, trick=True):
    '''
    From BSN code
    :param df:
    :param nms_threshold:
    :return:
    '''
    df = df.sort_values(by="score", ascending=False)

    tstart = list(df.xmin.values[:])
    tend = list(df.xmax.values[:])
    tscore = list(df.score.values[:])

    rstart = []
    rend = []
    rscore = []

    # frost: I use a trick here, remove the detection XDD
    # which is longer than 300
    if trick:
        for idx in range(0, len(tscore)):
            if tend[idx] - tstart[idx] >= 300:
                tscore[idx] = 0

    while len(tscore) > 1 and len(rscore) < num_prop and max(tscore) > 0:
        max_index = tscore.index(max(tscore))
        for idx in range(0, len(tscore)):
            if idx != max_index:
                tmp_iou = IOU(tstart[max_index], tend[max_index], tstart[idx], tend[idx])
                if tmp_iou > soft_nms_low_thres + (soft_nms_high_thres - soft_nms_low_thres):
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
    return newDf


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data


def get_type_data(opt, mode='valid'):
    """Get data of cetrain type. Save information in 'video_dict'. """
    # total video: 19228
    # training:9649, validation:4728, testing:4851.

    video_dict = {}

    videos_info = pd.read_csv(opt.video_info_path)
    with open(opt.video_anno_path) as f:
        videos_anno = json.load(f)
    for i in range(len(videos_info)):
        video_subset = videos_info.subset.values[i]
        if mode in video_subset:
            video_name = videos_info.video.values[i]
            video_anno = videos_anno[video_name]
            video_dict[video_name] = video_anno

    return video_dict


def _gen_detection_video(video_name, video_cls, thu_label_id, opt, num_prop=200, topk=2):
    results_path = opt.output_path + "/results/"
    files = [results_path + f for f in os.listdir(results_path) if video_name in f]
    if len(files) == 0:
        # raise FileNotFoundError('Missing result for video {}'.format(video_name))
        print('Missing result for video {}'.format(video_name))
    else:
        # print('Post processing video {}'.format(video_name))
        pass

    dfs = []  # merge pieces of video
    for snippet_file in files:
        snippet_df = pd.read_csv(snippet_file)
        snippet_df['score'] = snippet_df.boundary_score.values[:] * snippet_df.proposal_score.values[:]
        snippet_df = Soft_NMS(snippet_df, soft_nms_low_thres=opt.soft_nms_low_thres,
                              soft_nms_high_thres=opt.soft_nms_high_thres,
                              nms_threshold=opt.soft_nms_alpha, num_prop=num_prop)
        dfs.append(snippet_df)
    df = pd.concat(dfs)
    if len(df) > 1:
        df = Soft_NMS(df, soft_nms_low_thres=opt.soft_nms_low_thres, soft_nms_high_thres=opt.soft_nms_high_thres,
                      nms_threshold=opt.soft_nms_alpha, num_prop=num_prop)
    df = df.sort_values(by="score", ascending=False)

    # sort video classification
    video_cls_rank = sorted((e, i) for i, e in enumerate(video_cls))
    unet_classes = [thu_label_id[video_cls_rank[-k - 1][1]] + 1 for k in range(topk)]
    unet_scores = [video_cls_rank[-k - 1][0] for k in range(topk)]

    fps = result[video_name]['fps']
    num_frames = result[video_name]['num_frames']
    proposal_list = []
    for j in range(min(num_prop, len(df))):
        for k in range(topk):
            tmp_proposal = {}
            tmp_proposal["label"] = thumos_class[int(unet_classes[k])]
            tmp_proposal["score"] = float(round(df.score.values[j] * unet_scores[k], 6))
            tmp_proposal["segment"] = [float(round(max(0, df.xmin.values[j]) / fps, 1)),
                                       float(round(min(num_frames, df.xmax.values[j]) / fps, 1))]
            proposal_list.append(tmp_proposal)
    return {video_name: proposal_list}


def gen_detection_multicore(opt):
    # get video list
    thumos_test_anno = pd.read_csv(opt.test_anno_path)
    video_list = thumos_test_anno.video.unique()
    thu_label_id = np.sort(thumos_test_anno.type_idx.unique())[1:] - 1  # get thumos14 class id
    thu_video_id = np.array([int(i[-4:]) - 1 for i in video_list])  # -1 is to match python index

    # load video level classification
    cls_data = np.load(opt.uNet_cls_res_path)
    cls_data = cls_data[thu_video_id, :][:, thu_label_id]  # order by video list, output 213x20

    # detection_result
    thumos_gt = pd.read_csv(opt.test_gt_path)
    global result
    result = {
        video:
            {
                'fps': thumos_gt.loc[thumos_gt['video-name'] == video]['frame-rate'].values[0],
                'num_frames': thumos_gt.loc[thumos_gt['video-name'] == video]['video-frames'].values[0]
            }
        for video in video_list
    }

    parallel = Parallel(n_jobs=opt.post_process_thread, prefer="processes")
    detection = parallel(delayed(_gen_detection_video)(video_name, video_cls, thu_label_id, opt)
                         for video_name, video_cls in zip(video_list, cls_data))
    detection_dict = {}
    [detection_dict.update(d) for d in detection]
    output_dict = {"version": "THUMOS14", "results": detection_dict, "external_data": {}}

    with open(opt.result_json_path, "w") as out:
        json.dump(output_dict, out)


def task_adjust_params(opt):
    opt.soft_nms_low_thres = 0.
    opt.soft_nms_high_thres = 0.
    opt.soft_nms_alpha = 0.5


if __name__ == "__main__":

    opt = MyConfig()
    opt.parse()

    task_adjust_params(opt)

    start_time = str(get_cur_time_stamp())
    results_path = os.path.join(opt.output_path, "results")
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    """Load model and data, save scores of all possible proposals without selecting. """

    print("Load the model.")
    model = DCAN(opt)
    model = nn.DataParallel(model).cuda()

    if opt.test_epoch > 0 and opt.checkpoint_path is not None:
        # workdir + test_epoch
        checkpoint = torch.load(opt.checkpoint_path + str(opt.test_epoch) + '_param.pth.tar')
    elif opt.checkpoint_path is not None:
        # ckpt file
        checkpoint = torch.load(opt.checkpoint_path)
    else:
        raise "please set the checkpoint file or work_dir with test_epoch"

    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    test_dataset = MyDataset(opt, subset="validation", mode="inference")
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=opt.num_workers,
                                              pin_memory=True, drop_last=False)

    print('Process video data and save.')
    with torch.no_grad():
        for index_, data in enumerate(test_loader):
            index, video_feature = data

            video_name = test_loader.dataset.video_list[index[0]]
            offset = min(test_loader.dataset.data['indices'][index[0]])
            video_name = video_name + '_{}'.format(math.floor(offset / 250))

            video_feature = video_feature.cuda()
            bm_confidence_map, start, end = model(video_feature)

            bm_confidence_map_reg = bm_confidence_map[0][0].detach().cpu().numpy()
            bm_confidence_map_cls = bm_confidence_map[0][1].detach().cpu().numpy()
            start_g = start[:, 0][0].squeeze(0).detach().cpu().numpy()
            end_g = end[:, 0][0].squeeze(0).detach().cpu().numpy()
            start = start[:, 1][0].squeeze(0).detach().cpu().numpy()
            end = end[:, 1][0].squeeze(0).detach().cpu().numpy()

            new_props = []

            for idx in range(opt.max_duration):
                for jdx in range(opt.temporal_scale):
                    start_index = jdx
                    end_index = start_index + idx + 1
                    if end_index < opt.temporal_scale:
                        xmin = start_index * opt.skip_videoframes + offset  # map [0,99] to frames
                        xmax = end_index * opt.skip_videoframes + offset
                        clr_score = bm_confidence_map_cls[idx, jdx]  # 64, 128
                        reg_score = bm_confidence_map_reg[idx, jdx]
                        new_props.append(
                            [xmin, xmax, start_g[start_index] * end_g[end_index],
                             (clr_score * reg_score) ** opt.proposal_alpha])
            new_props = np.stack(new_props)

            columns = ["xmin", "xmax", "boundary_score", "proposal_score"]
            df = pd.DataFrame(new_props, columns=columns)
            df.to_csv(opt.output_path + "/results/" + video_name + ".csv", index=False)
    """Get all videoes' selected proposals in multi-processing.  """

    print("Get all videoes' selected proposals in multi-processing.")

    print("Detection post processing start")
    gen_detection_multicore(opt)
    print("Detection Post processing finished")

    tious = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    anet_detection = ANETdetection(
        ground_truth_filename=opt.eval_gt_path,
        prediction_filename=opt.result_json_path,
        subset='test', tiou_thresholds=tious,
        check_status=False)
    mAPs, average_mAP = anet_detection.evaluate()
    for (tiou, mAP) in zip(tious, mAPs):
        print("mAP at tIoU {} is {}".format(tiou, mAP))
