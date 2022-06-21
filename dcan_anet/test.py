# coding: utf-8

import json
import multiprocessing as mp
import os

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

matplotlib.use('Agg')
from dataset import MyDataset
from model import DCAN
from opt import MyConfig
from utils.opt_utils import get_cur_time_stamp

from Eval.eval_detection import ANETdetection

# GPU setting.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # range GPU in order
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# Basic test.
print("Pytorch's version is {}.".format(torch.__version__))
print("CUDNN's version is {}.".format(torch.backends.cudnn.version()))
print("CUDA's state is {}.".format(torch.cuda.is_available()))
print("CUDA's version is {}.".format(torch.version.cuda))
print("GPU's type is {}.".format(torch.cuda.get_device_name(0)))


# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True


def IOU(s1, e1, s2, e2):
    if (s2 > e1) or (s1 > e2):
        return 0
    Aor = max(e1, e2) - min(s1, s2)
    Aand = min(e1, e2) - max(s1, s2)
    return float(Aand) / Aor


def Soft_NMS(df, soft_nms_low_thres=0, soft_nms_high_thres=0, nms_threshold=0.85, num_prop=100):
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
                tmp_width = (tend[max_index] - tstart[max_index])
                if tmp_iou > soft_nms_low_thres + (soft_nms_high_thres - soft_nms_low_thres) * tmp_width:
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


def detection_result_gen(opt, video_list, video_dict, results, cuhk_data_score, cuhk_data_action):
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
            df = Soft_NMS(df, soft_nms_low_thres=opt.soft_nms_low_thres, soft_nms_high_thres=opt.soft_nms_high_thres,
                          nms_threshold=opt.soft_nms_alpha)

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


def task_adjust_params(opt):
    opt.soft_nms_low_thres = 0.
    opt.soft_nms_high_thres = 0.
    opt.soft_nms_alpha = 0.5


if __name__ == "__main__":

    opt = MyConfig()
    opt.parse()

    task_adjust_params(opt)

    start_time = str(get_cur_time_stamp())

    if not os.path.exists("output/BMN_results"):
        os.makedirs("output/BMN_results")
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

    test_dataset = MyDataset(opt, mode='valid')
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=opt.num_workers,
                                                  pin_memory=True)

    print('Process video data and save.')
    with torch.no_grad():
        for index_, (
                index, video_feature, gt_iou_map, start_score, end_score, start_short_score,
                end_short_score) in enumerate(tqdm(test_dataloader)):

            video_name = test_dataloader.dataset.video_list[index]

            video_feature = video_feature.cuda()

            bm_confidence_map, start, end = model(video_feature)

            bm_confidence_map_reg = bm_confidence_map[0][0].detach().cpu().numpy()
            bm_confidence_map_cls = bm_confidence_map[0][1].detach().cpu().numpy()
            start_g = start[:, 0][0].squeeze(0).detach().cpu().numpy()
            end_g = end[:, 0][0].squeeze(0).detach().cpu().numpy()
            start = start[:, 1][0].squeeze(0).detach().cpu().numpy()
            end = end[:, 1][0].squeeze(0).detach().cpu().numpy()

            max_start = max(start)
            max_end = max(end)

            # min_start = min(start)
            # min_end = min(end)

            # use BMN post-processing to boost performance

            start_bins = np.zeros(len(start))
            start_bins[0] = 1  # [1,0,0...,0,1]
            for idx in range(1, opt.temporal_scale - 1):
                if start[idx] > start[idx + 1] and start[idx] > start[idx - 1]:
                    start_bins[idx] = 1
                elif start[idx] > (0.5 * max_start):
                    start_bins[idx] = 1

            end_bins = np.zeros(len(end))
            end_bins[-1] = 1
            for idx in range(1, opt.temporal_scale - 1):
                if end[idx] > end[idx + 1] and end[idx] > end[idx - 1]:
                    end_bins[idx] = 1
                elif end[idx] > (0.5 * max_end):
                    end_bins[idx] = 1

            # Iterate over all time conbinations(start & end).
            new_props = []
            for i in range(opt.temporal_scale):
                for j in range(opt.temporal_scale):
                    start_index = i
                    end_index = j + 1
                    if start_index >= end_index:
                        continue

                    clr_score = bm_confidence_map_cls[i, j]
                    reg_score = bm_confidence_map_reg[i, j]
                    if end_index < opt.temporal_scale and start_bins[i] == 1 and end_bins[j] == 1:
                        xmin = start_index / opt.temporal_scale
                        xmax = end_index / opt.temporal_scale
                        clr_score = bm_confidence_map_cls[i, j]
                        reg_score = bm_confidence_map_reg[i, j]
                        new_props.append(
                            [xmin, xmax, start_g[i] * end_g[j], (reg_score * clr_score) ** opt.proposal_alpha])
            new_props = np.stack(new_props)

            columns = ["xmin", "xmax", "cls_score", "reg_score"]
            df = pd.DataFrame(new_props, columns=columns)
            df.to_csv("./output/BMN_results/" + video_name + ".csv", index=False)
    """Get all videoes' selected proposals in multi-processing.  """
    print("Get all videoes' selected proposals in multi-processing.")

    cuhk_data = load_json(opt.video_classification_file)
    cuhk_data_score = cuhk_data["results"]
    cuhk_data_action = cuhk_data["class"]
    video_dict = get_type_data(opt, mode='valid')
    video_list = list(video_dict.keys())
    num_video = len(video_list)
    num_video_per_thread = num_video // opt.post_process_thread

    # Multi-processing.
    results = mp.Manager().dict()
    processes = []
    for index_thread in range(opt.post_process_thread - 1):
        temp_video_list = video_list[index_thread * num_video_per_thread:(index_thread + 1) * num_video_per_thread]
        p = mp.Process(target=detection_result_gen,
                       args=(opt, temp_video_list, video_dict, results, cuhk_data_score, cuhk_data_action))
        p.start()
        processes.append(p)

    # final batch.
    temp_video_list = video_list[(opt.post_process_thread - 1) * num_video_per_thread:]
    p = mp.Process(target=detection_result_gen,
                   args=(opt, temp_video_list, video_dict, results, cuhk_data_score, cuhk_data_action))
    p.start()
    processes.append(p)

    # Make sure that all process is finished.
    for p in processes:
        p.join()
    results = dict(results)
    results_ = {"version": "1.3", "results": results, "external_data": {}}

    with open(opt.result_json_path, 'w') as j:
        json.dump(results_, j)
    print('Already saved the json, waiting for evaluation.')
    """Run evaluation and save figure. """
    anet_det = ANETdetection(

        ground_truth_filename=opt.evaluation_json_path,
        prediction_filename=opt.result_json_path,
        verbose=True,
        check_status=False,
    )

    anet_det.evaluate()
