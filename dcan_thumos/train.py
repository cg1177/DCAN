# coding: utf-8

import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.backends import cudnn
from tqdm import tqdm

from dataset import MyDataset
from loss import bmn_loss, get_mask
from model import DCAN
from opt import MyConfig
from utils.opt_utils import get_cur_time_stamp


def gpus_list(opts):
    # GPU setting.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # range GPU in order
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    # Basic test.
    print("Pytorch's version is {}.".format(torch.__version__))
    print("CUDNN's version is {}.".format(torch.backends.cudnn.version()))
    print("CUDA's state is {}.".format(torch.cuda.is_available()))
    print("CUDA's version is {}.".format(torch.version.cuda))
    print("GPU's type is {}.".format(torch.cuda.get_device_name(0)))
    return [i for i in range(len(eval("[{}]".format(os.environ["CUDA_VISIBLE_DEVICES"]))))]


# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True


if __name__ == "__main__":

    seed = 1538574472
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # np.random.seed(seed)
    cudnn.benchmark = True

    opt = MyConfig()
    opt.parse()

    start_time = str(get_cur_time_stamp())

    if not os.path.exists(opt.checkpoint_path):
        os.makedirs(opt.checkpoint_path)
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)

    if not os.path.exists(os.path.join(opt.save_path, start_time)):
        os.makedirs(os.path.join(opt.save_path, start_time))

    model = DCAN(opt)
    model = nn.DataParallel(model, device_ids=gpus_list(opt)).cuda()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=opt.learning_rate,
                           weight_decay=opt.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.step_size, gamma=opt.step_gamma)

    if opt.train_from_checkpoint:
        checkpoint = torch.load(opt.checkpoint_path + '9_param.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 1


    def worker_init(worker_id):
        np.random.seed(seed)


    train_dataset = MyDataset(opt, subset="train")
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=opt.batch_size,
                                                   shuffle=True,
                                                   num_workers=opt.num_workers,
                                                   pin_memory=True)

    valid_dataset = MyDataset(opt, subset="validation")
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset,
                                                   batch_size=opt.batch_size * 2,
                                                   shuffle=True,
                                                   num_workers=opt.num_workers,
                                                   pin_memory=True)

    valid_best_loss = float('inf')

    for epoch in tqdm(range(start_epoch, opt.epochs + 1)):

        # Train.
        model.train()
        torch.cuda.empty_cache()
        epoch_train_loss = defaultdict(float)

        for train_iter, train_data in tqdm(enumerate(train_dataloader, start=1)):
            optimizer.zero_grad()

            video_feature, gt_iou_map, start_score, end_score, start_short_score, end_short_score = train_data

            video_feature = video_feature.cuda()
            gt_iou_map = gt_iou_map.cuda()
            start_score = start_score.cuda()
            end_score = end_score.cuda()
            start_short_score = start_short_score.cuda()
            end_short_score = end_short_score.cuda()

            bm_confidence_map, start, end = model(video_feature)

            bm_mask = get_mask(opt.temporal_scale, opt.max_duration).cuda()
            # train_loss: total_loss, tem_loss, pem_reg_loss, pem_cls_loss
            train_loss = bmn_loss(bm_confidence_map, start, end, gt_iou_map, start_score, end_score, start_short_score,
                                  end_short_score, bm_mask)
            train_loss["cost"].backward()

            optimizer.step()

            for key, value in train_loss.items():
                epoch_train_loss[key] += value.item()

        for key, value in epoch_train_loss.items():
            epoch_train_loss[key] /= train_iter
        scheduler.step()

        # Valid.
        epoch_valid_loss = defaultdict(float)
        with torch.no_grad():
            model.eval()
            for valid_iter, valid_data in enumerate(valid_dataloader, start=1):
                video_feature, gt_iou_map, start_score, end_score, start_short_score, end_short_score = valid_data
                video_feature = video_feature.cuda()
                gt_iou_map = gt_iou_map.cuda()
                start_score = start_score.cuda()
                end_score = end_score.cuda()
                start_short_score = start_short_score.cuda()
                end_short_score = end_short_score.cuda()

                bm_confidence_map, start, end = model(video_feature)

                valid_loss = bmn_loss(bm_confidence_map, start, end, gt_iou_map, start_score, end_score,
                                      start_short_score, end_short_score, bm_mask)

                for key, value in valid_loss.items():
                    epoch_valid_loss[key] += value.item()

        for key, value in epoch_valid_loss.items():
            epoch_valid_loss[key] /= valid_iter

        if epoch <= 100 or epoch % 5 == 0:
            content = "Epoch {} Training:".format(epoch)
            for i, (key, value) in enumerate(epoch_train_loss.items()):
                content += "{} - {:.5f}".format(key, value)
                if i < len(epoch_train_loss) - 1:
                    content += ","
            content += "\nEpoch {} Validation:".format(epoch)
            for i, (key, value) in enumerate(epoch_valid_loss.items()):
                content += "{} - {:.5f}".format(key, value)
                if i < len(epoch_valid_loss) - 1:
                    content += ","
            print(content)

            with open(opt.save_path + start_time + '/log.txt', 'a') as f:
                f.write(content + "\n")

        if epoch_valid_loss["cost"] < valid_best_loss or True:
            print(opt.save_path + start_time + '/' + str(epoch) + '_param.pth.tar')
            # Save parameters.
            checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(checkpoint, opt.save_path + start_time + '/' + str(epoch) + '_param.pth.tar')
            valid_best_loss = epoch_valid_loss["cost"]

            # Save whole model.
            # torch.save(model, opt.save_path)
