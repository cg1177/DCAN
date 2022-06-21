# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from modules import CFM, MTCA, LocalPointPred, BaseNet


class DCAN(nn.Module):
    def __init__(self, cfg):
        super(DCAN, self).__init__()
        self.temporal_scale = cfg.temporal_scale
        self.proposal_alpha = cfg.proposal_alpha
        self.basenet = BaseNet(cfg.feat_dim, cfg.temporal_dim)
        self.lpp = LocalPointPred(cfg.temporal_dim)
        self.mtca = MTCA(cfg)
        self.cfm = CFM(cfg)

        self.start_pred = nn.Sequential(
            nn.Conv1d(cfg.temporal_dim, cfg.temporal_dim, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv1d(cfg.temporal_dim, 1, kernel_size=1), nn.Sigmoid())
        self.end_pred = nn.Sequential(
            nn.Conv1d(cfg.temporal_dim, cfg.temporal_dim, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv1d(cfg.temporal_dim, 1, kernel_size=1), nn.Sigmoid())

    def forward(self, x):
        base_feature = self.basenet(x)
        start_l, end_l = self.lpp(base_feature)
        temporal_feature = self.mtca(base_feature)

        start = self.start_pred(temporal_feature)
        end = self.end_pred(temporal_feature)

        proposal = self.cfm(base_feature)

        start = torch.stack((start, start_l), dim=1)
        end = torch.stack((end, end_l), dim=1)

        return proposal, start, end
