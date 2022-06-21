import math

import numpy as np
import torch
import torch.nn as nn


class ProposalGenerator(nn.Module):

    def __init__(self, cfg):

        super(ProposalGenerator, self).__init__()

        self.tscale = cfg.temporal_scale
        self.sparse_sample = cfg.sparse_sample
        self.prop_boundary_ratio = cfg.prop_boundary_ratio  # 0.5
        self.num_sample = cfg.num_sample  # 32
        self.num_sample_perbin = cfg.num_sample_per_bin  # 3

        self.hidden_dim_1d = cfg.temporal_dim
        self.hidden_dim_2d = cfg.proposal_dim
        self.hidden_dim_3d = cfg.proposal_hidden_dim

        self._get_sample_mask()

        self.x_1d_p = nn.Sequential(
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_2d, kernel_size=3, padding=1, groups=1),
            nn.ReLU(inplace=True)
        )
        self.x_3d_p = nn.Sequential(
            nn.Conv3d(self.hidden_dim_2d, self.hidden_dim_3d, kernel_size=(self.num_sample, 1, 1),
                      stride=(self.num_sample, 1, 1), groups=1),
            nn.ReLU(inplace=True)
        )

        self.x_2d_p = nn.Sequential(
            nn.Conv2d(self.hidden_dim_3d, self.hidden_dim_2d, kernel_size=1, stride=1, groups=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, base_feature):

        # Proposal Evaluation Module.
        bm_feature_map = self.x_1d_p(base_feature)
        bm_feature_map = self._boundary_matching_layer(bm_feature_map)
        bm_feature_map = self.x_3d_p(bm_feature_map).squeeze(2)
        bmn_proposal_map = self.x_2d_p(bm_feature_map)
        # bmn_proposal_map = self.ps(bm_feature_map)

        return bmn_proposal_map

    def _get_sample_mask(self):

        mask = []

        for end_index in range(self.tscale // self.sparse_sample):
            mask_ = []
            for start_index in range(self.tscale // self.sparse_sample):
                if start_index <= end_index:

                    start_proposal, end_proposal = self.sparse_sample * start_index, self.sparse_sample * (
                            end_index + 1)
                    length_proposal = float(end_proposal - start_proposal) + 1
                    # Expand the proposal and add context feature from both side.
                    start_proposal = start_proposal - length_proposal * self.prop_boundary_ratio
                    end_proposal = end_proposal + length_proposal * self.prop_boundary_ratio
                    mask_.append(self._get_sample_mask_per_proposal(start_proposal, end_proposal,
                                                                    self.tscale, self.num_sample,
                                                                    self.num_sample_perbin))
                else:
                    mask_.append(np.zeros([self.tscale, self.num_sample]))

            # For each end index, add 'tscale' proposal-masks to 'mask_'.
            # before stack, mask_.shape: [start_index][T][N], [100][100][32]
            # after stack, mask_.shape: [T][N][start_index], [100][32][100]
            mask_ = np.stack(mask_, axis=2)
            mask.append(mask_)
        # before stack, mask.shape: [end_index][T][N][start_index], [100][100][32][100]
        # after stack, mask.shape: [T][N][start_index][end_index], [100][32][100][100]
        mask = np.stack(mask, axis=3).astype(np.float32)
        self.sample_mask = nn.Parameter(torch.Tensor(mask).view(self.tscale, -1), requires_grad=False)

    def _get_sample_mask_per_proposal(self, start_proposal, end_proposal, tscale, num_sample, num_sample_perbin):
        length_proposal = end_proposal - start_proposal

        length_sample_perbin = length_proposal / (num_sample * num_sample_perbin - 1.0)
        samples = [start_proposal + length_sample_perbin * i for i in range(num_sample * num_sample_perbin)]

        mask = []
        for i in range(num_sample):
            samples_perbin = samples[i * num_sample_perbin: (i + 1) * num_sample_perbin]
            mask_perbin = np.zeros([tscale])
            for j in samples_perbin:
                j_fractional, j_integral = math.modf(j)
                j_integral = int(j_integral)
                if 0 <= j_integral < (tscale - 1):
                    mask_perbin[j_integral] += 1 - j_fractional
                    mask_perbin[j_integral + 1] += j_fractional
            mask_perbin = 1.0 / num_sample_perbin * mask_perbin
            mask.append(mask_perbin)

        mask = np.stack(mask, axis=1)
        return mask

    def _boundary_matching_layer(self, bm_feature_map):
        feature_size = bm_feature_map.size()  # bm_feature_map.shape: [3][128][100]
        output = torch.matmul(bm_feature_map, self.sample_mask).reshape(feature_size[0], feature_size[1],
                                                                        self.num_sample,
                                                                        self.tscale // self.sparse_sample,
                                                                        self.tscale // self.sparse_sample)
        return output


class CFM(nn.Module):
    def __init__(self, cfg):
        super(CFM, self).__init__()
        self.proposal_gen = ProposalGenerator(cfg)
        self.refineNet = nn.Sequential(
            nn.ConvTranspose2d(cfg.proposal_dim, cfg.proposal_dim, kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            nn.ReLU(),
            nn.Conv2d(cfg.proposal_dim, cfg.proposal_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(cfg.proposal_dim, 2, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        proposal_feature = self.proposal_gen(x)
        proposal = self.refineNet(proposal_feature)
        return proposal
