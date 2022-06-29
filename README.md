# DCAN
The official implementation of the paper "[DCAN: Improving Temporal Action Detection via Dual Context Aggregation](https://arxiv.org/abs/2112.03612)".

## News
(2022/06/21) Code and models are released.


## Abstract

Temporal action detection aims to locate the boundaries of action in the video. The current method based on boundary matching enumerates and calculates all possible boundary matchings to generate proposals. However, these methods neglect the long-range context aggregation in boundary prediction. At the same time, due to the similar semantics of adjacent matchings, local semantic aggregation of densely-generated matchings cannot improve semantic richness and discrimination. In this paper, we propose the endto-end proposal generation method named Dual Context Aggregation Network (DCAN) to aggregate context on two levels, namely, boundary level and proposal level, for generating high-quality action proposals, thereby improving the performance of temporal action detection. Specifically, we design the Multi-Path Temporal Context Aggregation (MTCA) to achieve smooth context aggregation on boundary level and precise evaluation of boundaries. For matching evaluation, Coarse-to-fine Matching (CFM) is designed to aggregate context on the proposal level and refine the matching map from coarse to fine. We conduct extensive experiments on ActivityNet v1.3 and THUMOS-14. DCAN obtains an average mAP of 35.39% on ActivityNet v1.3 and reaches mAP 54.14% at IoU@0.5 on THUMOS-14, which demonstrates DCAN can generate high-quality proposals and achieve state-of-the-art performance.

## Usage 
### Environment
We use Miniconda3 to manage our python environments.
```sh
conda create -n dcan python=3.7
conda activate dcan
conda install matplotlib tqdm joblib h5py
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
```

### Data preparation



**ActivityNet v1.3**: 
We use the TSN feature extracted by the two-stream network. 
The frame interval is set to 16.
Using linear interpolation, each video feature sequence is rescaled to L = 100 snippets.

**THUMOS-14**: 
We use the TSN feature extracted by the TSN network.
The FPS of the feature is consistent with the original videos.
This feature is stored through HDF5 for rapid reading.

---

The TSN features for both datasets are available on the website. We also provide our **download links** to long-term support.

Baidu Netdisk:

[ActivityNet v1.3 and THUMOS-14](https://pan.baidu.com/s/1uY2nnLOBJ71mPl2KwBSBug), code: uvpk

Zenodo: 

[ActivityNet v1.3](https://zenodo.org/record/6650813) , [THUMOS-14](https://zenodo.org/record/6652094)

---

In order to use the downloaded video features, the **feature_path** attributes in the two **opt.py** need to be modified respectively.

### Evaluate our model
Download the checkpoints in the [release](https://github.com/cg1177/DCAN/releases) page.


For example on THUMOS-14, run the followed code.
```sh
cd anet_thumos14
python test.py --checkpoint_path ./save/thumos_4_param.pth.tar
```

Then, you can get the evaluation results:
```
mAP at tIoU 0.1 is 0.7425103702614687
mAP at tIoU 0.2 is 0.7168985921400817
mAP at tIoU 0.3 is 0.6794231339444753
mAP at tIoU 0.4 is 0.6248513734821018
mAP at tIoU 0.5 is 0.5399065183118822
mAP at tIoU 0.6 is 0.4393354468419945
mAP at tIoU 0.7 is 0.3242030486843789
mAP at tIoU 0.8 is 0.1953019732952785
mAP at tIoU 0.9 is 0.06211617080697596
```

For ActivityNet v1.3, the evaluation procedure is similar to THUMOS-14. 

### Training and Testing
For example on THUMOS-14. 

```sh
cd anet_thumos
```

Training model using 4 GPUS (ids=0,1,2,3).

```sh
Python train.py --gpus 0,1,2,3
```

Testing model on a sepific epoch. 
```sh
python test.py --checkpoint_path ./save/xxx-xx/ --test_epoch 4
```

The 'xxx-xx' is the training **work directory** in the 'save' directory named by a timestamp, such as "./save/20220116-1417". 

For ActivityNet v1.3, the training and testing procedure is similar to THUMOS-14. 





## Citation
If this work is helpful for your research, please consider citing the following BibTeX entry.
```
@inproceedings{2022dcan,
  title     = {{DCAN:} Improving Temporal Action Detection via Dual Context Aggregation},
  author    = {Guo Chen and
               Yin{-}Dong Zheng and
               Limin Wang and
               Tong Lu},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2022},
}
```

## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

