from multiprocessing import cpu_count

from utils.opt_utils import ConfigBase


class MyConfig(ConfigBase):
    def __init__(self, save_txt_flag=True, save_json_flag=False):
        super(MyConfig, self).__init__()

        self.save_txt_flag = save_txt_flag
        self.save_json_flag = save_json_flag

        # mode.
        self.mode = 'train'
        self.train_from_checkpoint = False

        # path.
        self.video_info_path = './data/thumos_annotations/'
        self.video_anno_path = './data/thumos_annotations/'
        self.feature_path = '/home/data4/cg/data/thumos14_tstsn_hdf5/'
        self.evaluation_json_path = './data/thumos_annotations/thumos_gt.json'
        self.result_json_path = './output_thumos/result_proposal.json'
        self.test_anno_path = "./data/thumos_annotations/test_Annotation.csv"
        self.uNet_cls_res_path = "./data/uNet_test.npy"
        self.test_gt_path = "./data/thumos_annotations/thumos14_test_groundtruth.csv"
        self.eval_gt_path = "./data/thumos_annotations/thumos_gt.json"

        self.eval_output_path = "./eval_fig_thumos/"

        self.dataset_name = "thumos14"
        self.save_path = './save/'
        self.log_path = './save/'
        self.checkpoint_path = './save/20210605-2019/'
        self.save_fig_path = './output/evaluation_result.jpg'
        self.output_path = './output_thumos/'

        self.skip_videoframes = 5
        self.max_duration = 64
        self.min_duration = 0
        self.override = True

        # Hyper-parameters.
        self.epochs = 5
        self.batch_size = 16
        self.learning_rate = 0.0004
        self.weight_decay = 1e-4

        self.step_size = 10
        self.step_gamma = 0.1

        self.post_process_thread = cpu_count()
        self.soft_nms_thres = 0.5
        self.proposal_alpha = 1.0

        # Parameters.
        self.temporal_scale = 256
        self.prop_boundary_ratio = 0.5
        self.feat_dim = 2048

        # Model
        self.temporal_dim = 256
        self.mtca_layer_num = 7
        self.num_sample_per_bin = 6
        self.num_sample = 32
        self.sparse_sample = 1
        self.proposal_dim = 128
        self.proposal_hidden_dim = 512

        self.soft_nms_high_thres = 0.9
        self.soft_nms_low_thres = 0.5
        self.soft_nms_alpha = 0.4

        self.num_workers = 16
        self.gpus = "0,1,2,3,4,5,6,7"

        # distributed
        self.rank = 0
        self.local_rank = 0
        self.world_size = 1
