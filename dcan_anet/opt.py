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
        self.video_info_path = './data/anet/video_info_new.csv'
        self.video_anno_path = './data/anet/anet_anno_action.json'
        self.feature_path = '/path/to/anet_feature/'
        self.evaluation_json_path = './data/eval/activity_net_1_3_new.json'
        self.result_json_path = './output/result_proposal.json'
        self.save_fig_path = "./output/result_proposal.jpg"
        self.video_classification_file = "./data/anet/cuhk_val_simp_share.json"

        self.dataset_name = "anet13"

        self.save_path = './save/'
        self.log_path = './save/'
        self.checkpoint_path = './save/20210605-2019/'
        self.save_fig_path = './output/evaluation_result.jpg'

        # Hyper-parameters.
        self.epochs = 10
        self.batch_size = 16
        self.learning_rate = 1e-3
        self.weight_decay = 1e-4

        self.step_size = 7
        self.step_gamma = 0.1
        self.proposal_alpha = 1.0

        self.post_process_thread = 8

        # Parameters.
        self.temporal_scale = 100
        self.max_duration = 100
        self.prop_boundary_ratio = 0.5
        self.feat_dim = 400

        # Model
        self.temporal_dim = 256
        self.mtca_layer_num = 6
        self.num_sample_per_bin = 3
        self.num_sample = 32
        self.sparse_sample = 2
        self.proposal_dim = 128
        self.proposal_hidden_dim = 512

        # Soft NMS
        self.soft_nms_high_thres = 0.9
        self.soft_nms_low_thres = 0.5
        self.soft_nms_alpha = 0.5

        self.num_workers = 8
        self.gpus = "0,1,2,3,4,5,6,7"

        # eval
        self.test_epoch = -1

        # distributed
        self.rank = 0
        self.local_rank = 0
        self.world_size = 1
