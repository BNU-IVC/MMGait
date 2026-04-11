import os
import pickle
import os.path as osp
import torch.utils.data as tordata
import json
from utils import get_msg_mgr
import numpy as np

class DataSet(tordata.Dataset):
    def __init__(self, data_cfg, training):
        """
            seqs_info: the list with each element indicating 
                            a certain gait sequence presented as [label, type, view, paths];
        """
        self.__dataset_parser(data_cfg, training)
        self.cache = data_cfg['cache']
        # print(self.seqs_info)
        self.label_list = [seq_info[0] for seq_info in self.seqs_info]
        self.types_list = [seq_info[1] for seq_info in self.seqs_info]
        self.views_list = [seq_info[2] for seq_info in self.seqs_info]

        self.label_set = sorted(list(set(self.label_list)))
        self.types_set = sorted(list(set(self.types_list)))
        self.views_set = sorted(list(set(self.views_list)))
        self.seqs_data = [None] * len(self)
        self.indices_dict = {label: [] for label in self.label_set}
        for i, seq_info in enumerate(self.seqs_info):
            self.indices_dict[seq_info[0]].append(i)
        if self.cache:
            self.__load_all_data()

    def __len__(self):
        return len(self.seqs_info)

    def __placeholder__(self, path):
        correspond_placeholder = {
            # T = 2
            "depth.pkl": np.zeros((2,3,64,64)),
            "event.pkl": np.zeros((2,3,64,64)),
            "heatmap.pkl": np.zeros((2,2,64,64)),
            "ir_sils.pkl": np.zeros((2,64,64)),
            "ir.pkl": np.zeros((2,3,64,64)),
            "lidar_depth.pkl": [np.zeros((3,64,64)),np.zeros((3,64,64))],
            "lidar_points.pkl": [np.zeros((10,3)),np.zeros((10,3))],
            "pose2d.pkl": np.zeros((2,17,3)),
            "pose3d.pkl": np.zeros((2,17,3)),
            "radar_depth.pkl": np.zeros((2,3,64,64)),
            "radar4d.pkl": [np.zeros((10,3)),np.zeros((10,3))],
            "rgb_sils.pkl": np.zeros((2,64,64)),
            "rgb.pkl": np.zeros((2,3,64,64))
        }
        return correspond_placeholder[path.split("/")[-1]]

    def __loader__(self, modal_valid, paths):

        paths = sorted(paths)
        # print(paths)
        # print(modal_valid)
        data_list = []
        for i in range(len(paths)):
            pth = paths[i]
            if modal_valid[i]: 
                if pth.endswith('.pkl'):
                    with open(pth, 'rb') as f:
                        _ = pickle.load(f)
                    f.close()
                else:
                    raise ValueError('- Loader - just support .pkl !!!')

            else:
                _ = self.__placeholder__(pth)
            data_list.append(_)
        return data_list

    def __getitem__(self, idx):
        if not self.cache:
            data_list = self.__loader__(self.seqs_info[idx][-2], self.seqs_info[idx][-1])
        elif self.seqs_data[idx] is None:
            data_list = self.__loader__(self.seqs_info[idx][-2], self.seqs_info[idx][-1])
            self.seqs_data[idx] = data_list
        else:
            data_list = self.seqs_data[idx]
        seq_info = self.seqs_info[idx]
        
        return data_list, seq_info

    def __load_all_data(self):
        for idx in range(len(self)):
            self.__getitem__(idx)

    def __dataset_parser(self, data_config, training):
        dataset_root = data_config['dataset_root']
        try:
            data_in_use = data_config['data_in_use']  # [n], true or false
        except:
            data_in_use = None

        with open(data_config['dataset_partition'], "rb") as f:
            partition = json.load(f)
        train_set = partition["TRAIN_SET"]
        test_set = partition["TEST_SET"]
        label_list = os.listdir(dataset_root)
        train_set = [label for label in train_set if label in label_list]
        test_set = [label for label in test_set if label in label_list]
        miss_pids = [label for label in label_list if label not in (
            train_set + test_set)]
        msg_mgr = get_msg_mgr()

        def log_pid_list(pid_list):
            if len(pid_list) >= 3:
                msg_mgr.log_info('[%s, %s, ..., %s]' %
                                 (pid_list[0], pid_list[1], pid_list[-1]))
            else:
                msg_mgr.log_info(pid_list)

        if len(miss_pids) > 0:
            msg_mgr.log_debug('-------- Miss Pid List --------')
            msg_mgr.log_debug(miss_pids)
        if training:
            msg_mgr.log_info("-------- Train Pid List --------")
            log_pid_list(train_set)
        else:
            msg_mgr.log_info("-------- Test Pid List --------")
            log_pid_list(test_set)

        def get_seqs_info_list(label_set):
            seqs_info_list = []
            for lab in label_set:
                for typ in sorted(os.listdir(osp.join(dataset_root, lab))):
                    for vie in sorted(os.listdir(osp.join(dataset_root, lab, typ))):
                        seq_info = [lab, typ, vie]
                        seq_path = osp.join(dataset_root, *seq_info)
                        seq_dirs = sorted(os.listdir(seq_path))
                        if seq_dirs != []:
                            seq_dirs = [osp.join(seq_path, dir)
                                        for dir in seq_dirs]
                            if data_in_use is not None:
                                # print(seq_dirs)
                                seq_dirs = [dir for dir, use_bl in zip(
                                    seq_dirs, data_in_use) if use_bl]
                            
                            modal_valid =  self._validate_sequence(seq_dirs)


                            if sum(modal_valid) == len(modal_valid): 
                                seqs_info_list.append([*seq_info, modal_valid, seq_dirs])
                            else:
                                msg_mgr.log_debug(
                                    f'Skip invalid sequence in {lab}-{typ}-{vie}.')
                        else:
                            msg_mgr.log_debug(
                                'Find no .pkl file in %s-%s-%s.' % (lab, typ, vie))
            return seqs_info_list

        self.seqs_info = get_seqs_info_list(
            train_set) if training else get_seqs_info_list(test_set)
    def _validate_sequence(self, paths):
        try:
            modal_valid = []
            for pth in paths:
                if osp.exists(pth) and osp.getsize(pth) > 10:
                    modal_valid.append(True)
                else:
                    modal_valid.append(False)

            return modal_valid
        except Exception:
            return False


