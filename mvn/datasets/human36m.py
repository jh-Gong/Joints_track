'''
Date: 2024-11-27 13:51:21
LastEditors: gjhhh 1377019164@qq.com
LastEditTime: 2024-12-17 22:58:33
Description: example
'''
from torch.utils.data import Dataset
import torch

class Human36MMultiJointsDataset(Dataset):
    def __init__(self, data, seq_len=5, preload=False):
        self.data_info = []
        self.labels_info = []
        self.seq_len = seq_len
        self.data = data
        self.preload = preload

        if self.preload:
            self.preloaded_data = {}
            self.preloaded_labels = {}
            self.preload_data()

        self.prepare_data_info()

    def prepare_data_info(self):
        for subject in self.data:
            for action in self.data[subject]:
                for subaction in self.data[subject][action]:
                    for video_id in self.data[subject][action][subaction]:
                        num_frames = self.data[subject][action][subaction][video_id]['root'].shape[0]
                        valid_frames = num_frames - self.seq_len

                        for i in range(valid_frames):
                            self.data_info.append((subject, action, subaction, video_id, i, i + self.seq_len))
                            self.labels_info.append((subject, action, subaction, video_id, i + 1, i + self.seq_len + 1))

    def preload_data(self):
        print("Preloading data into memory...")
        for subject in self.data:
            self.preloaded_data[subject] = {}
            self.preloaded_labels[subject] = {}
            for action in self.data[subject]:
                self.preloaded_data[subject][action] = {}
                self.preloaded_labels[subject][action] = {}
                for subaction in self.data[subject][action]:
                    self.preloaded_data[subject][action][subaction] = {}
                    self.preloaded_labels[subject][action][subaction] = {}
                    for video_id in self.data[subject][action][subaction]:
                        self.preloaded_data[subject][action][subaction][video_id] = {
                            'root': torch.tensor(self.data[subject][action][subaction][video_id]['root'], dtype=torch.float32),
                            'rotations': torch.tensor(self.data[subject][action][subaction][video_id]['rotations'], dtype=torch.float32),
                            'bone_lengths': torch.tensor(self.data[subject][action][subaction][video_id]['bone_lengths'], dtype=torch.float32)
                        }
                        self.preloaded_labels[subject][action][subaction][video_id] = {
                            'root': torch.tensor(self.data[subject][action][subaction][video_id]['root'], dtype=torch.float32),
                            'rotations': torch.tensor(self.data[subject][action][subaction][video_id]['rotations'], dtype=torch.float32)
                        }
        print("Data preloaded.")

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        subject, action, subaction, video_id, start_idx, end_idx = self.data_info[idx]
        subject_label, action_label, subaction_label, video_id_label, start_idx_label, end_idx_label = self.labels_info[idx]

        if self.preload:
            # 从预加载的数据中获取
            x = {
                'root': self.preloaded_data[subject][action][subaction][video_id]['root'][start_idx:end_idx],
                'rotations': self.preloaded_data[subject][action][subaction][video_id]['rotations'][start_idx:end_idx],
                'bone_lengths': self.preloaded_data[subject][action][subaction][video_id]['bone_lengths'][start_idx]
            }
            y = {
                'root': self.preloaded_labels[subject_label][action_label][subaction_label][video_id_label]['root'][start_idx_label:end_idx_label],
                'rotations': self.preloaded_labels[subject_label][action_label][subaction_label][video_id_label]['rotations'][start_idx_label:end_idx_label]
            }
        else:
            # 实时加载并转换为 Tensor
            x = {
                'root': torch.tensor(self.data[subject][action][subaction][video_id]['root'][start_idx:end_idx], dtype=torch.float32),
                'rotations': torch.tensor(self.data[subject][action][subaction][video_id]['rotations'][start_idx:end_idx], dtype=torch.float32),
                'bone_lengths': torch.tensor(self.data[subject][action][subaction][video_id]['bone_lengths'][start_idx], dtype=torch.float32)
            }
            y = {
                'root': torch.tensor(self.data[subject_label][action_label][subaction_label][video_id_label]['root'][start_idx_label:end_idx_label], dtype=torch.float32),
                'rotations': torch.tensor(self.data[subject_label][action_label][subaction_label][video_id_label]['rotations'][start_idx_label:end_idx_label], dtype=torch.float32)
            }

        metadata = (subject, action, subaction, video_id)
        return x, y, metadata