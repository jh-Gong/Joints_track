from torch.utils.data import Dataset
import torch

class Human36MMultiJointsDataset(Dataset):
    def __init__(self, data, seq_len=5):
        self.data_info = []
        self.labels_info = [] 
        self.seq_len = seq_len
        self.data = data
        self.prepare_data()
        
    def prepare_data(self):
        for subject in self.data:
            for action in self.data[subject]:
                for subaction in self.data[subject][action]:
                    for video_id in self.data[subject][action][subaction]:
                        num_frames = self.data[subject][action][subaction][video_id]['root'].shape[0]
                        valid_frames = num_frames - self.seq_len

                        for i in range(valid_frames):
                            self.data_info.append((subject, action, subaction, video_id, i, i + self.seq_len))
                            self.labels_info.append((subject, action, subaction, video_id, i + 1, i + self.seq_len + 1))

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        subject, action, subaction, video_id, start_idx, end_idx = self.data_info[idx]
        # 获取输入数据
        root = torch.tensor(self.data[subject][action][subaction][video_id]['root'][start_idx:end_idx], dtype=torch.float32)
        rotations = torch.tensor(self.data[subject][action][subaction][video_id]['rotations'][start_idx:end_idx], dtype=torch.float32)
        bone_lengths = torch.tensor(self.data[subject][action][subaction][video_id]['bone_lengths'][start_idx], dtype=torch.float32)
        x = {
            'root': root,
            'rotations': rotations,
            'bone_lengths': bone_lengths
        }
        # 获取标签数据
        subject_label, action_label, subaction_label, video_id_label, start_idx_label, end_idx_label = self.labels_info[idx]
        root = torch.tensor(self.data[subject_label][action_label][subaction_label][video_id_label]['root'][start_idx_label:end_idx_label], dtype=torch.float32)
        rotations = torch.tensor(self.data[subject_label][action_label][subaction_label][video_id_label]['rotations'][start_idx_label:end_idx_label], dtype=torch.float32)
        y = {
            'root': root,
            'rotations': rotations
        }

        metadata = (subject, action, subaction, video_id)
        return x, y, metadata