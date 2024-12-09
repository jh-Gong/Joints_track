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
                        video_data = self.data[subject][action][subaction][video_id]['joints_3d']
                        num_frames = video_data.shape[0]
                        valid_frames = num_frames - self.seq_len

                        for i in range(valid_frames):
                            self.data_info.append((subject, action, subaction, video_id, i, i + self.seq_len))
                            self.labels_info.append((subject, action, subaction, video_id, i + 1, i + self.seq_len + 1))

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        subject, action, subaction, video_id, start_idx, end_idx = self.data_info[idx]
        x = torch.tensor(self.data[subject][action][subaction][video_id]['joints_3d'][start_idx:end_idx], dtype=torch.float32)

        subject, action, subaction, video_id, start_idx, end_idx = self.labels_info[idx]
        y = torch.tensor(self.data[subject][action][subaction][video_id]['joints_3d'][start_idx:end_idx], dtype=torch.float32)

        metadata = (subject, action, subaction, video_id)
        return x, y, metadata