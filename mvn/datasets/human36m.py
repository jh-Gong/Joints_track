from torch.utils.data import Dataset
import torch
import numpy as np
from typing import Tuple

class Human36MMultiJointsDataset(Dataset):
    def __init__(self, data, seq_len=5):
        self.data = []
        self.labels = []
        self.metadata = []  
        self.seq_len = seq_len
        self.prepare_data(data)
        self.data = torch.tensor(np.array(self.data), dtype=torch.float32)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.float32)

    def prepare_data(self, data):
        for subject in data:
            for action in data[subject]:
                for subaction in data[subject][action]:
                    for video_id in data[subject][action][subaction]:
                        video_data = data[subject][action][subaction][video_id]['joints_3d']
                        x, y = self.create_dataset_from_one_video(video_data, self.seq_len)
                        self.data.extend(x)
                        self.labels.extend(y)
                        self.metadata.extend([(subject, action, subaction, video_id)] * len(x))

    def create_dataset_from_one_video(self, data: np.array, seq_len: int = 5) -> Tuple[list, list]:
        num_frames = data.shape[0]
        dataset_x, dataset_y = [], []
        valid_frames = num_frames - seq_len
        
        for i in range(valid_frames):
            input_sequence = data[i:i + seq_len, :]
            output_sequence = data[i + 1:i + seq_len + 1, :]
            dataset_x.append(input_sequence)
            dataset_y.append(output_sequence)
        
        return dataset_x, dataset_y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        metadata = self.metadata[idx]
        return x, y, metadata