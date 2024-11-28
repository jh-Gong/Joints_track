from torch.utils.data import Dataset

class Human36MMultiJointsDataset(Dataset):
    """
    Human36M dataset with multiple joints
    """
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y
    
    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        x = self.data_x[idx]
        y = self.data_y[idx]
        return x, y
