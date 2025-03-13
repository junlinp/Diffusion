from torch.utils.data import Dataset
import os
import numpy as np
import einops

def unpickle(file_path):
    import pickle
    with open(file_path, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

CIFAR_BASE="./cifar-10-python"
class CifarDataset(Dataset):
    def __init__(self, train:bool):
        train_path = [
            os.path.join(CIFAR_BASE, "cifar-10-batches-py/data_batch_1"),
           os.path.join(CIFAR_BASE, "cifar-10-batches-py/data_batch_2"),
           os.path.join(CIFAR_BASE, "cifar-10-batches-py/data_batch_3"),
           os.path.join(CIFAR_BASE, "cifar-10-batches-py/data_batch_4"),
           os.path.join(CIFAR_BASE, "cifar-10-batches-py/data_batch_5"),
        ]
        test_path = [
            os.path.join(CIFAR_BASE, "cifar-10-batches-py/test_batch"),
        ]

        self.train = train
        if self.train:
            data_dicts = [unpickle(item) for item in train_path]
            data_list =  [data[b'data'] for data in data_dicts]
            label_list = [data[b'labels'] for data in data_dicts]
            self.data = (np.concatenate(data_list, axis = 0) - 127.5) / 127.5
            self.label = np.concatenate(label_list, axis = 0, dtype = np.int32)
        else:
            data_dicts = [unpickle(item) for item in test_path]
            data_list =  [data[b'data'] for data in data_dicts]
            label_list = [data[b'labels'] for data in data_dicts]
            self.data = (np.concatenate(data_list, axis = 0) -127.5) / 127.5
            self.label = np.concatenate(label_list, axis = 0, dtype=np.int32)

        self.data = einops.rearrange(self.data, "batch_size (channel height width) -> batch_size channel height width", channel = 3, height = 32, width = 32)
        print(f"data.shape : {self.data.shape}")
        print(f"label.shape : {self.label.shape}")
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, index):
        return self.data[index,:], self.label[index]

