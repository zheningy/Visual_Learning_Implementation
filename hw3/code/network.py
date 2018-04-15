import os
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
import pickle

class HDMBDataset(data.Dataset):
    def __init__(self, data, num_classes=51, num_frames=10, transform=None):
        if len(data) == 0:
            raise(RuntimeError("Found 0 action sequences."))
        self.data = data
        self.num_class = num_classes
        self.num_frame = num_frames

    def __getitem__(self, index):
        """

        :param index(int): Index
        :return: tuple(np.array, np.array): features, labels
        """
        #target = np.zeros([self.num_frame, self.num_class])
        target = np.zeros([1, self.num_class])
        target[:, self.data[index]['class_num']] = 1
        features = self.data[index]['features']

        return features, target

    def __len__(self):
        return len(self.data)


class SimpleNet(nn.Module):
    def __init__(self, num_classes=51):
        super(SimpleNet, self).__init__()

        self.classifer = nn.Sequential(
                        nn.Linear(5120, 512),
                        nn.ReLU(),
                        nn.Linear(512, 256),
                        nn.ReLU(),
                        nn.Linear(256, num_classes),
                        nn.Softmax()
        )

    def forward(self, x):
        x = self.classifer(x)
        return x
