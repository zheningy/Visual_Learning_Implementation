import torch.nn as nn
import torch.utils.data as data
import numpy as np


class HDMBDataset(data.Dataset):
    def __init__(self, data, num_classes=51, num_frames=10):
        if len(data) == 0:
            raise(RuntimeError("Found 0 action sequences."))
        self.train = True
        if len(data[0].keys()) == 1:
            self.train = False

        self.data = data
        self.num_class = num_classes
        self.num_frame = num_frames

    def __getitem__(self, index):
        """

        :param index(int): Index
        :return: tuple(np.array, np.array): features, labels
        """
        #target = np.zeros([self.num_frame, self.num_class])
        features = self.data[index]['features']
        if self.train:
            target = np.zeros([1, self.num_class])
            target[:, self.data[index]['class_num']] = 1
            return features, target
        else:
            return features

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


class RNN(nn.Module):

    def __init__(self, num_classes=51, input_size=512, hidden_size=128, batch_size=1, num_layers=2, use_gpu=True):
        super(RNN, self).__init__()
        self.LSTM1 = nn.LSTM(input_size, 512, 1, batch_first=True, bidirectional=True, dropout=0.5)
        self.LSTM2 = nn.LSTM(1024, 512, 1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(1024, num_classes)


    def forward(self, x):
        x, _ = self.LSTM1(x)
        x, _ = self.LSTM2(x)
        x = self.fc(x)
        return x.cuda()


# class RNN(nn.Module):
#
#     def __init__(self, num_classes=51, input_size=512, hidden_size=128, batch_size=1, num_layers=2, use_gpu=True):
#         super(RNN, self).__init__()
#         self.num_classes = num_classes
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.batch_size = batch_size
#         self.num_layers = num_layers
#         self.use_gpu = use_gpu
#
#         self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, num_classes)
#         self.relu = nn.ReLU()
#         self.softmax = nn.Softmax()
#
#     def forward(self, embedded, seq_len):
#         hidden = self.init_hidden(seq_len)
#
#         # Pack them up nicely
#
#         embedded = embedded.view(seq_len, self.batch_size, self.input_size)
#         #print(embedded.size())
#         # propagate input through RNN
#         out, hidden = self.rnn(embedded, hidden)
#         # print(out.size())
#         # print(hidden.size())
#         out = self.relu(out)
#         out = self.fc(out[-1])
#         out = self.softmax(out)
#         if self.use_gpu:
#             out = out.cuda()
#
#         return out
#
#     def init_hidden(self, seq_len):
#
#         if self.use_gpu:
#             return Variable(torch.zeros(self.num_layers, seq_len, self.hidden_size), requires_grad=True).cuda()
#
#         return Variable(torch.zeros(self.num_layers, seq_len, self.hidden_size), requires_grad=True)


# class RNN(nn.Module):
#     def __init__(self, num_classes=51, input_size=512, hidden_size=512):
#         super(RNN, self).__init__()
#
#         self.hidden_size = hidden_size
#
#         self.i2h = nn.Sequential(
#                     nn.Linear(input_size + hidden_size, hidden_size),
#                     nn.ReLU()
#                 )
#         self.i2o = nn.Sequential(
#                     nn.Linear(input_size + hidden_size, num_classes),
#                     nn.Softmax()
#                 )
#         self.mediate = nn.Sequential(
#                     nn.Linear(input_size + hidden_size, input_size + hidden_size),
#                     nn.ReLU()
#                 )
#         self.softmax = nn.Softmax()
#
#     def forward(self, x, hidden):
#         combined = torch.cat((x, hidden), 1)
#         m = self.mediate(combined)
#         hidden = self.i2h(m)
#         output = self.i2o(m)
#         return output, hidden
#
#     def initHidden(self):
#         return Variable(torch.zeros(1, self.hidden_size))
