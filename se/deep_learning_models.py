import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np

class Deep_model():

    def __init__(self, model_name, cin, ch, bs=32, lr=1e-2, epoch=100, cuda=True):
        self.bs = bs
        self.cuda = cuda
        self.epoch = epoch
        if model_name == 'CNN':
            self.model = CNN_classifier(cin, ch)
        elif model_name == 'DNN':
            self.model = DNN_classifier(cin, ch)
        else:
            print("NO SUCH MODEL")
            exit(-1)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.criterion = nn.BCEWithLogitsLoss()

        if self.cuda:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

    def to_var(self, tensor):
        return Variable(self.to_tensor(tensor), requires_grad=True).float()

    def to_tensor(self, arr):
        return torch.from_numpy(arr).float().cuda()

    def fit(self, x, y):
        for i in range(self.epoch):
            idx = np.arange(x.shape[0])
            np.random.shuffle(idx)

            x_ = x[idx]
            y_ = y[idx]

            for j in range(len(x) // self.bs):
                self.step(x_[j*self.bs: (j+1)*self.bs], y_[j*self.bs: (j+1)*self.bs])

    def step(self, x, y):
        x = self.to_var(x)
        y = self.to_tensor(y)

        if self.cuda:
            x = x.cuda()
            y = y.cuda()

        y_ = self.model(x)

        self.optimizer.zero_grad()
        loss = self.criterion(y_, y)
        loss.backward()
        self.optimizer.step()

    def predict(self, x):
        self.model.eval()
        if not self.cuda:
            x = self.to_tensor(x)
            y = self.model(x).detach().numpy()
        else:
            x = self.to_tensor(x).cuda()
            y = self.model(x).cpu().detach().numpy()

        y[np.where(y>0)] = 1.
        y[np.where(y<0)] = 0.
        return y

class DNN_classifier(nn.Module):
    '''
    three layer DNN
    '''
    def __init__(self, cin, ch):
        super(DNN_classifier, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(cin, ch),
            nn.ReLU(),
            nn.Linear(ch, ch),
            nn.ReLU(),
            nn.Linear(ch, 1)
        )

    def forward(self, x):
        return self.main(x)

class CNN_classifier(nn.Module):
    '''
    two layer CNN + linear classifier
    '''

    def __init__(self, cin, ch):
        super(CNN_classifier, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, ch, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(ch, ch, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
        )
        self.clf = nn.Linear(88*4, 1)

    def forward(self, x):
        x = torch.reshape(x, [x.size()[0], 1, x.size()[1]])
        h = self.feature_extractor(x).reshape([x.size()[0], -1])
        return self.clf(h)