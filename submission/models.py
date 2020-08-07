from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from torch.autograd import Variable


from torch import nn
import torch

import numpy as np

class CNN(nn.Module):
    def __init__(self, cin, ch):
        super(CNN, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(cin, ch, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(ch, ch, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
        )
        self.clf = nn.Linear(306, 1)

    def forward(self, x):
        x = torch.reshape(x, [x.size()[0], 1, x.size()[1]])
        h = self.feature_extractor(x).reshape([x.size()[0], -1])
        return self.clf(h)

class Deep_model():

    def __init__(self, model_name, cin, ch, bs=32, lr=1e-2, epoch=100, cuda=True):
        self.bs = bs
        self.cuda = cuda
        self.epoch = epoch
        self.model = CNN(cin, ch)
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
            # print(x.shape, y.shape)
            idx = np.arange(x.shape[0])
            np.random.shuffle(idx)

            x_ = x[idx]
            y_ = y[idx]

            for j in range(len(x) // self.bs):
                self.step(x_[j*self.bs: (j+1)*self.bs], y_[j*self.bs: (j+1)*self.bs])

    def step(self, x, y):

        # print(len(x), len(y))

        x = self.to_var(x)
        y = self.to_tensor(y)

        if self.cuda:
            x = x.cuda()
            y = y.cuda()

        y_ = self.model(x)
        y_ = y_.view(-1)
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

def classifier(training_data, testing_data, model_name, kwargs):

    train_x = np.array(training_data[0])
    train_y = np.array(training_data[1])

    test_x = np.array(testing_data[0])
    test_y = np.array(testing_data[1])

    if model_name is not 'CNN':
        train_y = train_y.reshape(-1)
        test_y = test_y.reshape(-1)

    if model_name == 'SVM':
        clf = SVC(gamma='auto', **kwargs)
    elif model_name == 'Decision_Tree':
        clf = DecisionTreeClassifier(**kwargs)
    elif model_name == 'Random_Forest':
        clf = RandomForestClassifier(**kwargs)
    elif model_name == 'Ada_Boost':
        clf = AdaBoostClassifier(**kwargs)
    elif model_name == 'CNN':
        clf = Deep_model('CNN', 1, train_x.shape[-1], **kwargs)
    else:
        print('No such model')
        exit(-2)

    clf.fit(train_x, train_y)

    return clf.predict(test_x), test_y, clf