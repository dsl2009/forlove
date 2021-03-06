from torch import nn
import torch.utils.data as Data

import os
from torch.nn import functional as F
import numpy as np

import torch
class RNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_step1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2, padding=2)
        self.cnn_step2 = nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3, stride=2, padding=1)
        self.rnn=torch.nn.LSTM(
            input_size=48,
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )
        self.liner = nn.Linear(32, 3)

    def forward(self,x):
        x = self.cnn_step1(x)
        x = F.relu(x)
        x = self.cnn_step2(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = x.view(90,-1)
        x = x.unsqueeze(0)
        output,(h_n,c_n)=self.rnn(x)
        x = output[:,-15:,:]
        x = self.liner(x)
        return x


NUM_TIMESTEPS = 90
data = np.load('data.npy')
label = np.load('traget.npy')
mean = [257.717463, 381.688098, 381.688098]
std = [389.969604, 696.205022, 692.336528]

def run(name):
    X = np.zeros((274 - NUM_TIMESTEPS - 15, NUM_TIMESTEPS, 7, 14, 3))
    Y = np.zeros((274 - NUM_TIMESTEPS - 15, 15, 3))
    for i in range(len(data) - NUM_TIMESTEPS - 15):
        if len(data[i + NUM_TIMESTEPS:i + NUM_TIMESTEPS + 15]) == 15:
            X[i] = data[i:i + NUM_TIMESTEPS]
            Y[i] = label[name, i + NUM_TIMESTEPS:i + NUM_TIMESTEPS + 15]
    sp = int(0.9 * X.shape[0])
    Xtrain, Xtest, Ytrain, Ytest = X[0:sp], X[sp:], Y[0:sp], Y[sp:]
    mse = nn.MSELoss(size_average=False)
    batch_size = 1
    net = RNN()
    net.cuda()
    optimzer = torch.optim.Adam(net.parameters(), lr=0.002, amsgrad=True)

    state={'train_loss':100, 'test_loss':100, 'best_loss':100}
    def revert(dts):
        dts[:, 0] = dts[:, 0] * std[0] + mean[0]
        dts[:, 1] = dts[:, 1] * std[1] + mean[1]
        dts[:, 2] = dts[:, 2] * std[2] + mean[2]
        return dts


    def train():
        net.train()
        ls = []
        for k in range(Xtrain.shape[0]):
            ip = Xtrain[k]
            target = Y[k:k + 1]
            ip = np.transpose(ip, [0, 3, 1, 2])
            ip = torch.from_numpy(ip).float()
            target = torch.from_numpy(target).float()

            img, target = torch.autograd.Variable(ip.cuda()), torch.autograd.Variable(target.cuda())
            out = net(img)
            optimzer.zero_grad()

            loss = mse(out, target)
            ls.append(loss.cpu().detach().numpy())

            loss.backward()
            optimzer.step()
            train_loss =  sum(ls) / Xtrain.shape[0]
            state['train_loss'] = train_loss

        print('train_loss', sum(ls) / Xtrain.shape[0])

    def test():
        with torch.no_grad():
            net.eval()
            ls = []
            for k in range(Xtest.shape[0]):
                ip = Xtrain[k]
                target = Y[k:k + 1]
                ip = np.transpose(ip, [0, 3, 1, 2])
                ip = torch.from_numpy(ip).float()
                target = torch.from_numpy(target).float()
                img, target = torch.autograd.Variable(ip.cuda()), torch.autograd.Variable(target.cuda())
                out = net(img)
                loss = mse(out, target)
                ls.append(loss.cpu().detach().numpy())
                ave_val_loss = sum(ls) / Xtest.shape[0]
                state['test_loss'] = ave_val_loss
            print('val_loss', sum(ls) / Xtest.shape[0])

    def pred():
        with torch.no_grad():
            net.eval()
            ip = data[-NUM_TIMESTEPS:]
            ip = np.transpose(ip, [0, 3, 1, 2])
            ip = torch.from_numpy(ip).float()
            img = torch.autograd.Variable(ip.cuda())
            out = net(img)
            result = revert(out.cpu().detach().numpy()[0])
            return result


    min_val_loss = 100
    for step in range(200):
        train()
        test()
        results = pred()
        if min_val_loss>state['test_loss']:
            state['best_loss'] = state['test_loss']
            min_val_loss = state['test_loss']
            torch.save(net.state_dict(), os.path.join('./log', '1' + '.pth'))
            np.save('log/'+str(name)+'.npy', results)
        print(state)


for k in range(98):
    run(k)






