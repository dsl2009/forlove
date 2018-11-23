from torch import nn
import torch
class RNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn=torch.nn.LSTM(
            input_size=3,
            hidden_size=20,
            num_layers=1,
            batch_first=True
        )
        self.liner = nn.Linear(20, 3)

    def forward(self,x):
        output,(h_n,c_n)=self.rnn(x)
        x = output[:,-15:,:]
        print(x.size())
        x = self.liner(x)
        return x

x = torch.randn(1, 90, 3)
rnn = RNN()
x = rnn(x)
print(x.size())