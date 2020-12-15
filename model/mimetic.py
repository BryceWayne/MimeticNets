import torch
import torch.nn as nn
from mole.div import div2D as DIV2D
from mole.grad import grad2D as GRAD2D
from pprint import pprint


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
  def __init__(self, d_in, d_out):
      super(MimeticNet, self).__init__()
      self.grad = torch.from_numpy(GRAD2D(2, 26, 1, 26, 1)).float().to(device)#.type(torch.DoubleTensor)
      self.grad.requires_grad = False
      self.div = torch.from_numpy(DIV2D(2, 26, 1, 26, 1)).float().to(device)
      self.div.requires_grad = False
      self.dt = 1E-2
      self.fc1 = nn.Linear(d_in, 28)
      self.relu = nn.ReLU()
      self.fc2 = nn.Linear(28, d_in)
      self.fc3 = nn.Linear(36*39, 28)
      self.fc4 = nn.Linear(28, d_out)
      self.fcOut = nn.Linear(d_out, 1)

  def forward(self, x):
      x = x.reshape(x.shape[0], -1)
      out = self.fc1(x)
      out = self.relu(out)
      out = self.relu(self.fc2(out))
      out = self.grad@(out.T)         
      out = out.T
      out = self.fc3(out)
      out = self.relu(out)
      out = self.fc4(out)
      out = self.relu(self.fcOut(out))
      out = out.view(-1)
      out = out*torch.mm(self.grad,x.T)
      out = self.div@out
      out = out.T
      out = x + self.dt*out
      out = out.reshape(out.shape[0], 1, 28, 28)
      return out


# if __name__ == '__main__':
#       model = Net(28**2, 28**2)
#       pprint(model)