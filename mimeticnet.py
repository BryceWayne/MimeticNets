import os
import argparse
import time
import torch
import torch.nn as nn
import numpy as np
from dataloader import get_fmnist_loaders
from utils import init_logger, RunningAverageMeter, accuracy
from model.mnist import mnist_model
from mimetic_container import mimetic_trainer
from mole.div import div2D as DIV2D
from mole.grad import grad2D as GRAD2D


parser = argparse.ArgumentParser("mimetic")
parser.add_argument("--model", type=str, default="ssp2", choices=["res", "ssp2"])
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--block", type=eval, default=3)
parser.add_argument("--hist", type=eval, default=False)
parser.add_argument("--norm", type=str, default="g")
parser.add_argument("--save", type=str, default="exp")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--adv", type=str, default="pgd", choices=["none", "fgsm", "pgd"])
parser.add_argument("--iters", type=int, default=10)
parser.add_argument("--opt", type=str, default="adam", choices=["sgd", "adam", "rms"])
parser.add_argument("--repeat", type=int, default=5)
parser.add_argument("--init", type=str, default="kn")
parser.add_argument("--N", type=int, default=1, choices=range(1, 11))

args = parser.parse_args()


if __name__ == "__main__" :
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.save = f'./trained_networks/mimeticNet_{args.N}_{args.model}_{args.adv}'
    if os.path.exists(args.save) :
        # raise NameError("previous experiment '{}' already exists!".format(args.save))
        pass
    else:
        os.makedirs(args.save)

    logger = init_logger(logpath=args.save, experiment_name="logs-" + args.model)
    logger.info(args)

    if args.gpu >= 0 :
       args.device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    else :
        args.device = torch.device("cpu")
    device = args.device
    model = mnist_model(args.model, layers=args.block, norm_type=args.norm, init_option=args.init)
    # logger.info(model)
    model.to(args.device)

    old_model = torch.load(f'./trained_networks/{args.model}_{args.adv}/model_acc.pt')
    model.load_state_dict(old_model['state_dict'])
    model.eval()

    def swish(x):
        return x * torch.sigmoid(x)

    class MimeticNet(nn.Module):
        def __init__(self, d_in, d_out):
            super(MimeticNet, self).__init__()
            self.grad = torch.from_numpy(GRAD2D(2, 26, 1, 26, 1)).float().to(device)
            self.grad.requires_grad = False
            self.div = torch.from_numpy(DIV2D(2, 26, 1, 26, 1)).float().to(device)
            self.div.requires_grad = False
            self.dt = 1
            self.fc1 = nn.Linear(d_in, 28)
            self.fc2 = nn.Linear(28, d_in)
            self.fc3 = nn.Linear(36*39, d_out)
            self.fcOut = nn.Linear(d_out, 1)

        def forward(self, x):
            x = x.reshape(x.shape[0], -1)
            out = self.fc2(swish(self.fc1(swish(x))))
            out = self.grad@(out.T) 
            out = out.T
            out = self.fcOut(swish(self.fc3(out)))
            out = out.view(-1)
            return out

    
    def DE(x, g, div, grad):
        batch = x.shape[0]
        x = x.reshape(batch, -1)
        g = g*torch.mm(grad, x.T)
        x = (x + (div@g).T).reshape(batch, 1, 28, 28)
        return x 


    modelM = MimeticNet(28**2, 28**2)
    modelM.to(args.device)
    modelM.train()


    train_loader, test_loader, train_eval_loader = get_fmnist_loaders(batch_size=64, test_batch_size=500)
    loader = {"train_loader": train_loader, "train_eval_loader": train_eval_loader, "test_loader": test_loader}
    if args.opt == "sgd" :
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,50,70], gamma=0.1)
    elif args.opt == "adam" :
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        scheduler = None
    elif args.opt == "rms" :
        optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)
        scheduler = None
    
    adv_train = args.adv if args.adv != "none" else None
    modelM, data = mimetic_trainer(model, logger, loader, args, "fmnist", optimizer, scheduler, adv_train=adv_train, modelM=modelM)


