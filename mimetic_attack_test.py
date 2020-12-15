import os
import argparse
import torch
import torch.nn as nn
import torchvision
import numpy as np
import pickle
from pprint import pprint

from dataloader import get_mnist_loaders, get_cifar10_loaders, get_fmnist_loaders
from dataloader_tiny_imagenet import get_tinyimagenet_loaders
from utils import init_logger, RunningAverageMeter, accuracy
from mimetic_container import test
from adversarial import FGSM, LinfPGD
from adversarial import EpsilonAdversary
from torch.utils.tensorboard import SummaryWriter
from mole.div import div2D as DIV2D
from mole.grad import grad2D as GRAD2D


parser = argparse.ArgumentParser("Attack")

parser.add_argument("--model", type=str, default="res", choices=["res", "ssp2"])
parser.add_argument("--eval", type=str, default="fmnist")
parser.add_argument("--attack", type=str, default="pgd", choices=["fgsm", "pgd"])
parser.add_argument("--archi", type=str, default="cifar10", choices=["imagenet", "cifar10"])
parser.add_argument("--multi", type=eval, default=False)
parser.add_argument("--metric", type=str, default="Linf", choices=["Linf"])
parser.add_argument("--block", type=int, default=3)
parser.add_argument("--load", type=str, default="./")
parser.add_argument("--bsize", type=int, default=100)
parser.add_argument("--norm", type=str, default="g")
parser.add_argument("--eps", type=float, default=8.)
parser.add_argument("--alpha", type=float, default=2.)
parser.add_argument("--iters", type=int, default=10)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--bin", type=eval, default=False)
parser.add_argument("--crit", type=str, default="acc")
parser.add_argument("--network", type=str, default="none")
parser.add_argument("--adv_save", type=eval, default=True)
parser.add_argument("--N", type=int, default=2, choices=list(range(1, 11)))
parser.add_argument("--id", type=str, default='')

args = parser.parse_args()
args.device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")

def l1_distance(tensor1, tensor2) :
    assert tensor1.size() == tensor2.size()
    residual = torch.abs(tensor1-tensor2)
    if len(tensor1.size()) == 4 :
        return residual.sum() / (tensor1.size(0)*tensor1.size(1)*tensor1.size(2)*tensor1.size(3))
    return residual.sum()

def l2_distance(tensor1, tensor2) :
    assert tensor1.size() == tensor2.size()
    residual = (tensor1 - tensor2) ** 2
    if len(tensor1.size()) == 4 :
        return torch.sqrt(torch.sum(residual, (1,2,3))).sum() / tensor1.size(0)
    return residual.sum()


def swish(x):
    return x * torch.sigmoid(x)

class MimeticNet(nn.Module):
    def __init__(self, d_in, d_out):
        super(MimeticNet, self).__init__()
        self.grad = torch.from_numpy(GRAD2D(2, 26, 1, 26, 1)).float().to(args.device)
        self.grad.requires_grad = False
        self.div = torch.from_numpy(DIV2D(2, 26, 1, 26, 1)).float().to(args.device)
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


def get_data():
    try:
        with open(os.path.join(args.load, 'data_to_save.pickle'), 'rb') as handle:
            data = pickle.load(handle)
        if 0 in data.keys():
            print("check.")
            data = {'Training': data}
            if 0 in data.keys():
                print("oops.")
                exit()
        handle.close()
    except:
        data = {}
    return data

def save_data(data, name='att'):
    with open(os.path.join(args.load, 'data_to_save_{name}.pickle'), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()
    print("Data saved.")
                      
def adversarial_attack(model, logger, target_loader, args, modelM, data, **kwargs) :
    logger.info("="*80)
    logger.info("Natural result")
    orig_acc, orig_loss = test(model, target_loader, args.device, modelM, args.N)
    logger.info("Accuracy : {:.4f}".format(orig_acc))
    logger.info("Loss : {:.4f}".format(orig_loss))

    logger.info("-"*80)
    logger.info("Attack parameters : eps={}".format(args.eps))

    repeat = 1
    alpha = None
    k = None
    norm = args.eval
    if args.metric == "Linf" :
        if args.attack == "fgsm" :
            attack_module = FGSM(model, bound=args.eps, norm=norm, device=args.device)
        elif args.attack == "pgd" :
            attack_module = LinfPGD(model, bound=args.eps, step=args.alpha, iters=args.iters, random_start=False, norm=norm, device=args.device)
   
    # Adversarial attack
    writer = SummaryWriter(log_dir=os.path.join(args.load, args.attack+"_"+args.metric+"_"+str(args.eps)))
    device = args.device
    total_correct = 0
    criterion = nn.CrossEntropyLoss().to(device)
    grad = torch.from_numpy(GRAD2D(2, 26, 1, 26, 1)).float().to(device)
    div = torch.from_numpy(DIV2D(2, 26, 1, 26, 1)).float().to(device)
    l1_arr = []
    l2_arr = []
    adv_saver = []
    for i, (x,y) in enumerate(target_loader) :
        if attack_module is not None :
            x_nat = x.detach().clone().to(device)
            x = attack_module.perturb(x.to(device), y.to(device), device=device, modelM=modelM, N=args.N, div=div, grad=grad)
            if repeat != 1 :
                y = torch.cat([y for _ in range(repeat)])

            if args.attack != "ball" :
                l1_dist = l1_distance(attack_module.inverse_normalize(x_nat), attack_module.inverse_normalize(x))
                l2_dist = l2_distance(attack_module.inverse_normalize(x_nat), attack_module.inverse_normalize(x))
                writer.add_scalar("L1", l1_dist, i)
                writer.add_scalar("L2", l2_dist, i)
                l1_arr.append(l1_dist.cpu().numpy())
                l2_arr.append(l2_dist.cpu().numpy())

        x = x.to(device)
        y = y.to(device)
        for _ in range(args.N):
            in_ = x.clone().to(device)
            g = modelM(in_)
            x = DE(x, g, div, grad)
        pred = model(x)
        loss = criterion(pred, y).cpu().detach().numpy()
        predicted_class = torch.argmax(pred.cpu().detach(), dim=1)
        correct = (predicted_class == y.cpu())
        total_correct += torch.sum(correct).item()

        if args.adv_save :
            adv_saver.append((x.cpu(), y.cpu()))

        if args.eval == "cifar10" :
            x_nat = attack_module.inverse_normalize(x_nat)
            x = attack_module.inverse_normalize(x)
        nat_image = torchvision.utils.make_grid(x_nat.cpu(), nrow=5, scale_each=False)
        adv_image = torchvision.utils.make_grid(x.cpu(), nrow=5, scale_each=False)
        writer.add_image("natural_image", nat_image, i)
        writer.add_image("adversarial_image", adv_image, i)

    adv_acc = total_correct / (len(target_loader.dataset) * repeat)
    writer.add_text("natural_acc", str(orig_acc), 1)
    writer.add_text("natural_loss", str(orig_loss), 1)
    writer.add_text("adversarial_acc", str(adv_acc), 1)
    if args.attack != "ball" :
        writer.add_text("L1_distance", str(np.mean(l1_arr)), 1)
        writer.add_text("L2_distance", str(np.mean(l2_arr)), 1)
    if alpha is not None :
        writer.add_text("alpha(stepsize)", str(alpha), 1)
    if k is not None :
        writer.add_text("Iteration", str(k), 1)

    writer.close()
    if args.adv_save :
        if not os.path.exists(os.path.join(args.load, args.attack+"_"+str(args.eps))) :
            os.makedirs(os.path.join(args.load, args.attack+"_"+str(args.eps)))
        with open(os.path.join(args.load, args.attack+"_"+str(args.eps), "adversary.pkl"), "wb") as f :
            pickle.dump(adv_saver,f)

    logger.info("Attacked Accuracy : {:.4f}".format(adv_acc))
    logger.info("Finished")
    logger.info("="*80)
    data = {}
    data[args.attack] = {
                         'orig_acc': orig_acc,
                         'attack_acc': adv_acc,
                         'num_correct': total_correct,
                         }

    save_data(args.load, name=args.attack)
    
if __name__ == "__main__" :
    args.device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    logger = init_logger(logpath=args.load, experiment_name="attack-"+str(args.attack)+"-"+str(args.eps))
    
    if args.eval == "mnist" or args.eval == "fmnist" :
        in_channels = 1
    else :
        in_channels = 3

    if args.eval in ("mnist", "fmnist") :
        from model.mnist import mnist_model
        model = mnist_model(args.model, layers=args.block, norm_type=args.norm)
        modelM = MimeticNet(28**2, 28**2)

    print(f'{args.load}/modelM_acc.pt')
    modelM_dict = torch.load(f'{args.load}/modelM_acc.pt', map_location=str(args.device))
    modelM.load_state_dict(modelM_dict["state_dict"], strict=True)
    modelM.to(args.device)
    modelM.eval()
    logger.info(modelM)

    # if args.network == "none":
    #     model_dict = torch.load(os.path.join(f'./trained_networks/{args.model}_{args.attack}/model_acc.pt'), map_location=str(args.device))
    # else:
    model_dict = torch.load(os.path.join(f'{args.network}/model_acc.pt'), map_location=str(args.device))
    model.load_state_dict(model_dict["state_dict"], strict=True)
    model.to(args.device)
    model.eval()

    # logger.info(model)

    train_loader, test_loader, train_eval_loader = get_fmnist_loaders(test_batch_size=args.bsize)
    # data = get_data()
    # pprint(data)
    data = {}
    adversarial_attack(model, logger, test_loader, args, modelM, data)
