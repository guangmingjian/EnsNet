#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/29 17:39
# @Author  : mingjian
# @Version : 1.0
# @File    : EnsNet.py


import torch
import model
import torch.nn.functional as F
from utils import tools


def entropy_error(vec, delta=1e-7):
    nx = - vec * torch.log2(vec + delta)
    return torch.sum(nx, dim=1)


# def confi_error(pre, truth):
#     l = F.pairwise_distance(pre, truth, p=1)
#     return torch.exp(-l ** 2)

def confi_error(pre, truth):
    l = F.cosine_similarity(pre, truth)
    return l ** 2


class EnsNet(torch.nn.Module):
    """"""

    def __init__(self, in_channels, out_channels, hidden, device, ds_name, model_type="SAGPool", temperature=1e-4,
                 beta=0.5, gama=0, yta=30, alpha=1,graph_norm=True,**kwargs):
        super(EnsNet, self).__init__()
        self.alpha = alpha
        self.yta = yta
        self.epoch = None
        self.gama = gama
        self.beta = beta
        self.nepoch = -1
        """"""
        self.device = device
        self.model_type = model_type
        self.temperature = temperature
        self.out_channels = out_channels
        if self.model_type == "SAGPool":
            gspara = tools.load_json(f"config/Params/SAGPool(H)_{ds_name}.json")
            self.model1 = model.SAGPool(in_channels, out_channels, gspara['num_layers'], hidden, device,
                                         gspara['ratio'], gspara['dropout'], "H",graph_norm)
            gspara = tools.load_json(f"config/Params/SAGPool(G)_{ds_name}.json")
            self.model2 = model.SAGPool(in_channels, out_channels, gspara['num_layers'], hidden, device,
                                         gspara['ratio'], gspara['dropout'], "G",graph_norm)
        elif self.model_type == "ASAP":
            gspara = tools.load_json(f"config/Params/ASAP(H)_{ds_name}.json")
            self.model1 = model.ASAP(in_channels, out_channels, gspara['num_layers'], hidden, device, gspara['ratio'],
                                      gspara['dropout'], "H",graph_norm)
            gspara = tools.load_json(f"config/Params/ASAP(G)_{ds_name}.json")
            self.model2 = model.ASAP(in_channels, out_channels, gspara['num_layers'], hidden, device, gspara['ratio'],
                                      gspara['dropout'], "G",graph_norm)
        elif self.model_type == "SAGASAP":
            gspara = tools.load_json(f"config/Params/SAGPool(H)_{ds_name}.json")
            self.model1 = model.SAGPool(in_channels, out_channels, gspara['num_layers'], hidden, device,
                                         gspara['ratio'], gspara['dropout'], "H",graph_norm)
            gspara = tools.load_json(f"config/Params/ASAP(H)_{ds_name}.json")
            self.model2 = model.ASAP(in_channels, out_channels, gspara['num_layers'], hidden, device, gspara['ratio'],
                                      gspara['dropout'], "H",graph_norm)
        elif self.model_type == "ThreeModel":
            gspara = tools.load_json(f"config/Params/SAGPool(G)_{ds_name}.json")
            self.model1 = model.SAGPool(in_channels, out_channels, gspara['num_layers'], hidden, device,
                                         gspara['ratio'], gspara['dropout'], "G",graph_norm)
            gspara = tools.load_json(f"config/Params/SAGPool(H)_{ds_name}.json")
            self.model2 = model.SAGPool(in_channels, out_channels, gspara['num_layers'], hidden, device,
                                         gspara['ratio'], gspara['dropout'], "H",graph_norm)
            gspara = tools.load_json(f"config/Params/GlobalAttention_{ds_name}.json")
            self.model3 = model.GlobalAttentionNet(in_channels, out_channels, gspara['num_layers'], device, hidden,
                                                    gspara['dropout'],graph_norm)
        else:
            raise ModuleNotFoundError

        self.models = [self.model1, self.model2, self.model3] if self.model_type == "ThreeModel" else [self.model1,
                                                                                                       self.model2]
        self.confiNet = torch.nn.Sequential(torch.nn.Linear(hidden + out_channels, (hidden + out_channels) // 2),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear((hidden + out_channels) // 2, 1), torch.nn.Sigmoid())
        # self.tri_loss = layers.TripletLoss(margin=1)

    def reset_parameters(self):
        for model in self.models:
            model.reset_parameters()

    # @torchsnooper.snoop()
    def forward(self, data):
        if self.training:
            self.epoch = data.epoch
        self.ys = []
        self.confis = []
        for model in self.models:
            self.ys.append(model(data))
            self.confis.append(self.confiNet(torch.cat([self.ys[-1], model.xs], dim=1)))
        self.confis = torch.cat(self.confis, dim=-1)
        # confuse_mask = torch.mean((confis < self.pthe),-1) == 1
        scores = F.softmax(self.confis / self.temperature, dim=1)
        output = 0
        for i, out in enumerate(self.ys):
            output = output + out * scores[:, i].view(-1, 1)
        return output

    def split_dataset(self, ys, y_true):
        ebs_mask = None
        y_pres = None
        for y in ys:
            y = F.softmax(y)
            ymin, _ = torch.min(y, dim=-1)
            ymax, y_pre = torch.max(y, dim=-1)
            yebs = ymax - ymin
            if y_pres is None:
                y_pres = y_pre.view(-1, 1)
            else:
                y_pres = torch.cat([y_pres, y_pre.view(-1, 1)], dim=-1)
            q = torch.exp(torch.tensor(- self.epoch / self.yta)).to(self.device)
            ebs = torch.quantile(yebs, q)
            if ebs_mask is None:
                ebs_mask = (yebs <= ebs).view(-1, 1)
            else:
                ebs_mask = torch.cat([ebs_mask, (yebs <= ebs).view(-1, 1)], dim=-1)

        ebs_nums = torch.sum(ebs_mask, dim=-1)
        y_trues = torch.cat([y_true.view(-1, 1) for _ in range(len(ys))], dim=-1)
        ycorrent_mask = (y_pres == y_trues)
        # *******************************Consistent Correct******************************
        ycc_mask = torch.mean((ycorrent_mask * ~ebs_mask).float(), dim=-1) == 1
        # *******************************Consistent Error******************************
        yce_mask = torch.mean((y_pres != y_trues).float(), dim=-1) == 1
        # *******************************Uncertain******************************
        yconfuse_mask = (torch.mean(ycorrent_mask.float(), dim=-1) > 0) * (ebs_nums > 0)
        # *******************************Inconsistent******************************
        yagainst_mask = (torch.mean(ycorrent_mask.float(), dim=-1) > 0) * (ebs_nums == 0) * (
                torch.mean(ycorrent_mask.float(), dim=-1) < 1)
        return ycc_mask, yce_mask, yconfuse_mask, yagainst_mask


    def loss(self, y_pre, y_true):
        ycc_mask, yce_mask, yconfuse_mask, yagainst_mask = self.split_dataset(self.ys, y_true)
        label_one_hot = torch.nn.functional.one_hot(y_true, self.out_channels).long().to(self.device)
        l_confi = 0
        l_compent = 0
        for i, out in enumerate(self.ys):
            out_soft = F.softmax(out)
            l_confi = l_confi + (confi_error(out_soft, label_one_hot) - self.confis[:, i].view(-1)) ** 2
            l_compent = l_compent + F.cross_entropy(out, y_true)
        weights = torch.ones(len(y_true)).to(self.device)
        weights2 = torch.ones(len(y_true)).to(self.device)
        for mk, factor, factor2 in zip([ycc_mask, yagainst_mask, yconfuse_mask, yce_mask], [1, 1, 2, 3], [1, 2, 1, 1]):
            weights[mk] = factor ** self.beta
            weights2[mk] = factor2
        l_out = F.cross_entropy(y_pre, y_true, reduce=False)
        loss = torch.sum(weights / torch.sum(weights) * l_out) + \
               self.gama * (weights2 / torch.sum(weights2)* l_confi).mean() + self.alpha * l_compent
        return loss
