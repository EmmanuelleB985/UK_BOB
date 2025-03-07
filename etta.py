import os
import sys
import logging
import cv2
import json
import numpy as np
import scipy
import math
from tqdm import tqdm
from PIL import Image
from copy import deepcopy
import os
import copy
import json
import shutil
import matplotlib.pyplot as plt
from monai.transforms import (
    AddChanneld,
    Compose,
    CenterScaleCropd,
    LoadImaged,
    ScaleIntensityRanged,
    CropForegroundd,
    Orientationd,
    Resized,
    Spacingd,
    EnsureTyped,
)
from monai.networks.nets import SwinUNETR
from run_TotalVibeSegmentator import run_total_seg
from monai.inferers import sliding_window_inference
import nibabel as nib
import warnings
import argparse
import pydicom
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch import optim
from torch.utils.data import DataLoader, random_split
import time


warnings.filterwarnings("ignore")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--task_form', type=str, required=True, help="task form")
parser.add_argument('-o', '--model_out_channels', type=int, required=True, help="model out channels")
parser.add_argument('-w', '--model_weights_path', type=str, required=True, help="model weights path")
parser.add_argument('-d', '--data_path', type=str, required=True, help="data path")
parser.add_argument('-c', '--crop', type=str, required=True, help="crop")
parser.add_argument('-p', '--prediction_path', type=str, required=True, help="prediction path")
parser.add_argument('-v', '--visualization_path', type=str, required=True, help="visualization path")
args = parser.parse_args()
s
task_form = args.task_form
model_out_channels = args.model_out_channels
model_weights_path = args.model_weights_path
data_path = args.data_path
crop = args.crop
prediction_path = args.prediction_path
visualization_path = args.visualization_path


def calculate_ratio(curr_pred_list):
    ratio_list = []
    for curr_pred in curr_pred_list:
        prob_curr = torch.sigmoid(curr_pred)
        entropy_full = softmax_entropy(prob_curr, mode='binary', full=True)

        if torch.sum(prob_curr>=0.5) > 0:
            ratio = 0
            neg = entropy_full[prob_curr<0.5].mean().item()
            ratio += neg

            pos = entropy_full[prob_curr>=0.5].mean().item()
            ratio += pos

            ret = ratio / 2
        else:
            ret = entropy_full.mean().item()

        ratio_list.append(ret)

    return ratio_list

def update_pred(net_list, mean_list, std_list, val_imgs, val_masks, checkpoint_list=None, rho=0.05):
    pred_list, conf_list, sharpness_list, dc_list = [], [], [], []
    
    for idx in range(len(net_list)):
        update_stats(net_list[idx], mean_list[idx], std_list[idx])

        curr_time = time.time()

        # Simple Forward
        pred, conf, dc = make_prediction(net_list[idx], val_imgs, val_masks)

        net_list[idx].eval()
        mask_pred = net_list[idx](val_imgs)

        # Compute sharpness-aware entropy
        loss = softmax_entropy(torch.sigmoid(mask_pred))
        loss.backward()
        for p in net_list[idx].parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any():
                    p.grad = torch.ones(p.grad.shape).cuda()

        state = {}
        with torch.no_grad():
            grad_norm = torch.norm(torch.stack([p.grad.norm(p=2) for p in net_list[idx].parameters() if p.grad is not None]), p=2)
            scale = rho / (grad_norm + 1e-12)

            for n,p in net_list[idx].named_parameters():
                if p.grad is None: continue
                state[n] = p.data.clone()

                e_w = p.grad * scale.to(p)
                p.add_(e_w)
        
        # Maximum prediction
        _, conf_max, _ = make_prediction(net_list[idx], val_imgs, val_masks)
        sharp = conf_max-conf
        
        # Reset weights 
        net_list[idx].load_state_dict(torch.load(checkpoint_list[idx]))

        pred_list.append(pred)
        conf_list.append(conf)
        sharpness_list.append(sharp)
        dc_list.append(dc)

    return pred_list, conf_list, sharpness_list, dc_list


def predict(net, imgs, masks):
    with torch.no_grad():
        pred = net(imgs)
        pred_prob = torch.sigmoid(pred)

        e = softmax_entropy(pred_prob, mode='binary')
        confidence = 1 - e 

        pred_mask = (pred_prob > 0.5).float()
        dc = dice_coeff(pred_mask, masks).item()
        return pred.cpu(), confidence.item(), dc

def softmax_entropy(x, mode='standard', full=False):
    if mode == 'binary':
        ret = -x*torch.log2(x)-(1-x)*torch.log2(1-x)
        ret[x==0] = 0
        ret[x==1] = 0
    elif mode == 'standard':
        ret = -x*torch.log(x)-(1-x)*torch.log(1-x)
        ret[x==0] = 0
        ret[x==1] = 0

    if full:
        return ret
    else:
        return ret.mean()

# Manual record/update running mean
def get_stats(net):
    mean, var = [], []
    for nm, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            mean.append(m.running_mean.clone().detach())
            var.append(m.running_var.clone().detach())
    return mean, var

def fine_tune_BN(net, mean, var):
    count = 0
    for nm, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.running_mean = mean[count].clone().detach().cuda()
            m.running_var = var[count].clone().detach().cuda()
            count += 1

def train_net(net_list,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=1.0,
              encoder=None,
              phase=1,
              vis=False):
    
    if args.dataset == 'btcv':
        prefix = 'base'
        dir_img =  '../btcv/Images_Tr/'
        dir_mask = '../btcv/masks/'

    if args.dataset == 'amos':
        prefix = 'base'
        dir_img =  '../amos/Images_Tr/'
        dir_mask = '../amos/masks/'

    if args.dataset == 'brats':
        prefix = 'base'
        dir_img  = '../dataset/ASNR-MICCAI-BraTS2023-GLI/images'
        dir_mask = '../dataset/ASNR-MICCAI-BraTS2023-GLI/masks'
    
    site_count = [0] + site_count
    site_count = np.cumsum(site_count)

    n_train = len(train)
    train_loader = DataLoader(train, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    val_loaders = []
    for val in vals:
        val_loaders.append(DataLoader(val, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=False))

    # Initialize network with pretrained weights
    checkpoint_list = []
    for idx, net in enumerate(net_list):
        if args.dataset == 'gmsc':
            phase_num = train_phase[-1]
        else:
            phase_num = train_class
        curr_checkpoint =  '../checkpoints/%s_%s_%s_%s_final.pth' % (args.dataset, prefix, phase_num, idx)
        net.load_state_dict(torch.load(curr_checkpoint))
        checkpoint_list.append(curr_checkpoint)
    
    # Store mean and std of each train sample
    ori_mean_list, ori_std_list = [], []
    for idx, net in enumerate(net_list):
        ori_mean, ori_std = get_stats(net)
        ori_mean_list.append(ori_mean)
        ori_std_list.append(ori_std)
    
    # Tracker
    interp_num = 5
    step_size = 1 / interp_num

    names = ['valstats_%.2f' % (i*step_size) for i in range(interp_num+1)]
    names += ['weights_%s' % i for i in range(7)]
    dc_tracker = []
    for n in names:
        dc_tracker.append([])
    
    # Start testing
    val_mean_all_list, val_std_all_list = [], []

    for val_idx, val_batch in enumerate(val_loaders[1]):
        if not vis:
            print('*' * 10)
            print(val_batch['path'])
            print(val_idx, len(val_loaders[1]))
        else:
            if val_idx != 27:
                continue
            print('Vis', val_batch['path'])

        tmp_pred, tmp_conf, tmp_sharpness, tmp_score = [], [], [], []
        tmp_ratio = []

        val_imgs = val_batch['image']
        val_masks = val_batch['mask']

        val_imgs = val_imgs.to(device=device, dtype=torch.float32)
        val_masks = val_masks.to(device=device, dtype=torch.float32)

        val_mean_list, val_std_list = [], []

        for net_idx, net in enumerate(net_list):
            # Obtain val mean and std
            update_stats(net, ori_mean_list[net_idx], ori_std_list[net_idx])
            net.train()
            with torch.no_grad():
                net(val_imgs)
            val_mean, val_std = get_stats(net)
            net.eval()
            update_stats(net, ori_mean_list[net_idx], ori_std_list[net_idx])

            val_mean_list.append(val_mean)
            val_std_list.append(val_std)
        
        input_means = []
        input_stds =  []

        for i in range(interp_num+1):
            mix_avg_mean_list, mix_avg_std_list = [], []
            rate = i * step_size

            for idx in range(len(net_list)):
                tmp_mean = [(1-rate)*m1.cpu()+rate*m2.cpu() for m1,m2 in zip(ori_mean_list[idx], val_mean_list[idx])]
                tmp_std =  [(1-rate)*s1.cpu()+rate*s2.cpu() for s1,s2 in zip(ori_std_list[idx], val_std_list[idx])]
                mix_avg_mean_list.append(tmp_mean)
                mix_avg_std_list.append(tmp_std)
            input_means.append(mix_avg_mean_list)
            input_stds.append(mix_avg_std_list)

        # Run the prediction with different stats
        for stats_idx, (input_mean_list, input_std_list) in enumerate(zip(input_means, input_stds)):
            pred_list, conf_list, sharpness_list, dc_list = update_and_predict(net_list, input_mean_list, input_std_list, val_imgs, val_masks, \
                                                                                   checkpoint_list=checkpoint_list, rho=0.1)
            ratio_list = calculate_ratio(pred_list)
            tmp_ratio.append(ratio_list)

            tmp_pred.append(torch.cat(pred_list, dim=0).unsqueeze(0))
            tmp_conf.append(conf_list)
            tmp_score.append(dc_list)
            tmp_sharpness.append(sharpness_list)

        # N * num_network * shape
        tmp_pred_raw = torch.cat(tmp_pred)
        tmp_pred = torch.sigmoid(tmp_pred_raw)
        
        # N * num_network
        tmp_conf  = torch.tensor(tmp_conf)
        tmp_score = torch.tensor(tmp_score)
        tmp_sharpness = torch.tensor(tmp_sharpness)
        
        if vis:
            for x in range(tmp_pred.shape[0]):
                for y in range(2):
                    curr_prob = tmp_pred[x,y].squeeze()

                    curr_prob_binary = np.uint8((curr_prob.numpy()>0.5) * 255)
                    image = Image.fromarray(curr_prob_binary)
                    image.save('vis/net_%s_choice_%s_%.4f_binary.png' % (y,x,tmp_score[x,y].item()))

                    curr_prob = np.uint8(curr_prob.numpy() * 255)
                    image = Image.fromarray(curr_prob)
                    image.save('vis/net_%s_choice_%s_%.4f_vis.png' % (y,x,tmp_score[x,y].item()))
                    
                    curr_entropy = softmax_entropy(torch.tensor(curr_prob), full=True)
                    curr_entropy = np.uint8(curr_entropy.numpy() * 255)
                    image = Image.fromarray(curr_entropy)
                    image.save('vis/net_%s_choice_%s_%.4f_entropy.png' % (y,x,tmp_score[x,y].item()))

            print("Vis done")
            break
        
        # N * num_network, first column is dummy
        tmp_ratio = torch.tensor(tmp_ratio)

        # Individual output
        dc_list = []
        for i in range(len(tmp_score)):
            dc_tracker[i].append(torch.mean(tmp_score[i]))
            print('%s mean dc: %.4f, confidence: %.4f, diff: %.4f, ratio: %.4f' % (names[i], torch.mean(tmp_score[i]).item(), torch.mean(tmp_conf[i]).item(), torch.mean(tmp_sharpness[i]).item(), torch.mean(tmp_ratio[i]).item()))
        
        # Simple average
        select_idx = len(tmp_pred)

        def get_score(entropy=False, sharpness=False, ratio=False, k=-1, normalize=False):
            weighted_score = []
            for pred_idx in range(len(net_list)):
                curr_net_pred = torch.clone(tmp_pred[:,pred_idx])
                
                if entropy:
                    weights = tmp_conf[:,pred_idx]
                    weight_format = 'entropy'
                else:
                    weights = None
                    weight_format = 'average'

                if normalize:
                    weights = (weights - weights.min()) / (weights.max() - weights.min())
                
                if k > 0:
                    _, weighted_idx = torch.topk(weights, k=k)

                    weighted_pred = []
                    for wi in weighted_idx:
                        weighted_pred.append(curr_net_pred[wi].unsqueeze(0))
                    if normalize:
                        tmp = torch.cat(tmp, dim=0)
                        tmp_weight = torch.tensor(tmp_weight)
                        tmp_weight = (tmp_weight - tmp_weight.min()) / (tmp_weight.max() - tmp_weight.min())
                        tmp_weight = torch.softmax(tmp_weight / 1.0, dim=0)
                        weighted_pred = tmp.T @ tmp_weight
                    else:
                        weighted_pred = torch.cat(weighted_pred, dim=0).mean(0)
                    mask_pred = (weighted_pred > 0.5).float().unsqueeze(0).contiguous()
                else:
                    if weights is not None:
                        weights = torch.softmax(weights / 1.0, dim=0)
                        weighted_pred = curr_net_pred.T @ weights
                        mask_pred = (weighted_pred.T > 0.5).float().unsqueeze(0).contiguous()
                    else:
                        weighted_pred = (curr_net_pred>0.5).float().mean(0)
                        mask_pred = (weighted_pred > 0.5).float().unsqueeze(0).contiguous()

                dc = dice_coeff(mask_pred, val_masks.cpu()).item()
                weighted_score.append(dc)
            
            ret = np.mean(weighted_score)

            print('Score: %.4f. Weight type: %s, Topk: %s, Normalize: %s' % (ret, weight_format, k, normalize))
            return ret

        # simple avg
        ret = get_score()
        dc_tracker[select_idx].append(ret)
        
        # entropy
        ret = get_score(entropy=True)
        dc_tracker[select_idx+1].append(ret)
        
        # entropy min
        ret = get_score(entropy=True, k=1)
        dc_tracker[select_idx+2].append(ret)

        # entropy norm
        ret = get_score(entropy=True, normalize=True)
        dc_tracker[select_idx+4].append(ret)

        # entropy balance
        ret = get_score(ratio=True, normalize=True)
        dc_tracker[select_idx+5].append(ret)
        
    if not vis:
        print('Final performance:')
        for i in range(len(dc_tracker)):
            dc = np.mean(dc_tracker[i])
            print('%s dc mean: %.4f' % (names[i], dc))
        
        for c_idx in range(len(site_count)):
            print('Split %s' % c_idx)
            for i in range(len(dc_tracker)):
                if c_idx == len(site_count) - 1:
                    dc = np.mean(dc_tracker[i][site_count[-1]:])
                else:
                    dc = np.mean(dc_tracker[i][site_count[c_idx]:site_count[c_idx+1]])
                print('%s dc mean: %.4f' % (names[i], dc))


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=32,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=2e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1.0,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--fixed', action='store_true',
                        help='Whether init conv is fixed')
    parser.add_argument('--dataset', default='chest', type=str)
    parser.add_argument('--out_class', default=1, type=int)
    parser.add_argument('--in_class', default=1, type=int)
    parser.add_argument('--phase', default=2, type=int)
    parser.add_argument('--loss', default='ours', type=str)
    parser.add_argument('--vis', default=False, type=bool)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    repeat_num = 10
    net_list = []
    model = SwinUNETR
    # run total vibe segmentator
    model = run_total_seg
    for i in range(repeat_num):
        net_list.append(model(n_channels=args.in_class, n_classes=args.out_class, momentum=1.0).to(device=device))

    train_net(net_list=net_list,
              epochs=args.epochs,
              batch_size=args.batchsize,
              lr=args.lr,
              device=device,
              img_scale=args.scale,
              val_percent=args.val/100, 
              phase=args.phase,
              vis=args.vis)