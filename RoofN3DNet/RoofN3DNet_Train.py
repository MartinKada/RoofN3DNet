# add the parent folder to the python path to access convpoint library
import sys
sys.path.append('../')

import numpy as np
import argparse
from datetime import datetime
import os
import random
from tqdm import tqdm
import time
from sklearn.metrics import confusion_matrix

import torch
import torch.utils.data
import torch.nn.functional as F

import utils.metrics as metrics

import shutil

from pathlib import Path

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# wrap blue / green
def wblue(str):
    return bcolors.OKBLUE+str+bcolors.ENDC
def wgreen(str):
    return bcolors.OKGREEN+str+bcolors.ENDC

from RoofN3DDataset import RoofN3DDataset    
from RoofN3DDataset import angles_from_slopes_residuals

from RoofN3DNet import RoofN3DNet

def run_one_epoch(epoch, net, optimizer, dataloader, num_faces, num_slope_classes, train=True):
    
    if train:
        train_loss = 0

    # initialize variables that compute performance metrics per epoch
    cm_seg = np.zeros((num_faces, num_faces))
    cm_slp = np.zeros((num_slope_classes, num_slope_classes))

    count_buildings = 0.0 # counts all buildings
    count_faces = 0.0 # counts all existing faces (where objectness is 1)

    sum_slope_residuals = 0.0 # sum of all face slope residuals
    num_correct_faces = 0.0
    sum_angles = 0.0
    
    t = tqdm(dataloader, ncols=200+17*train, desc="Epoch {}".format(epoch))

    for item in t:
        
        # unpack dictionary
        pts = item["points"] # [B, N, 3]
        fts = item["features"] # [B, N, 1]
        
        gt_seg = item["points_labels"] # [B, N]
        gt_slope_cls = item["face_slope_classes"] # [B, 4, 1]
        gt_slope_res = item["face_slope_residuals"] # [B, 4, 1]
        gt_face_obj = item["face_objectness"] # [B, 4, 1]
        gt_slope_ang = item["face_slope_angles"] # [B, 4, 1]

        if train:
            optimizer.zero_grad()
            
        # outputs: [B, N, 5], roof_faces: [B, 4, 20]
        pr_seg, pr_roof_faces = net(fts.cuda(), pts.cuda())

        pr_slope_cls, pr_slope_res, pr_face_obj = torch.split(pr_roof_faces, (num_slope_classes, 1, 1), dim=2)            

        # create Boolean (index) mask from roof face objectness (with True values where objectness is equal to 1)
        face_obj_mask = gt_face_obj.view(-1).eq(1)        
                
        if train:
        
            #
            # LOSS (WARNING: if batch size > 1, then tensor dimensions should be reconsidered)
            #

            # cross entropy loss for roof point segmentation (with mean reduction)
            loss_seg = F.cross_entropy(pr_seg.view(-1, num_faces), gt_seg.cuda().view(-1))

            # cross entropy loss for roof face slope classes (where ground truth objectness is true) (with mean reduction)
            loss_slp_cls = F.cross_entropy(pr_slope_cls.view(-1, num_slope_classes)[face_obj_mask], gt_slope_cls.cuda().view(-1)[face_obj_mask]) 

            # l1 loss for roof face slope residuals (where ground truth objectness is true) (with mean reduction)
            loss_slp_res = F.l1_loss(pr_slope_res.view(-1)[face_obj_mask], gt_slope_res.cuda().view(-1)[face_obj_mask])

            # binary cross entropy loss for roof face objectness (with mean reduction)
            loss_obj = F.binary_cross_entropy_with_logits(pr_face_obj.view(-1), gt_face_obj.cuda().view(-1).float())

            loss = loss_seg + loss_slp_cls + loss_slp_res + loss_obj

            #
            # TRAINING STEP
            # 

            loss.backward()
            optimizer.step()

        #
        # PERFORMANCE METRICS
        #

        # roof face objectness mask
        face_obj_mask_np = face_obj_mask.cpu().detach().numpy()

        # count the number of roof faces
        count_faces += np.sum(face_obj_mask_np)            

        # count the number of buildings
        count_buildings += 1.0

        # IoU for roof point segmentation
        cm_seg += confusion_matrix(
            gt_seg.cpu().detach().numpy().copy().ravel(),
            np.argmax(pr_seg.cpu().detach().numpy(), axis=2).copy().ravel(),
            labels=list(range(num_faces)))

        seg_iou, seg_iou_per_class = metrics.stats_iou_per_class(cm_seg)

        # OA for roof face slope classes
        cm_slp += confusion_matrix(
            gt_slope_cls.cpu().detach().numpy().copy().ravel()[face_obj_mask_np],
            np.argmax(pr_slope_cls.cpu().detach().numpy(), axis=2).copy().ravel()[face_obj_mask_np],
            labels=list(range(num_slope_classes)))

        slope_cls_oa = metrics.stats_overall_accuracy(cm_slp)

        # MAE for roof face slope residuals
        sum_slope_residuals += np.sum(F.l1_loss(pr_slope_res.view(-1, 1)[face_obj_mask], gt_slope_res.cuda().view(-1, 1)[face_obj_mask], reduction='none').cpu().detach().numpy())

        slope_res_mae = sum_slope_residuals / count_faces

        # OA for roof face objectness
        num_correct_faces += (torch.round(torch.sigmoid(pr_face_obj.view(-1,1))) == gt_face_obj.cuda()).sum().float().cpu().detach().numpy()

        face_obj_oa = num_correct_faces / ((num_faces-1) * count_buildings)

        # MAE for face slope angles
        sum_angles += np.sum(F.l1_loss(angles_from_slopes_residuals(pr_slope_cls, pr_slope_res).view(-1, 1), gt_slope_ang.cuda().view(-1, 1), reduction='none')[face_obj_mask].cpu().detach().numpy())

        slope_angle_mae = sum_angles / count_faces

        if train:
            train_loss += loss.detach().cpu().item()

            t.set_postfix(ordered_dict = {"PTS_IOU":0, "OBJ_OA":1, "SLOPE_OA":2, "SLOPE_RES":3, "SLOPE_MAE":4, "LOSS":5},
                          SLOPE_RES=wblue(f"{slope_res_mae:.5f}"), 
                          SLOPE_OA=wblue(f"{slope_cls_oa:.5f}"), 
                          PTS_IOU=wblue(f"{seg_iou:.5f}"), 
                          OBJ_OA=wblue(f"{face_obj_oa:.5f}"),
                          SLOPE_MAE=wblue(f"{slope_angle_mae:.5f}"),
                          LOSS=wblue(f"{train_loss/count_buildings:.4e}"))
        else:
            t.set_postfix(ordered_dict = {"PTS_IOU":0, "OBJ_OA":1, "SLOPE_OA":2, "SLOPE_RES":3, "SLOPE_MAE":4},
                      SLOPE_RES=wblue(f"{slope_res_mae:.5f}"), 
                      SLOPE_OA=wblue(f"{slope_cls_oa:.5f}"), 
                      PTS_IOU=wblue(f"{seg_iou:.5f}"), 
                      OBJ_OA=wblue(f"{face_obj_oa:.5f}"), 
                      SLOPE_MAE=wblue(f"{slope_angle_mae:.5f}"))
            
    return seg_iou, face_obj_oa, slope_cls_oa, slope_res_mae, slope_angle_mae
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--savedir", type=str, default="./results")
    parser.add_argument("--epochs", type=int, default=100)
#    parser.add_argument("--batch_size", "-b", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    
    N_FACE_CLASSES = 5
    N_SLOPE_CLASSES = 18

    time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    # create save folder to store training run
    save_folder = os.path.join(args.savedir, "{}".format(time_string))
    os.makedirs(save_folder, exist_ok=True)
    
    # copy Python files (Network, Dataset, Training (this file))
    shutil.copy2('RoofN3DDataset.py', save_folder)
    shutil.copy2('RoofN3DNet.py', save_folder)
    shutil.copy2('RoofN3DNet_Train.py', save_folder)        
    
    # create RoofN3DNet model
    print("Creating RoofN3DNet network...", end="", flush=True)    
    net = RoofN3DNet(in_channels=1, out_channels=20, roof_faces=4, args=args)
    net.cuda()
    print("Done")

    # create optimizer
    print("Creating optimizer...", end="", flush=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    print("Done")
    
    # construct RoofN3D dataset object
    print("Loading RoofN3D dataset...", flush=True)
    ds = RoofN3DDataset(training=True)
    print("Done")

    # create log file
    logs = open(os.path.join(save_folder, "log.csv"), "w")
    logs.write(f"epoch, train_iou, train_obj_oa, train_slope_oa, train_slope_res, train_slope_mae, val_iou, val_obj_oa, val_slope_oa, val_slope_res, val_slope_mae \n")

    # iterate over the specified number of epochs
    for epoch in range(args.epochs):

        #
        # TRAINING
        #
        
        net.train()

        # set dataset object to training split
        ds.set_training()
        train_loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True,
                                                   num_workers=args.threads)
        
        train_seg_iou, train_face_obj_oa, train_slope_cls_oa, train_slope_res_mae, train_slope_mae = run_one_epoch(epoch, net, optimizer, train_loader, N_FACE_CLASSES, N_SLOPE_CLASSES, train=True)

        # save the model
        torch.save(net.state_dict(), os.path.join(save_folder, "state_dict.pth"))
        
        #
        # VALIDATION
        #
        
        # set dataset object to validation split        
        ds.set_validation()
        
        # run for one epoch if validation data exists
        if len(ds) > 0:
            eval_loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False,
                                                       num_workers=args.threads)
            with torch.no_grad():            
                val_seg_iou, val_face_obj_oa, val_slope_cls_oa, val_slope_res_mae, val_slope_mae = run_one_epoch(epoch, net, optimizer, eval_loader, N_FACE_CLASSES, N_SLOPE_CLASSES, train=False)

            # write training and validation metrics to log file
            logs.write(f"{epoch}, {train_seg_iou:.5f}, {train_face_obj_oa:.5f}, {train_slope_cls_oa:.5f}, {train_slope_res_mae:.5f}, {train_slope_mae:.5f}, {val_seg_iou:.5f}, {val_face_obj_oa:.5f}, {val_slope_cls_oa:.5f}, {val_slope_res_mae:.5f}, {val_slope_mae:.5f} \n")
            logs.flush()

    #
    # TESTING
    #

    # set dataset object to testing split
    ds.set_testing()

    # run for one epoch if testing data exists        
    if len(ds) > 0:
        
        print("Testing.")
        
        test_loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False,
                                                   num_workers=args.threads)
        with torch.no_grad():            
            test_seg_iou, test_face_obj_oa, test_slope_cls_oa, test_slope_res_mae, test_slope_mae = run_one_epoch(epoch, net, optimizer, test_loader, N_FACE_CLASSES, N_SLOPE_CLASSES, train=False)
            
    logs.close()
        

if __name__ == '__main__':
    main()
    print('{}-Done.'.format(datetime.now()))