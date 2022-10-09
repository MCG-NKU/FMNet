import cv2 
import numpy as np
import skimage.metrics as sm
from tqdm import tqdm
import os
import argparse

from evaUtil import *

parser = argparse.ArgumentParser()
parser.add_argument('--pd_folder', type=str)
parser.add_argument('--gt_folder', type=str)
parser.add_argument('--norm', type=int)
args = parser.parse_args()

pd_folder = args.pd_folder
gt_folder = args.gt_folder
norm = args.norm

psnr_list = []
ssim_list = []
deltaITP_list = []    
hdrvdp3_list = []
srRIM_list = []

_, _, pd_files = traverse_under_folder(pd_folder)
pd_files.sort()
for i in tqdm(range(len(pd_files))):
    pd_file = pd_files[i]
    gt_file = os.path.join(gt_folder, pd_files[i][len(pd_folder) + 1:])
    pd_ndy_bgr = (cv2.imread(pd_file, cv2.IMREAD_UNCHANGED) / norm).astype(np.float32)
    gt_ndy_bgr = (cv2.imread(gt_file, cv2.IMREAD_UNCHANGED) / norm).astype(np.float32)
    
    pd_ndy_rgb = pd_ndy_bgr[...,::-1]
    gt_ndy_rgb = gt_ndy_bgr[...,::-1]
    
    psnr_list.append(calculate_psnr(img1=pd_ndy_bgr, img2=gt_ndy_bgr))
    ssim_list.append(calculate_ssim(img=pd_ndy_bgr * 255, img2=gt_ndy_bgr * 255))
    deltaITP_list.append(calculate_hdr_deltaITP(img1=pd_ndy_bgr, img2=gt_ndy_bgr))

print('PSNR {:f}\nSSIM {:f}\nDeltaEITP {:f}'.format(np.mean(psnr_list), np.mean(ssim_list), np.mean(deltaITP_list)))