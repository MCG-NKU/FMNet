import torch
import glob
import numpy as np
import os
from collections import OrderedDict
import argparse

from utils import util
from models.modules.FMNet import FMNet
import options.options as option
import data.util as dataUtil

parser = argparse.ArgumentParser()
parser.add_argument('--test_hdr', type=str)
parser.add_argument('--test_sdr', type=str)
parser.add_argument('--config', type=str)
parser.add_argument('--parameter', type=str)
args = parser.parse_args()

test_hdr = args.test_hdr
test_sdr = args.test_sdr
config = args.config
parameter = args.parameter

opt = option.parse(config, is_train=True)
param = torch.load(parameter)

net = FMNet(in_nc=3, out_nc=opt['network_G']['out_nc'], nf=opt['network_G']['nf'], nb=opt['network_G']['nb'], act_type=opt['network_G']['act_type'], opt=opt['network_G']).cuda()
net.load_state_dict(state_dict=param, strict=True)

sdr_images = glob.glob(os.path.join(test_sdr, '*.png'))
sdr_images.sort()
util.mkdir(opt['path']['val_images'])

for sdr_image in sdr_images:
    LQ_path = sdr_image
    GT_path = os.path.join(test_hdr, os.path.basename(sdr_image))
    print('Processing: {:s} -> {:s}'.format(LQ_path, GT_path))

    # NORM-HWC-BGR
    img_LQ = dataUtil.read_img(None, LQ_path)
    img_GT = dataUtil.read_img(None, GT_path)

    # NORM-HWC-RGB
    img_LQ = img_LQ[:, :, [2, 1, 0]]
    img_GT = img_GT[:, :, [2, 1, 0]]

    H, W, _ = img_LQ.shape

    # NORM-CHW-RGB
    img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float().unsqueeze(0).cuda()
    img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float().unsqueeze(0).cuda()

    net.eval()
    with torch.no_grad():
        img_PD = net(img_LQ)
        visuals = OrderedDict()
        visuals['PD'] = img_PD.detach()[0].float().cpu()
        visuals['GT'] = img_GT.detach()[0].float().cpu()
        img_PD = util.tensor2img(visuals['PD'], out_type=np.uint16)
        img_GT = util.tensor2img(visuals['GT'], out_type=np.uint16)
        util.save_img(img_PD, os.path.join(opt['path']['val_images'], '{:s}.png'.format(os.path.splitext(os.path.basename(sdr_image))[0])))

exit()