import os
import time
import torch
import numpy as np
from torch.nn import functional as F
# from ssim_psnr import compute_ssim, compute_psnr
import argparse
import logging
import pathlib
import random
import shutil
import time
import h5py
from collections import defaultdict
import numpy as np
import torch
import torchvision
from torch.nn import functional as F
# from cinenet import (load_recon_model, build_optim)
from unet_model import build_reconstruction_model
# from reconstruction.utils import save_json, , str2none
from loading_testData import create_data_loader, str2bool
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from scipy.io import savemat
from pathlib import Path
from typing import Dict
import fastmri

def save_reconstructions(reconstructions: Dict[str, np.ndarray], out_dir: Path):
    out_dir.mkdir(exist_ok=True, parents=True)
    for fname, recons in reconstructions.items():
        with h5py.File(out_dir / fname, "w") as hf:
            hf.create_dataset("reconstruction", data=recons)


def load_recon_model(args):
    checkpoint = torch.load(args.recon_model_checkpoint)
    recon_args = checkpoint['args']
    recon_model = build_reconstruction_model()
    # No gradients for this model
    for param in recon_model.parameters():
        param.requires_grad = False

    if recon_args.data_parallel:  # if model was saved with data_parallel
        recon_model = torch.nn.DataParallel(recon_model)
    recon_model.load_state_dict(checkpoint['model'])

    del checkpoint
    return recon_args, recon_model

def complex_center_crop(data, shape):
    if not (0 < shape[0] <= data.shape[-2] and 0 < shape[1] <= data.shape[-1]):
        raise ValueError("Invalid shapes.")
    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to]

def evaluate(args):
    recon_args, model = load_recon_model(args)
    recon_args.data_path = args.data_path  # in case model was trained on different machine
    data_loader = create_data_loader(args, 'ValidationSet', shuffle=False)
    model.eval()
    losses = []
    start = time.perf_counter()
    true_avg_loss = 0.
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            input, sx, sy, fpath, mean, std = data
            input = input.to(args.device)
            mean = mean.squeeze(0).to(args.device)
            std = std.squeeze(0).to(args.device)
            recon_slice00 = model(input[:, 0, 0])
            recon_slice10 = model(input[:, 1, 0])
            recon_slice20 = model(input[:, 2, 0])
            recon0 = torch.cat((recon_slice00, recon_slice10, recon_slice20), dim=0)
            recon_slice01 = model(input[:, 0, 1])
            recon_slice11 = model(input[:, 1, 1])
            recon_slice21 = model(input[:, 2, 1])
            recon1 = torch.cat((recon_slice01, recon_slice11, recon_slice21), dim=0)

            recon = torch.cat((recon0.unsqueeze(1), recon1.unsqueeze(1)), dim=1)
            recons = recon*std + mean # norm_data = (data - mean) / (std + 0.0)
            recons_2part = recons.transpose(-2,-3).transpose(-1,-2)
            recons_complex = recons_2part[:,:,:,:,0] + 1j*recons_2part[:,:,:,:,1]
            recons_complex = recons_complex.cpu().detach().numpy()
            recons_trans = np.transpose(recons_complex, (3,2,1,0))
            r1, r2, _, _ = recons_trans.shape
            # recons_crop = recons_trans[r1 // 2 - sx // 6:r1 // 2 + sx // 6, r2 // 2 - sy // 4:r2 // 2 + sy // 4]
            savePath = os.path.join('/home/ilkay/Documents/ruru/CMRxRecon/deep-cine-cardiac-mri/output/singleCoil/unet', fpath[0])
            if (os.path.exists(savePath.split('/cine')[0]) == False):
                os.makedirs(savePath.split('/cine')[0])
            savemat(savePath, {'img4ranking': recons_trans, 'sx': np.double(sx.numpy()), 'sy': np.double(sy.numpy())})
            # savemat(savePath, {'img4ranking': recons_crop})

def main(args):
    evaluate(args)


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=pathlib.Path, default='/home/ilkay/Documents/ruru/CMRxRecon/deep-cine-cardiac-mri/datasets/SingleCoil',
                        help='Path to the dataset')
    parser.add_argument('--val_batch_size', default=1, type=int, help='Mini batch size for validation')
    parser.add_argument('--batch_size', default=1, type=int, help='Mini batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')  # 1e-3 in Kendall&Gal, fastMRI base
    parser.add_argument('--lr_step_size', type=int, default=40,
                        help='Period of learning rate decay')
    parser.add_argument('--lr_gamma', type=float, default=0.1,
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--weight_decay', type=float, default=0,  # 1e-4 in Kendall&Gal (replaces dropout regularis)
                        help='Strength of weight decay regularization')
    parser.add_argument('--report_interval', type=int, default=100, help='Period of loss reporting')
    parser.add_argument('--data_parallel', type=str2bool, default=True,
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--exp_dir', type=pathlib.Path, default='/home/ilkay/Documents/ruru/CMRxRecon/deep-cine-cardiac-mri/output/singleCoil/unet',
                        help='Path where model and results should be saved')
    # parser.add_argument('--exp_dir', type=pathlib.Path, default='/home/ilkay/Documents/ruru/pg_mri/output/debug',
    #                     help='Path where model and results should be saved')
    parser.add_argument('--resume', type=str2bool, default=False,
                        help='If set, resume the training from a previous model checkpoint. '  
                             '"--recon_model_checkpoint" should be set with this')
    parser.add_argument('--recon_model_checkpoint', type=pathlib.Path, default='/home/ilkay/Documents/ruru/CMRxRecon/deep-cine-cardiac-mri/output/singleCoil/unet/cine_lax/complex_2channel_mseSSIM/best_model.pt',
                        help='Path to an existing checkpoint. Used along with "--resume"')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers to use for data loading')
    parser.add_argument('--partition', type=str, default='ValidationSet', choices=['val', 'ValidationSet'],
                        help='Partition to evaluate model on (used with do_train=False).')
    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    main(args)
