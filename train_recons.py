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
from unet_model import (load_recon_model, build_optim)
from unet_model import build_reconstruction_model
from data_loading import create_data_loader, str2bool
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import fastmri


def train_epoch(args, epoch, model, data_loader, optimizer):
    model.train()
    avg_loss = 0.
    true_avg_loss = 0.
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)
    criterion = fastmri.SSIMLoss().cuda()
    for iter, data in enumerate(data_loader):
        input, target, max_full, full_mean, full_std = data
        input = input.to(args.device)
        target = target.to(args.device)
        max_full = max_full.to(args.device)
        full_mean = full_mean.to(args.device)
        full_std = full_std.to(args.device)
        # original_max = original_max.unsqueeze(-1).unsqueeze(-1).to(args.device)
        # original_min = original_min.unsqueeze(-1).unsqueeze(-1).to(args.device)

        optimizer.zero_grad()
        recon = model(input)
        # img_sub_abs = fastmri.complex_abs(img_sub) # Compute absolute value to get a real image
        # img_full_abs = fastmri.complex_abs(img_full)
        # recon = recon * (original_max - original_min) + original_min
        # recon = recon[:, 0] + 1j*recon[:, 1]
        # target = target[:, 0] + 1j * target[:, 1]
        loss_mse = F.mse_loss(recon, target)
        unnorm_recon = recon * full_std + full_mean
        unnorm_target = target * full_std + full_mean
        loss_ssim = criterion(fastmri.complex_abs(unnorm_recon.transpose(1,2).transpose(2,3)).unsqueeze(1), fastmri.complex_abs(unnorm_target.transpose(1,2).transpose(2,3)).unsqueeze(1), data_range=max_full.unsqueeze(1))
        loss = (loss_mse + loss_ssim) * 0.5
        loss.backward()
        optimizer.step()

        avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()
        true_avg_loss = (true_avg_loss * iter + loss.mean()) / (iter + 1)
        # writer.add_scalar('TrainLoss', loss.item(), global_step + iter)
        # writer.add_scalar('TrueTrainLossL1', true_avg_loss, global_step + iter)

        if iter % args.report_interval == 0:
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} AvgLoss = {avg_loss:.4g} TrueAvgLoss = {true_avg_loss:.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
        start_iter = time.perf_counter()
    return avg_loss, time.perf_counter() - start_epoch


def evaluate_loss(args, epoch, model, data_loader):
    model.eval()
    losses = []
    start = time.perf_counter()
    true_avg_loss = 0.
    criterion = fastmri.SSIMLoss().cuda()
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            input, target, max_full, full_mean, full_std = data
            input = input.to(args.device)
            target = target.to(args.device)
            max_full = max_full.to(args.device)
            full_mean = full_mean.to(args.device)
            full_std = full_std.to(args.device)
            recon = model(input)
            loss_mse = F.mse_loss(recon, target, reduction='mean')
            l1_loss = (recon - target).abs()

            unnorm_recon = recon * full_std + full_mean
            unnorm_target = target * full_std + full_mean
            loss_ssim = criterion(fastmri.complex_abs(unnorm_recon.transpose(1, 2).transpose(2, 3)).unsqueeze(1),
                             fastmri.complex_abs(unnorm_target.transpose(1, 2).transpose(2, 3)).unsqueeze(1),
                             data_range=max_full.unsqueeze(1))
            loss = 0.7*loss_mse + 0.3*loss_ssim
            true_avg_loss = (true_avg_loss * iter + l1_loss.mean()) / (iter + 1)
            losses.append(loss.item())
        # writer.add_scalar('Dev_Loss', np.mean(losses), epoch)
        # writer.add_scalar('TrueDevLossL1', true_avg_loss, epoch)
    return np.mean(losses), true_avg_loss, time.perf_counter() - start

def save_model(args, exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')


def train_unet(args):
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        recon_model, args, start_epoch, optimizer = load_recon_model(args.recon_model_checkpoint, optim=True)
    else:
        model = build_reconstruction_model()
        if args.data_parallel:
            model = torch.nn.DataParallel(model)
        optimizer = build_optim(args, model.parameters())
        best_dev_loss = 1e9
        start_epoch = 0
    logging.info(args)
    logging.info(model)

    train_loader = create_data_loader(args, 'train', shuffle=True)
    dev_loader = create_data_loader(args, 'val')
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)

    for epoch in range(start_epoch, args.num_epochs):
        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer)
        dev_loss, dev_l1loss, dev_time = evaluate_loss(args, epoch, model, dev_loader)
        scheduler.step()

        is_new_best = dev_loss < best_dev_loss
        best_dev_loss = min(best_dev_loss, dev_loss)
        save_model(args, args.exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best)
        logging.info(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainL1Loss = {train_loss:.4g} DevL1Loss = {dev_l1loss:.4g} '
            f'DevLoss = {dev_loss:.4g} TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s',
        )

def main(args):
    logging.info(args)
    train_unet(args)

def create_arg_parser(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=pathlib.Path, default='/home/ilkay/Documents/ruru/CMRxRecon/deep-cine-cardiac-mri/datasets/SingleCoil/debug',
                        help='Path to the dataset')
    parser.add_argument('--val_batch_size', default=1, type=int, help='Mini batch size for validation')
    parser.add_argument('--batch_size', default=1, type=int, help='Mini batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0008, help='Learning rate')  # 1e-3 in Kendall&Gal, fastMRI base
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
    parser.add_argument('--exp_dir', type=pathlib.Path, default='/home/ilkay/Documents/ruru/CMRxRecon/deep-cine-cardiac-mri/output/singleCoil/unet/cine_lax',
                        help='Path where model and results should be saved')
    # parser.add_argument('--exp_dir', type=pathlib.Path, default='/home/ilkay/Documents/ruru/pg_mri/output/debug',
    #                     help='Path where model and results should be saved')
    parser.add_argument('--resume', type=str2bool, default=False,
                        help='If set, resume the training from a previous model checkpoint. ' 
                             '"--recon_model_checkpoint" should be set with this')
    parser.add_argument('--recon_model_checkpoint', type=pathlib.Path, default='/home/ilkay/Documents/ruru/pg_mri/output/part2/ocmr_recons/best_model.pt',
                        help='Path to an existing checkpoint. Used along with "--resume"')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers to use for data loading')
    parser.add_argument('--partition', type=str, default='val', choices=['val', 'test'],
                        help='Partition to evaluate model on (used with do_train=False).')
    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    main(args)
