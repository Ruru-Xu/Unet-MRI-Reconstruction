import h5py
import numpy as np
import mat73
import os
import torch
from torch.utils.data import DataLoader, Dataset
import fastmri
from fastmri.data import transforms as T

from mat73 import savemat
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')

def complex_abs(data):
    assert data.shape[-1] == 2
    return np.sqrt((data ** 2).sum(axis=-1))

def ifft2c(x):
    S = np.shape(x) # 获取 x 的 shape
    fctr = S[0] * S[1] # 计算缩放因子
    x = np.reshape(x, (S[0], S[1], np.prod(S[2:]))) # 重塑 x
    res = np.zeros(np.shape(x), dtype=complex) # 初始化结果数组
    for n in range(np.shape(x)[2]): # 对每一个通道执行二维傅立叶逆变换
        res[:,:,n] = np.sqrt(fctr) * np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(x[:,:,n])))
    res = np.reshape(res, S) # 重塑结果数组
    return res

def reconsUnderSample(path_kspace, accFactor):
    kspace = mat73.loadmat(path_kspace)['kspace_single_sub' + accFactor]
    recon = ifft2c(kspace[:, :, :, 0: 3])
    sx, sy, sz, t = recon.shape
    sx = round(sx / 3)
    sy = round(sy / 2)
    # reconImg = complex_abs(recon)
    # crop the middle 1/6 of the original image for ranking
    reconImg = np.abs(recon)[:, :, round(sz / 2) - 1:round(sz / 2) + 1, :, ]
    cropped_img = np.abs(reconImg)[reconImg.shape[0]//2-128:reconImg.shape[0]//2+128, reconImg.shape[1]//2-64:reconImg.shape[1]//2+64]
    recons = cropped_img.astype(np.float32)
    return recons, sx, sy

def normolize(recons):
    recons = (recons - torch.amin(recons, (-2, -1)).unsqueeze(1).unsqueeze(1)) / (
            torch.amax(recons, (-2, -1)) - torch.amin(recons, (-2, -1))).unsqueeze(1).unsqueeze(1)
    return recons

def normalize_instance(data):
    mean = torch.mean(data, (-2,-1)).unsqueeze(-1).unsqueeze(-1)
    std = torch.std(data, (-2,-1)).unsqueeze(-1).unsqueeze(-1)
    norm_data = (data - mean) / (std + 0.0)
    return norm_data, mean, std

def readfile2numpy(file_name):
    '''
    read the data from mat and convert to numpy array
    '''
    hf = h5py.File(file_name)
    keys = list(hf.keys())
    assert len(keys) == 1, f"Expected only one key in file, got {len(keys)} instead"
    new_value = hf[keys[0]][()]
    data = new_value["real"] + 1j*new_value["imag"]
    return data

class SliceData(Dataset):

    def __init__(self, base_root, imgtype='cine_lax.mat'):
        self.examples = []
        for folder in os.listdir(base_root):
            path = os.path.join(base_root, folder)
            for root, dirs, nofiles in os.walk(path, topdown=False):
                for dir in dirs:
                    p_numdir = os.path.join(path, dir)
                    for fname in os.listdir(p_numdir):
                        if fname == imgtype:
                            fpath = os.path.join(folder, dir, fname)
                            path_underSampled = os.path.join(p_numdir, imgtype)
                            kspace_sub = readfile2numpy(path_underSampled)
                            kspace2_sub = T.to_tensor(kspace_sub[0:3])  # Convert from numpy array to pytorch tensor
                            img_sub = fastmri.ifft2c(kspace2_sub)  # Apply Inverse Fourier Transform to get the complex image
                            # img_sub_abs = fastmri.complex_abs(img_sub)  # Compute absolute value to get a real image
                            # img_sub = img_sub[:, :, :, :, 0] + 1j * img_sub[:, :, :, :, 1]
                            # img_trans = np.transpose(img_sub.numpy(), (3, 2, 1, 0))
                            t, slice, sy, sx, _ = img_sub.shape
                            self.examples += [(img_sub[:, int(slice/2)-1:int(slice/2)+1,
                                      int(sy/2) - 64:int(sy/2) + 64,
                                      int(sx/2) - 128:int(sx/2) + 128].transpose(-1,-2).transpose(-2,-3), sx, sy, fpath)]


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        img_sub, sx, sy, fpath = self.examples[i]
        img_sub_norm, sub_mean, sub_std = normalize_instance(img_sub)
        return img_sub_norm, sx, sy, fpath, sub_mean, sub_std


def create_fastmri_dataset(args, partition):
    if partition == 'ValidationSet':
        path = args.data_path / f'ValidationSet'
    else:
        raise ValueError(f"partition should be in ['train', 'val', 'test'], not {partition}")
    dataset = SliceData(base_root=path)
    print(f'{partition.capitalize()} slices: {len(dataset)}')
    return dataset

def create_data_loader(args, partition, shuffle=False):
    dataset = create_fastmri_dataset(args, partition)
    batch_size = args.val_batch_size
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return loader
