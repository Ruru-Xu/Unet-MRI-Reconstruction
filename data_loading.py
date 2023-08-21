import numpy as np
import mat73
import os
import torch
from torch.utils.data import DataLoader, Dataset
import h5py
# import matplotlib.pyplot as plt
import fastmri
from fastmri.data import transforms as T

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

# load the file
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

# def show_coils(data, slice_nums, cmap=None, vmax = 0.0005):
#     '''
#     plot the figures along the first dims.
#     '''
#     fig = plt.figure()
#     for i, num in enumerate(slice_nums):
#         plt.subplot(1, len(slice_nums), i + 1)
#         plt.imshow(data[num], cmap=cmap,vmax=vmax)

def format_mask(data):
    data = data * (1+1j)
    x_real = data.real
    x_imag = data.imag
    # y = np.array([x_real, x_imag]).astype(np.float).transpose(1,2,0)
    y = np.array([x_real, x_imag]).transpose(1, 2, 0)
    return y

def recons(kspace):
    recon = ifft2c(kspace[:, :, :, 0: 3])
    sx, sy, sz, t = recon.shape
    sx = round(sx / 3)
    sy = round(sy / 2)
    # reconImg = complex_abs(recon)
    # crop the middle 1/6 of the original image for ranking
    cropped_img = np.abs(recon)[recon.shape[0]//2-128:recon.shape[0]//2+128, recon.shape[1]//2-64:recon.shape[1]//2+64]
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

class SliceData(Dataset):

    def __init__(self, base_root, imgtype='cine_lax.mat', masktyppe='cine_lax_mask.mat'):
        self.examples = []
        for folder in os.listdir(base_root):
            path = os.path.join(base_root, folder)
            for root, dirs, nofiles in os.walk(path, topdown=False):
                if folder != 'FullSample':
                    for dir in dirs:
                        p_numdir = os.path.join(path, dir)
                        for fname in os.listdir(p_numdir):
                            if fname == imgtype:
                                path_underSampled = os.path.join(p_numdir, imgtype)
                                path_fully = os.path.join(base_root, 'FullSample/' + dir + '/' + imgtype)
                                path_mask = os.path.join(p_numdir, masktyppe)

                                kspace_sub = readfile2numpy(path_underSampled)
                                kspace_full = readfile2numpy(path_fully)
                                mask = np.float32(mat73.loadmat(path_mask)['mask' + folder.strip('AccFactor')].transpose(1, 0))

                                mask_aug8 = mask.copy()
                                mask_aug8[np.random.choice((1 - mask_aug8)[:, 0].nonzero()[0], 5, replace=False)] = 1
                                kspace_sub_aug8 = kspace_full * mask_aug8

                                mask_aug6 = mask.copy()
                                mask_aug6[np.random.choice((1 - mask_aug6)[:, 0].nonzero()[0], 10, replace=False)] = 1
                                kspace_sub_aug6 = kspace_full * mask_aug6

                                mask_aug10 = mask.copy()
                                mask_aug10[np.random.choice((1 - mask_aug10)[:, 0].nonzero()[0], 15, replace=False)] = 1
                                kspace_sub_aug10 = kspace_full * mask_aug10

                                num_slices = kspace_sub.shape[1]
                                for t in range(3):
                                    kspace2_full = T.to_tensor(kspace_full[t])
                                    img_full = fastmri.ifft2c(kspace2_full)
                                    img_full = img_full[:, img_full.shape[1]//2-64:img_full.shape[1]//2+64, img_full.shape[2]//2-128:img_full.shape[2]//2+128]

                                    kspace2_sub = T.to_tensor(kspace_sub[t])  # Convert from numpy array to pytorch tensor
                                    img_sub = fastmri.ifft2c(kspace2_sub)  # Apply Inverse Fourier Transform to get the complex image
                                    img_sub = img_sub[:, img_sub.shape[1]//2-64:img_sub.shape[1]//2+64, img_sub.shape[2]//2-128:img_sub.shape[2]//2+128]
                                    self.examples += [(img_sub[slice].transpose(0,1).transpose(0,2), img_full[slice].transpose(0,1).transpose(0,2)) for slice in range(num_slices)]

                                    kspace2_sub_aug8 = T.to_tensor(kspace_sub_aug8[t])  # Convert from numpy array to pytorch tensor
                                    img_sub_aug8 = fastmri.ifft2c(kspace2_sub_aug8)  # Apply Inverse Fourier Transform to get the complex image
                                    img_sub_aug8 = img_sub_aug8[:, img_sub_aug8.shape[1]//2-64:img_sub_aug8.shape[1]//2+64, img_sub_aug8.shape[2]//2-128:img_sub_aug8.shape[2]//2+128]
                                    self.examples += [(img_sub_aug8[slice].transpose(0,1).transpose(0,2), img_full[slice].transpose(0,1).transpose(0,2)) for slice in range(num_slices)]

                                    kspace2_sub_aug6 = T.to_tensor(kspace_sub_aug6[t])  # Convert from numpy array to pytorch tensor
                                    img_sub_aug6 = fastmri.ifft2c(kspace2_sub_aug6)  # Apply Inverse Fourier Transform to get the complex image
                                    img_sub_aug6 = img_sub_aug6[:, img_sub_aug6.shape[1]//2-64:img_sub_aug6.shape[1]//2+64, img_sub_aug6.shape[2]//2-128:img_sub_aug6.shape[2]//2+128]
                                    self.examples += [(img_sub_aug6[slice].transpose(0,1).transpose(0,2), img_full[slice].transpose(0,1).transpose(0,2)) for slice in range(num_slices)]

                                    kspace2_sub_aug10 = T.to_tensor(kspace_sub_aug10[t])  # Convert from numpy array to pytorch tensor
                                    img_sub_aug10 = fastmri.ifft2c(kspace2_sub_aug10)  # Apply Inverse Fourier Transform to get the complex image
                                    img_sub_aug10 = img_sub_aug10[:, img_sub_aug10.shape[1]//2-64:img_sub_aug10.shape[1]//2+64, img_sub_aug10.shape[2]//2-128:img_sub_aug10.shape[2]//2+128]
                                    self.examples += [(img_sub_aug10[slice].transpose(0,1).transpose(0,2), img_full[slice].transpose(0,1).transpose(0,2)) for slice in range(num_slices)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        img_sub, img_full = self.examples[i]
        img_sub_norm, sub_mean, sub_std = normalize_instance(img_sub)
        img_full_norm, full_mean, full_std = normalize_instance(img_full)
        max_full = torch.amax(fastmri.complex_abs(img_full.transpose(0,1).transpose(1,2))).unsqueeze(-1).unsqueeze(-1)

        return img_sub_norm, img_full_norm, max_full, full_mean, full_std


def create_fastmri_dataset(args, partition):
    if partition == 'train':
        path = args.data_path / f'TrainingSet'
    elif partition == 'val':
        path = args.data_path / f'val'
    elif partition == 'test':
        path = args.data_path / f'ValidationSet'
    else:
        raise ValueError(f"partition should be in ['train', 'val', 'test'], not {partition}")
    dataset = SliceData(base_root=path)
    print(f'{partition.capitalize()} slices: {len(dataset)}')
    return dataset

def create_data_loader(args, partition, shuffle=False):
    dataset = create_fastmri_dataset(args, partition)
    if partition.lower() == 'train':
        batch_size = args.batch_size
    elif partition.lower() in ['val', 'test']:
        batch_size = args.val_batch_size
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return loader
