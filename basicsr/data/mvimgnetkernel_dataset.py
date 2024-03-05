# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
from torch.utils import data as data
from torchvision.transforms.functional import normalize
import torch
from basicsr.data.data_util import (paired_paths_from_folder,
                                    triplet_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file)
from basicsr.data.transforms import augment, paired_random_crop_gaussian, paired_random_crop, triplet_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding

import imageio
import numpy as np 

class MVImgNetKernelDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(MVImgNetKernelDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        self.kernel_folder = opt['dataroot_kernel']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'
        """
        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt[
                'meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.filename_tmpl)
        """
        if isinstance(self.lq_folder, str):
            self.paths = triplet_paths_from_folder(
                [self.lq_folder, self.gt_folder, self.kernel_folder], ['lq', 'gt', 'kernel'],
                self.filename_tmpl)
        elif isinstance(self.lq_folder, list):
            self.paths = []
            for i, _ in enumerate(self.lq_folder):
                lq_folder = self.lq_folder[i]
                gt_folder = self.gt_folder[i]
                kernel_folder = self.kernel_folder[i]

                paths = triplet_paths_from_folder(
                    [lq_folder, gt_folder, kernel_folder], ['lq', 'gt', 'kernel'],
                    self.filename_tmpl)
                self.paths.extend(paths)

    def build_kernel_map(self, kernel):
        window_size = 64
        K, _, H, W = kernel.shape
        
        #kernel_map = torch.zeros(1, H, W, window_size, window_size).float()

        
        coords_y = torch.arange(H)
        coords_x = torch.arange(W)
        coords_yx = torch.stack(torch.meshgrid([coords_y, coords_x], indexing='ij'), dim=0) 
        coords_yx = coords_yx[None].repeat(K, 1, 1, 1)
        
        kernel_x = torch.clip(kernel[:, 0] + window_size//2, 0, window_size-1)
        kernel_y = torch.clip(kernel[:, 1] + window_size//2, 0, window_size-1)

        coords_y = coords_yx[:, 0]
        coords_x = coords_yx[:, 1]
        
        value = torch.ones_like(kernel_x) / K

        indices = torch.stack([coords_y, coords_x, kernel_y, kernel_x])

        value = value.reshape(-1)
        indices = indices.reshape(4, -1)

        kernel_map_coo = torch.sparse_coo_tensor(indices, value, (H, W, window_size, window_size))

        kernel_map = kernel_map_coo.to_dense()
        return kernel_map
        #import imageio 
        #imageio.imwrite('viskernel.png', (255*kernel_map[100, 100]).numpy().astype(np.uint8))
        #import pdb; pdb.set_trace()


    def __getitem__(self, index):

        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)
        scale = self.opt['scale']
        scale_kernel = self.opt['scale_kernel']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        # print('gt path,', gt_path)
        img_bytes = self.file_client.get(gt_path, 'gt')
        
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lq_path = self.paths[index]['lq_path']
        # print(', lq path', lq_path)
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))
        kernel_path = self.paths[index]['kernel_path']
        kernel = torch.load(kernel_path, map_location=torch.device('cpu')).float()

        if torch.sum(torch.isnan(kernel)) >0 or torch.sum(torch.isinf(kernel)) >0:
            print(self.paths[index]['kernel_path'])
            kernel = torch.where(torch.isnan(kernel), torch.zeros_like(kernel), kernel)
            
        
        H, W, _ = img_lq.shape
        img_lq = img_lq[:H//scale_kernel*scale_kernel, :W//scale_kernel*scale_kernel]
        img_gt = img_gt[:H//scale_kernel*scale_kernel, :W//scale_kernel*scale_kernel]
        kernel = kernel[:, :, :H//scale_kernel, :W//scale_kernel]
        
        kernel = kernel + 1e-4*torch.randn(kernel.shape)

        # augmentation for training
        gt_size = self.opt['gt_size']
            # padding
        if self.opt['phase'] == 'train': #or self.opt['phase'] == 'val':
            
            #img_gt, img_lq = padding(img_gt, img_lq, gt_size)
            
            #std_rgb = 0.
            #count = 0
            
            #while(std_rgb < 10. or count > 5):
            max_iter = 5
            for i_crop in range(max_iter):
                # random crop
                #img_gt_crop, img_lq_crop = paired_random_crop_gaussian(img_gt, img_lq, gt_size, scale,
                #                                            gt_path)
                
                img_gt_crop, img_lq_crop, kernel_crop = triplet_random_crop(img_gt, img_lq, kernel, gt_size, scale_kernel,
                                                                                gt_path)
                
                
                img_lq_uint8 = (255*img_lq_crop).astype(np.uint8)
                std_rgb = np.max(np.std(img_lq_uint8, axis=(0,1)))
                
                if std_rgb > 20. or i_crop == max_iter - 1:
                    img_gt_out = img_gt_crop
                    img_lq_out = img_lq_crop
                    break
        else:
            img_gt_crop = img_gt[H//2-gt_size//2:H//2+gt_size//2, W//2-gt_size//2:W//2+gt_size//2]
            img_lq_crop = img_lq[H//2-gt_size//2:H//2+gt_size//2, W//2-gt_size//2:W//2+gt_size//2]
            
            
            kernel_crop = kernel[:, :, (H//2-gt_size//2)//scale_kernel:(H//2+gt_size//2)//scale_kernel, (W//2-gt_size//2)//scale_kernel:(W//2+gt_size//2)//scale_kernel]
            #print(kernel.shape, kernel_crop.shape)# (H//2-gt_size)//scale_kernel, (H//2+gt_size)//scale_kernel)
            #import pdb; pdb.set_trace()
            img_gt_out = img_gt_crop
            img_lq_out = img_lq_crop

        img_gt = img_gt_out
        img_lq = img_lq_out
        kernel = kernel_crop
        #kernel_map = self.build_kernel_map(kernel_crop)
        #kernel_map = kernel_map_crop
        #count += 1
        #print(std_rgb, count)
        
        
        # flip, rotation
        #img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'],
        #                         False)
        #                         #self.opt['use_rot'])
        
        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor

        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)
        kernel = kernel.permute(2, 3, 0, 1)

        # Normalize kernel:
        kernel = kernel / gt_size
        # H, W, w, w to H, W, 1, w, w
        #kernel_map = kernel_map.unsqueeze(2)
        
        """
        idx =  np.random.randint(100000)
        #print(img_lq.permute(1, 2, 0).numpy().shape)
        img_lq_uint8 = (255*img_lq.permute(1, 2, 0).numpy()).astype(np.uint8)
        std = np.max(np.std(img_lq_uint8, axis=(0,1)))
        imageio.imwrite(f'debug/{idx}_blurred_{std_rgb}_{i_crop}.png', img_lq_uint8)
        imageio.imwrite(f'debug/{idx}_gt_{std_rgb}_{i_crop}.png', (255*img_gt.permute(1, 2, 0).numpy()).astype(np.uint8))
        """
        
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
        
        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path,
            #'kernel_map': kernel_map,
            'kernel': kernel,
            'kernel_path': kernel_path,
        }

    def __len__(self):
        return len(self.paths)
