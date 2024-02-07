# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import cv2
import numpy as np
import os
import sys
from multiprocessing import Pool
from os import path as osp
from tqdm import tqdm
import glob

from basicsr.utils import scandir
from basicsr.utils.create_lmdb import create_lmdb_for_mvimgnet

import configargparse

def parse_dataset(datadir, outdir, repeats):
    split_list = ['mvi_selected0', 'mvi_selected1', 'mvi_selected2', 'mvi_selected3']

    outdir_blur = os.path.join(outdir, 'input')
    outdir_target = os.path.join(outdir, 'target')

    os.makedirs(outdir_blur, exist_ok=True)
    os.makedirs(outdir_target, exist_ok=True)

    all_scenes_list = []
    pool = Pool(len(split_list))
    
    for split in split_list:
        split_dir = os.path.join(datadir, split, '*')
        scene_list = [f for f in glob.glob(split_dir)]
        #pbar = tqdm(len(scene_list), unit='scene', dsec='Copy')
        pool.apply_async(
            worker_cp, args=(scene_list, outdir_blur, outdir_target, repeats)
        )    
    pool.close()
    pool.join()
            

def worker_cp(scene_list, outdir_blur, outdir_target, repeats):
    
        
    for scene in scene_list:
        print(scene)
        render_path = os.path.join(scene, 'train', 'ours_30000')
        

        if not os.path.exists(render_path):
            continue
        
        
        scene_name = scene.split('/')[-1]
        blurred_path = os.path.join(render_path, 'blurred')
        gt_path = os.path.join(render_path, 'gt')

        gt_list = [f for f in glob.glob(os.path.join(gt_path, '*'))] 

        for gt_path_in in gt_list:
            for i_repeat in range(repeats):
                gt_img_name = os.path.splitext(gt_path_in.split('/')[-1])[0]
                blur_img_name = f'{gt_img_name}_{i_repeat:02}.png'
                
                blur_path_in = os.path.join(blurred_path, blur_img_name)
                
                gt_path_out = os.path.join(outdir_target, scene_name+'_'+blur_img_name)
                blur_path_out = os.path.join(outdir_blur, scene_name+'_'+blur_img_name)
                

                os.system(f'cp {blur_path_in} {blur_path_out}')
                os.system(f'cp {gt_path_in} {gt_path_out}')
        
def main():
    opt = {}
    opt['n_thread'] = 20
    opt['compression_level'] = 3

    opt['input_folder'] = './datasets/MVImgNet-blur/input'
    opt['save_folder'] = './datasets/MVImgNet-blur/blur_crops'
    opt['crop_size'] = 512
    opt['step'] = 256
    opt['thresh_size'] = 0
    #extract_subimages(opt)

    opt['input_folder'] = './datasets/MVImgNet-blur/target'
    opt['save_folder'] = './datasets/MVImgNet-blur/sharp_crops'
    opt['crop_size'] = 512
    opt['step'] = 256
    opt['thresh_size'] = 0
    #extract_subimages(opt)

    create_lmdb_for_mvimgnet()


def extract_subimages(opt):
    """Crop images to subimages.

    Args:
        opt (dict): Configuration dict. It contains:
            input_folder (str): Path to the input folder.
            save_folder (str): Path to save folder.
            n_thread (int): Thread number.
    """
    input_folder = opt['input_folder']
    save_folder = opt['save_folder']
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        print(f'mkdir {save_folder} ...')
    else:
        print(f'Folder {save_folder} already exists. Exit.')
        sys.exit(1)

    img_list = list(scandir(input_folder, full_path=True))

    pbar = tqdm(total=len(img_list), unit='image', desc='Extract')
    pool = Pool(opt['n_thread'])
    for path in img_list:
        pool.apply_async(
            worker, args=(path, opt), callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()
    pbar.close()
    print('All processes done.')


def worker(path, opt):
    """Worker for each process.

    Args:
        path (str): Image path.
        opt (dict): Configuration dict. It contains:
            crop_size (int): Crop size.
            step (int): Step for overlapped sliding window.
            thresh_size (int): Threshold size. Patches whose size is lower
                than thresh_size will be dropped.
            save_folder (str): Path to save folder.
            compression_level (int): for cv2.IMWRITE_PNG_COMPRESSION.

    Returns:
        process_info (str): Process information displayed in progress bar.
    """
    crop_size = opt['crop_size']
    step = opt['step']
    thresh_size = opt['thresh_size']
    img_name, extension = osp.splitext(osp.basename(path))

    # remove the x2, x3, x4 and x8 in the filename for DIV2K
    img_name = img_name.replace('x2',
                                '').replace('x3',
                                            '').replace('x4',
                                                        '').replace('x8', '')

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img.ndim == 2:
        h, w = img.shape
    elif img.ndim == 3:
        h, w, c = img.shape
    else:
        raise ValueError(f'Image ndim should be 2 or 3, but got {img.ndim}')

    h_space = np.arange(0, h - crop_size + 1, step)
    if h - (h_space[-1] + crop_size) > thresh_size:
        h_space = np.append(h_space, h - crop_size)
    w_space = np.arange(0, w - crop_size + 1, step)
    if w - (w_space[-1] + crop_size) > thresh_size:
        w_space = np.append(w_space, w - crop_size)

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            cropped_img = img[x:x + crop_size, y:y + crop_size, ...]
            cropped_img = np.ascontiguousarray(cropped_img)
            cv2.imwrite(
                osp.join(opt['save_folder'],
                         f'{img_name}_s{index:03d}{extension}'), cropped_img,
                [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']])
    process_info = f'Processing {img_name} ...'
    return process_info



if __name__=='__main__':
    parser = configargparse.ArgumentParser()
    parser.add_argument('--datadir', '-d', type=str)
    parser.add_argument('--repeats', '-r', type=int)
    args = parser.parse_args()
    
    outdir = './datasets/MVImgNet-blur'
    #os.makedirs(outdir, exist_ok=True)

    parse_dataset(args.datadir, outdir, args.repeats)
    main()