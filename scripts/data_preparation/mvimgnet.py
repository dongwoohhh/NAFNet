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

def parse_dataset(datadir, outdir, repeats, n_threads=20):
    #split_list = ['mvi_selected0', 'mvi_selected1', 'mvi_selected2', 'mvi_selected3']

    outdir_blur = os.path.join(outdir, 'input')
    outdir_target = os.path.join(outdir, 'target')

    os.makedirs(outdir_blur, exist_ok=True)
    os.makedirs(outdir_target, exist_ok=True)

    all_scenes_list = [f for f in glob.glob(os.path.join(datadir, '*'))]

    pbar = tqdm(total=len(all_scenes_list), unit='scene', desc='Extract')
    pool = Pool(n_threads)
    
    for scene in all_scenes_list:
        #split_dir = os.path.join(datadir, split, '*')
        #scene_list = [f for f in glob.glob(split_dir)]
        #pbar = tqdm(len(scene_list), unit='scene', dsec='Copy')
        pool.apply_async(
            worker_cp, args=(scene, outdir_blur, outdir_target, repeats), callback=lambda arg: pbar.update(1)
        )    
    pool.close()
    pool.join()
    pbar.close()
            

def worker_cp(scene, outdir_blur, outdir_target, repeats):
    
    #for scene in scene_list:
    #    print(scene)
    #render_path = os.path.join(scene, 'train', 'ours_30000')

    #if not os.path.exists(render_path):
    #    continue
        
    scene_name = scene.split('/')[-1]
    blurred_path = os.path.join(scene, 'blurred')
    gt_path = os.path.join(scene, 'sharp')

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
        

if __name__=='__main__':
    parser = configargparse.ArgumentParser()
    parser.add_argument('--datadir', '-d', type=str)
    parser.add_argument('--repeats', '-r', type=int)
    args = parser.parse_args()
    
    outdir = './datasets/MVImgNet-blur'
    #os.makedirs(outdir, exist_ok=True)

    parse_dataset(args.datadir, outdir, args.repeats)
    #create_lmdb_for_mvimgnet()
    #main()