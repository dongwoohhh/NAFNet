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
from basicsr.utils.create_lmdb import create_lmdb_for_bsd_test

import configargparse

def parse_dataset(datadir, outdir, n_threads=20):#, repeats, ):
    #split_list = ['mvi_selected0', 'mvi_selected1', 'mvi_selected2', 'mvi_selected3']

    outdir_blur = os.path.join(outdir, 'test', 'input')
    outdir_target = os.path.join(outdir, 'test', 'target')

    os.makedirs(outdir_blur, exist_ok=True)
    os.makedirs(outdir_target, exist_ok=True)

    
    split_list = ['BSD_1ms8ms', 'BSD_2ms16ms', 'BSD_3ms24ms']
    
    for split in split_list:
        inputdir = os.path.join(datadir, split, 'test')

        scene_list = [f for f in glob.glob(os.path.join(inputdir, '*'))]

        for scene_dir in scene_list:
            scene_name = scene_dir.split('/')[-1]
            blur_dir = os.path.join(scene_dir, 'Blur', 'RGB')
            sharp_dir = os.path.join(scene_dir, 'Sharp', 'RGB')

            image_list = [f for f in glob.glob(os.path.join(blur_dir, '*'))]
            for blur_image_dir in image_list:
                image_name = blur_image_dir.split('/')[-1]
                sharp_image_dir = os.path.join(sharp_dir, image_name)
            
                blur_out = os.path.join(outdir_blur, f'{split}_{scene_name}_{image_name}')
                sharp_out = os.path.join(outdir_target, f'{split}_{scene_name}_{image_name}')
                
                os.system(f'mv {blur_image_dir} {blur_out}')
                os.system(f'mv {sharp_image_dir} {sharp_out}')
    



if __name__=='__main__':
    parser = configargparse.ArgumentParser()
    parser.add_argument('--datadir', '-d', type=str)
    args = parser.parse_args()
    
    outdir = args.datadir #'./datasets/BSD'
    #os.makedirs(outdir, exist_ok=True)

    parse_dataset(args.datadir, outdir)
    create_lmdb_for_bsd_test()
    #main()