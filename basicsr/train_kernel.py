# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import argparse
import datetime
import logging
import math
import random
import time
import torch
import torchvision
from os import path as osp

from basicsr.data import create_dataloader, create_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import create_model
from basicsr.utils import (MessageLogger, check_resume, get_env_info,
                           get_root_logger, get_time_str, init_tb_logger,
                           init_wandb_logger, make_exp_dirs, mkdir_and_rename,
                           set_random_seed)
from basicsr.utils.dist_util import get_dist_info, init_dist
from basicsr.utils.options import dict2str, parse


def parse_options(is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('--input_path', type=str, required=False, help='The path to the input image. For single image inference only.')
    parser.add_argument('--output_path', type=str, required=False, help='The path to the output image. For single image inference only.')

    args = parser.parse_args()
    opt = parse(args.opt, is_train=is_train)

    # distributed settings
    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)
            print('init dist .. ', args.launcher)

    opt['rank'], opt['world_size'] = get_dist_info()

    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    if args.input_path is not None and args.output_path is not None:
        opt['img_path'] = {
            'input_img': args.input_path,
            'output_img': args.output_path
        }

    return opt


def init_loggers(opt):
    log_file = osp.join(opt['path']['log'],
                        f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # initialize wandb logger before tensorboard logger to allow proper sync:
    if (opt['logger'].get('wandb')
            is not None) and (opt['logger']['wandb'].get('project')
                              is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, (
            'should turn on tensorboard when using wandb')
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        # tb_logger = init_tb_logger(log_dir=f'./logs/{opt['name']}') #mkdir logs @CLY
        tb_logger = init_tb_logger(log_dir=osp.join('logs', opt['name']))
    return logger, tb_logger


def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loader = None, None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = create_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'],
                                            opt['rank'], dataset_enlarge_ratio)
            train_loader = create_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio /
                (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            logger.info(
                'Training statistics:'
                f'\n\tNumber of train images: {len(train_set)}'
                f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                f'\n\tWorld size (gpu number): {opt["world_size"]}'
                f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')

        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(
                val_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=None,
                seed=opt['manual_seed'])
            logger.info(
                f'Number of val images/folders in {dataset_opt["name"]}: '
                f'{len(val_set)}')
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loader, total_epochs, total_iters


def main():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=True)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # automatic resume ..
    state_folder_path = 'experiments/{}/training_states/'.format(opt['name'])
    import os
    try:
        states = os.listdir(state_folder_path)
    except:
        states = []

    resume_state = None
    if len(states) > 0:
        print('!!!!!! resume state .. ', states, state_folder_path)
        max_state_file = '{}.state'.format(max([int(x[0:-6]) for x in states]))
        resume_state = os.path.join(state_folder_path, max_state_file)
        opt['path']['resume_state'] = resume_state

    # load resume states if necessary
    if opt['path'].get('resume_state'):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt['path']['resume_state'],
            map_location=lambda storage, loc: storage.cuda(device_id))
    else:
        resume_state = None

    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt[
                'name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join('tb_logger', opt['name']))

    # initialize loggers
    logger, tb_logger = init_loggers(opt)

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loader, total_epochs, total_iters = result

    # create model
    """
    if resume_state:  # resume training
        check_resume(opt, resume_state['iter'])
        model = create_model(opt)
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, "
                    f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        model = create_model(opt)
        start_epoch = 0
        current_iter = 0
    """

    
    import importlib
    loss_module = importlib.import_module('basicsr.models.losses')
    pixel_type = opt['train']['pixel_opt'].pop('type')
    cri_pix_cls = getattr(loss_module, pixel_type)
    cri_pix = cri_pix_cls(**opt['train']['pixel_opt']).cuda()
    optim_type = opt['train']['optim_g'].pop('type')

    from basicsr.models.archs.ResUNet_arch import ResUNet
    from basicsr.models.archs.KernelNet_arch import BlurCLIP
    model = BlurCLIP().cuda()
    #model = ResUNet().cuda()

    # Optimizer.
    optim_params = []
    for k, v in model.named_parameters():
        if v.requires_grad:
            optim_params.append(v)
    
    optimizer_g = torch.optim.AdamW([{'params': optim_params}],
                                        **opt['train']['optim_g'])

    start_epoch = 0
    current_iter = 0
    
    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # dataloader prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f'Wrong prefetch_mode {prefetch_mode}.'
                         "Supported ones are: None, 'cuda', 'cpu'.")

    # training
    logger.info(
        f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_time, iter_time = time.time(), time.time()
    start_time = time.time()

    # for epoch in range(start_epoch, total_epochs + 1):
    epoch = start_epoch
    while current_iter <= total_iters:
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            current_iter += 1
            if current_iter > total_iters:
                break

            logit_img, logit_ker, loss = model(train_data['lq'].cuda(), train_data['kernel'].cuda())

            optimizer_g.zero_grad()
            loss.backward()
            optimizer_g.step()

            if current_iter % 10 ==0:
                print(f'Epoch: {epoch} Iter: {current_iter} loss: {loss:.6f} time: {time.time()-iter_time:.3f}')
            
            if current_iter % 100 ==0:
                print(logit_img)
            #kernel_pred = model(train_data['lq'].cuda())
            #import pdb; pdb.set_trace()
            
            
            """
            optimizer_g.zero_grad()
            kernel_map = train_data['kernel_map'].cuda()
            
            n_fg = torch.sum(kernel_map>0, dim=(-1, -2))
            n_bg = torch.sum(kernel_map==0, dim=(-1, -2))
            #import pdb; pdb.set_trace()
            weight = torch.where(kernel_map>0, torch.ones_like(kernel_map)/n_fg[..., None, None], torch.ones_like(kernel_map)/n_bg[..., None, None])

            l_pix = cri_pix(kernel_pred, kernel_map.bool().float(), weight)
            l_pix.backward()
            optimizer_g.step()

            if current_iter % 10 ==0:
                print(f'Epoch: {epoch} Iter: {current_iter} loss: {l_pix:.6f} time: {time.time()-iter_time:.3f}')
                #torchvision.utils.save_image(reblur, f'debug_reblur/{current_iter}_reblur.png' )
                #torchvision.utils.save_image(model.img_hq.detach(), f'debug_reblur/{current_iter}_restored.png' )

            if current_iter % 100 ==0:
                image_pred = train_data['lq'].clone()[0]
                image_gt = train_data['lq'].clone()[0]
                _, H, W = image_pred.shape
                kernel = kernel_pred[0].cpu()
                kernel_gt = kernel_map[0].cpu()

                #kernel = torch.where(kernel>1/120, torch.ones_like(kernel), torch.zeros_like(kernel))
                #kernel_gt = torch.where(kernel_gt>1/120, torch.ones_like(kernel), torch.zeros_like(kernel))
                
                stride = 16
                window_size = 64
                for ix, x_i in enumerate(range(window_size, W-1-window_size, stride)):
                    for iy, y_i in enumerate(range(window_size, H-1-window_size, stride)):
                        kernel_pred_i = kernel[y_i//4, x_i//4, :, :]
                        kernel_gt_i = kernel_gt[y_i//4, x_i//4, :, :]

                        kernel_pred_i = kernel_pred_i /  torch.max(kernel_pred_i)
                        kernel_gt_i = kernel_gt_i /  torch.max(kernel_gt_i)

                        image_pred[0, y_i-window_size//2:y_i+window_size//2, x_i-window_size//2:x_i+window_size//2] += kernel_pred_i
                        image_gt[1, y_i-window_size//2:y_i+window_size//2, x_i-window_size//2:x_i+window_size//2] += kernel_gt_i
                image_gt = torch.clip(image_gt, 0, 1.)
                image_pred = torch.clip(image_pred, 0, 1.)
                torchvision.utils.save_image(image_pred, f'debug_kernel/{current_iter}_pred.png' )
                torchvision.utils.save_image(image_gt, f'debug_kernel/{current_iter}_gt.png' )
            """
            """
            # update learning rate
            model.update_learning_rate(
                current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            
            # training
            model.feed_data(train_data, is_val=False)
            result_code = model.optimize_parameters(current_iter, tb_logger)
            # if result_code == -1 and tb_logger:
            #     print('loss explode .. ')
            #     exit(0)
            iter_time = time.time() - iter_time
            # log
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter, 'total_iter': total_iters}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_time, 'data_time': data_time})
                log_vars.update(model.get_current_log())
                # print('msg logger .. ', current_iter)
                msg_logger(log_vars)

            # save models and training states
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)

            # validation
            if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0 or current_iter == 1000):
            # if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):
                rgb2bgr = opt['val'].get('rgb2bgr', True)
                # wheather use uint8 image to compute metrics
                use_image = opt['val'].get('use_image', True)
                model.validation(val_loader, current_iter, tb_logger,
                                 opt['val']['save_img'], rgb2bgr, use_image )
                log_vars = {'epoch': epoch, 'iter': current_iter, 'total_iter': total_iters}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)


            data_time = time.time()
            iter_time = time.time()
            train_data = prefetcher.next()
            """

            """
            model = KernelNet(train_data['lq'].cuda(), train_data['kernel'].cuda())
            # Optimizer.
            optim_params = []
            for k, v in model.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
            
            optimizer_g = torch.optim.AdamW([{'params': optim_params}],
                                              **opt['train']['optim_g'])
            
            #model(train_data['gt'].cuda())
            while 1:
                current_iter += 1
                iter_time = time.time()
                reblur = model()

                optimizer_g.zero_grad()

                l_pix = cri_pix(reblur, train_data['lq'].cuda())
                l_pix.backward()
                optimizer_g.step()
                if current_iter % 10 ==0:
                    print(f'Iter: {current_iter} loss: {l_pix:.3f} time: {time.time()-iter_time:.3f}')
                    torchvision.utils.save_image(reblur, f'debug_reblur/{current_iter}_reblur.png' )
                    torchvision.utils.save_image(model.img_hq.detach(), f'debug_reblur/{current_iter}_restored.png' )
            """
            train_data = prefetcher.next()
        # end of iter
        epoch += 1
        
    # end of epoch

    consumed_time = str(
        datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    if opt.get('val') is not None:
        rgb2bgr = opt['val'].get('rgb2bgr', True)
        use_image = opt['val'].get('use_image', True)
        metric = model.validation(val_loader, current_iter, tb_logger,
                         opt['val']['save_img'], rgb2bgr, use_image)
        # if tb_logger:
        #     print('xxresult! ', opt['name'], ' ', metric)
    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    import os
    os.environ['GRPC_POLL_STRATEGY']='epoll1'
    main()
