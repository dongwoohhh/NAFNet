# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import importlib
import torch
import torchvision
import torch.nn.functional as F
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
import os
from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.dist_util import get_dist_info
from einops.layers.torch import Rearrange

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')

class CLIPModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(CLIPModel, self).__init__(opt)

        # define network
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

        self.scale = int(opt['scale'])

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        # define losses
        #self.cri_embed = F.cross_entropy

        # define loss
        self.cri_embed = loss_module.JSDivergence()
        #

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
        #         if k.startswith('module.offsets') or k.startswith('module.dcns'):
        #             optim_params_lowlr.append(v)
        #         else:
                optim_params.append(v)
            # else:
            #     logger = get_root_logger()
            #     logger.warning(f'Params {k} will not be optimized.')
        # print(optim_params)
        # ratio = 0.1

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam([{'params': optim_params}],
                                                **train_opt['optim_g'])
        elif optim_type == 'SGD':
            self.optimizer_g = torch.optim.SGD(optim_params,
                                               **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW([{'params': optim_params}],
                                                **train_opt['optim_g'])
            pass
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data, is_val=False):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        self.kernel = data['kernel'].to(self.device)

    def grids(self):
        b, c, h, w = self.gt.size()
        self.original_size = (b, c, h, w)

        assert b == 1
        if 'crop_size_h' in self.opt['val']:
            crop_size_h = self.opt['val']['crop_size_h']
        else:
            crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)

        if 'crop_size_w' in self.opt['val']:
            crop_size_w = self.opt['val'].get('crop_size_w')
        else:
            crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)


        crop_size_h, crop_size_w = crop_size_h // self.scale * self.scale, crop_size_w // self.scale * self.scale
        #adaptive step_i, step_j
        num_row = (h - 1) // crop_size_h + 1
        num_col = (w - 1) // crop_size_w + 1

        import math
        step_j = crop_size_w if num_col == 1 else math.ceil((w - crop_size_w) / (num_col - 1) - 1e-8)
        step_i = crop_size_h if num_row == 1 else math.ceil((h - crop_size_h) / (num_row - 1) - 1e-8)

        scale = self.scale
        step_i = step_i//scale*scale
        step_j = step_j//scale*scale

        parts = []
        idxes = []

        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + crop_size_h >= h:
                i = h - crop_size_h
                last_i = True

            last_j = False
            while j < w and not last_j:
                if j + crop_size_w >= w:
                    j = w - crop_size_w
                    last_j = True
                parts.append(self.lq[:, :, i // scale :(i + crop_size_h) // scale, j // scale:(j + crop_size_w) // scale])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        self.origin_lq = self.lq
        self.lq = torch.cat(parts, dim=0)
        self.idxes = idxes

    def grids_inverse(self):
        preds = torch.zeros(self.original_size)
        b, c, h, w = self.original_size

        count_mt = torch.zeros((b, 1, h, w))
        if 'crop_size_h' in self.opt['val']:
            crop_size_h = self.opt['val']['crop_size_h']
        else:
            crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)

        if 'crop_size_w' in self.opt['val']:
            crop_size_w = self.opt['val'].get('crop_size_w')
        else:
            crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)

        crop_size_h, crop_size_w = crop_size_h // self.scale * self.scale, crop_size_w // self.scale * self.scale

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            preds[0, :, i: i + crop_size_h, j: j + crop_size_w] += self.outs[cnt]
            count_mt[0, 0, i: i + crop_size_h, j: j + crop_size_w] += 1.

        self.output = (preds / count_mt).to(self.device)
        self.lq = self.origin_lq

    def compute_pdist(self, kernel):
        #kernel_sparse = kernel[:, ::4, ::4]
        B, kH, kW, nK, _ = kernel.shape
        
        length_kernel = self.compute_length_kernel(kernel)
        #scale_kernel = 1.0 #16.0

        kernel_backward = kernel.flip(dims=(-2,))
        
        kernel_flatten = kernel.reshape(B, -1, 2)
        kernel_flatten_backward = kernel_backward.reshape(B, -1, 2)
        
        diff_forward = kernel_flatten.unsqueeze(1) - kernel_flatten.unsqueeze(0)
        diff_backward = kernel_flatten.unsqueeze(1) - kernel_flatten_backward.unsqueeze(0)
        #import pdb; pdb.set_trace()
        
        dist_forward = torch.sqrt(torch.sum(diff_forward**2, dim=-1)).reshape(B, B, kH, kW, nK)
        dist_backward = torch.sqrt(torch.sum(diff_backward**2, dim=-1)).reshape(B, B, kH, kW, nK)
        
        
        dist_forward = dist_forward / (length_kernel[:, None, :, :, None]+1.0)
        dist_backward = dist_backward / (length_kernel[:, None, :, :, None]+1.0)

        #dist_forward = dist_forward / scale_kernel
        #dist_backward = dist_backward / scale_kernel

        dist_forward = torch.mean(dist_forward, dim=-1)#.mean(-1).mean(-1)
        dist_backward = torch.mean(dist_backward, dim=-1)#.mean(-1).mean(-1)

        dist = torch.minimum(dist_forward, dist_backward)
        
        dist = (dist + dist.transpose(0, 1))/2.

        return dist

    def compute_length_kernel(self, kernel):
        delta_xy = kernel[..., :-1, :] - kernel[..., 1:, :]
        dist_xy = torch.norm(delta_xy, dim=-1)
        length_kernel = torch.sum(dist_xy, dim=-1)
        #import pdb;pdb.set_trace()

        return length_kernel
        

    def vis_kernel_for_debug(self, image, kernel, gt_size, scale_kernel):

        n_kernel= kernel.shape[1]
        N, _, H, W = image.shape
        kernel = kernel*gt_size
        
        stride = 32
        window_size = 100

        #output = torch.zeros((window_size*(H//stride + 1), window_size*(W//stride + 1)))
        
        images_out = []
        for i_b in range(N):
            image_chw = image[i_b].clone()#.permute(1,2,0)
            kernel_i = kernel[i_b]
            for ix, x_i in enumerate(range(stride//2, W-1, stride)):
                for iy, y_i in enumerate(range(stride//2, H-1, stride)):
                    """
                    window_i = torch.zeros(100, 100).long()
                    window_i[window_size//2, window_size//2] = 255
                    window_i[:, window_size-1] = 255
                    window_i[:, 0] = 255
                    window_i[0, :] = 255
                    window_i[window_size-1, :] = 255
                    
                    kx = torch.clip(kernel[:, 0, y_i//stride, x_i//stride]+window_size//2, 0, window_size-1).long()
                    ky = torch.clip(kernel[:, 1, y_i//stride, x_i//stride]+window_size//2, 0, window_size-1).long()
                    
                    window_i[kx, ky] = 255
                    output[window_size*iy:window_size*(iy+1), window_size*ix:window_size*(ix+1)] = window_i
                    """
                    
                    
                    kx_image = torch.clip(x_i+kernel_i[y_i//scale_kernel, x_i//scale_kernel, :, 0], 0, W-1).long()
                    ky_image = torch.clip(y_i+kernel_i[ y_i//scale_kernel, x_i//scale_kernel, :, 1,], 0, H-1).long()
                    
                    image_chw[0, ky_image, kx_image] = 1.
                    image_chw[1, ky_image, kx_image] = 0.
                    image_chw[2, ky_image, kx_image] = 0.

            images_out.append(image_chw)
        images_out = torch.stack(images_out)

        return images_out


    def optimize_parameters(self, current_iter, tb_logger):
        self.optimizer_g.zero_grad()
        
        logits_per_image, logits_per_kernel, logits_per_image_internal, logits_per_kernel_internal = self.net_g(self.lq, self.kernel)


        #if not isinstance(logits_per_kernel, list):
        #    logits_per_kernel = [logits_per_kernel]

        self.output = logits_per_kernel
        epsilon = self.opt['train']['epsilon']
        kernel_pixel = self.opt['datasets']['train']['gt_size']*self.kernel
        pdist_kernel = self.compute_pdist(kernel_pixel)
        
        #torch.set_printoptions(precision=3)
        #prob_kernel = torch.softmax(1/(scale_softmax*pdist_kernel+epsilon), dim=1)
        
        prob_kernel = torch.softmax(-epsilon*pdist_kernel, dim=1)
        prob_kernel = Rearrange('b1 b2 k1 k2 -> k1 k2 b1 b2')(prob_kernel)
        #prob_image = torch.softmax(-epsilon*pdist_kernel.)
        #import pdb; pdb.set_trace()
        #print(logits_per_kernel)
        l_total = 0
        loss_dict = OrderedDict()
        prob_kernel = Rearrange('k1 k2 b1 b2 -> (k1 k2 b1) b2')(prob_kernel)
        logits_per_kernel = Rearrange('k1 k2 b1 b2 -> (k1 k2 b1) b2')(logits_per_kernel)
        logits_per_image = Rearrange('k1 k2 b1 b2 -> (k1 k2 b1) b2')(logits_per_image)

        # cross entropy loss
        kernel_loss = self.cri_embed(logits_per_kernel, prob_kernel)
        image_loss = self.cri_embed(logits_per_image, prob_kernel)

        l_ce = (kernel_loss + image_loss) / 2.0
        
        l_total += l_ce
        loss_dict['l_jsd'] = l_ce
        # internal
        k_downscale = self.net_g.module.downscale
        kernel_pixel_internal = Rearrange('b h w nk d -> (h w) b 1 nk d')(kernel_pixel[:, ::k_downscale, ::k_downscale])# k_downscale//2::k_downscale, k_downscale//2::k_downscale]) #
        pdist_internal = self.compute_pdist(kernel_pixel_internal).squeeze(-1)
        prob_internal = torch.softmax(-epsilon*pdist_internal, dim=1)
        
        #import pdb; pdb.set_trace()
        prob_debug = prob_internal[..., 0]
        logit_debug = logits_per_kernel_internal[0]
        prob_internal = Rearrange('k1 k2 b -> (b k1) k2')(prob_internal)

        logits_per_kernel_internal = Rearrange('b k1 k2 -> (b k1) k2')(logits_per_kernel_internal)
        logits_per_image_internal = Rearrange('b k1 k2 -> (b k1) k2')(logits_per_image_internal)
        
        # cross entropy loss
        #kernel_loss_internal = F.cross_entropy(logits_per_kernel_internal, prob_internal)
        #image_loss_internal = F.cross_entropy(logits_per_image_internal, prob_internal)
        kernel_loss_internal = self.cri_embed(logits_per_kernel_internal, prob_internal, weighted=True)
        image_loss_internal = self.cri_embed(logits_per_image_internal, prob_internal, weighted=True)
        """
        save_fig_dir = osp.join(f'debug_confusion_gt.png')
        df_cm = pd.DataFrame(prob_debug.detach().cpu().numpy(), index = [i for i in range(9)], columns = [i for i in range(9)])
        confusion = sn.heatmap(df_cm, annot=True, annot_kws={"size": 4})
        figure = confusion.get_figure()
        figure.savefig(save_fig_dir, dpi=400)

        figure.clear()

        prob_pred_debug = torch.softmax(logit_debug, dim=1)
        save_fig_dir = osp.join(f'debug_confusion_pred.png')
        df_cm = pd.DataFrame(prob_pred_debug.detach().cpu().numpy(), index = [i for i in range(9)], columns = [i for i in range(9)])
        confusion = sn.heatmap(df_cm, annot=True, annot_kws={"size": 4})
        figure = confusion.get_figure()
        figure.savefig(save_fig_dir, dpi=400)

        figure.clear()

        torchvision.utils.save_image(self.lq[0],'debug_image.png')
        import pdb; pdb.set_trace()
        """
        
        l_ce_internal = (kernel_loss_internal + image_loss_internal) / 2.0

        l_total += l_ce_internal
        loss_dict['l_jsd_internal'] = l_ce_internal
        #loss_dict['l_weight'] = 0.001 * sum(p.sum() for p in self.net_g.parameters())
        l_total = l_total + 0 * sum(p.sum() for p in self.net_g.parameters())

        l_total.backward()

        use_grad_clip = self.opt['train'].get('use_grad_clip', True)
        if use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()

        loss_dict['m_ls'] = self.net_g.module.logit_scale
        loss_dict['m_ls_i'] = self.net_g.module.logit_scale_internal
        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            
            logits_per_image, logits_per_kernel, logits_per_image_internal, logits_per_kernel_internal = self.net_g(self.lq, self.kernel)
            self.output = logits_per_kernel.detach().cpu()
            self.output_internal = logits_per_kernel_internal.detach().cpu()
        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        gt_size = dataloader.dataset.opt['gt_size']
        scale_kernel = dataloader.dataset.opt['scale_kernel']
        n_images = dataloader.dataset.opt['batch_size_per_gpu']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                'l_jsd': 0
            }
        
        rank, world_size = get_dist_info()
        if rank == 0:
            pbar = tqdm(total=len(dataloader), unit='image')
            #pbar = tqdm(total=1000, unit='image')

        cnt = 0
        epsilon = self.opt['train']['epsilon']
        for idx, val_data in enumerate(dataloader):
            if idx % world_size != rank:
                continue
            if len(val_data['lq']) < n_images:
                continue
            self.feed_data(val_data, is_val=True)
            #if self.opt['val'].get('grids', False):
            #    self.grids()

            self.test()
            kernel_pixel = gt_size*self.kernel
            pdist_kernel = self.compute_pdist(kernel_pixel)
            #torch.set_printoptions(precision=3)
            prob_kernel = torch.softmax(-epsilon*pdist_kernel, dim=1).cpu()

            n_kernels = prob_kernel.shape[-1]
            prob_center = prob_kernel[:, :, n_kernels//2, n_kernels//2]

            prob_kernel = Rearrange('b1 b2 k1 k2 -> k1 k2 b1 b2')(prob_kernel)
            
            prob_kernel = Rearrange('k1 k2 b1 b2 -> (k1 k2 b1) b2')(prob_kernel)
            logits_per_kernel = Rearrange('k1 k2 b1 b2 -> (k1 k2 b1) b2')(self.output)
            logits_per_image = Rearrange('k1 k2 b1 b2 -> (k1 k2 b1) b2')(self.output.transpose(-1,-2)) 
    
            #kernel_loss = F.cross_entropy(logits_per_kernel, prob_kernel)
            #image_loss = F.cross_entropy(logits_per_image, prob_kernel)
            kernel_loss = self.cri_embed(logits_per_kernel, prob_kernel)
            image_loss = self.cri_embed(logits_per_image, prob_kernel)

            l_jsd = (kernel_loss + image_loss) / 2.0

            self.metric_results['l_jsd'] += l_jsd
            
            # visualize

            savedir = osp.join(self.opt['path']['visualization'], dataset_name, str(current_iter))#, f'{str(idx)}_{str(rank)}')
            os.makedirs(savedir, exist_ok=True)
            
            lq_cat = []
            save_img_dir = osp.join(savedir, f'{str(idx)}_images.png')
            lq_kernel = self.vis_kernel_for_debug(self.lq, self.kernel, gt_size, scale_kernel)
            lq_kernel_clone = lq_kernel.clone()
            #import pdb; pdb.set_trace()
            lq_cat = lq_kernel.permute(1,2,0,3).reshape(3, gt_size, -1)
            lq_upper = lq_cat[...,:n_images//2*gt_size]
            lq_lower = lq_cat[...,n_images//2*gt_size:]
            lq2line = torch.cat([lq_upper, torch.zeros_like(lq_upper), lq_lower], dim=1)
            torchvision.utils.save_image(lq2line, save_img_dir)
                
            
            save_fig_dir = osp.join(savedir, f'{str(idx)}_confusion_pred.png')
            prob_kernel_pred = F.softmax(self.output[n_kernels//2, n_kernels//2], dim=1).detach().cpu().numpy()
            df_cm = pd.DataFrame(prob_kernel_pred, index = [i for i in range(n_images)], columns = [i for i in range(n_images)])
            confusion = sn.heatmap(df_cm, annot=True, annot_kws={"size": 4})
            figure = confusion.get_figure()
            figure.savefig(save_fig_dir, dpi=400)

            figure.clear()

            save_fig_dir = osp.join(savedir, f'{str(idx)}_confusion_gt.png')
            df_cm = pd.DataFrame(prob_center.detach().cpu().numpy(), index = [i for i in range(n_images)], columns = [i for i in range(n_images)])
            confusion = sn.heatmap(df_cm, annot=True, annot_kws={"size": 4})
            figure = confusion.get_figure()
            figure.savefig(save_fig_dir, dpi=400)

            figure.clear()
            """
            positions = torch.tensor([[1,1], [1,3], [1, 6], [3,1], [3,3], [3,6], [6,1], [6,3], [6,6]])
            boxes = torch.tensor([[32,32,64,64], [96,32,128,64], [192,32,224,64],
                                  [32,96,64,128], [96,96,128,128],[192,96,224,128],
                                  [32,192,64,224],[96,192,128,224], [192,192,224,224]])
            """
            """
            positions = torch.tensor([[2,2], [2,8], [2, 13], [8, 2], [8,8], [8,13], [13,2], [13,8], [13,13]])
            boxes = torch.tensor([[64,64,96,96], [256, 64,288, 96], [416,64,448,96],
                                  [64,256, 96,288], [256,256,288,288], [416, 256, 448, 288],
                                  [64,416,96,448], [256, 416, 288, 448], [416,416,448,448]])
            """
            positions = torch.tensor([[2,2], [2,8], [2, 13], [8, 2], [8,8], [8,13], [13,2], [13,8], [13,13]])
            boxes = torch.tensor([[32,32,64,64], [128,32,160,64], [192,32,224,64],
                                  [32,128,64,160], [128,128,160,160],[192,128,224,160],
                                  [32,192,64,224],[128,192,160,224], [192,192,224,224]])
            n_positions = len(positions)

            kernel_pixel_corner = kernel_pixel.cpu()[:, positions[:,0],positions[:,1]]
            # gt
            kernel_pixel_corner = Rearrange('b np nk d -> np b 1 nk d')(kernel_pixel_corner)
            pdist_corner = self.compute_pdist(kernel_pixel_corner)
            pdist_corner = Rearrange('n1 n2 b 1 -> b n1 n2')(pdist_corner)
            prob_kernel = torch.softmax(-epsilon*pdist_corner, dim=1).numpy()
            # pred
            #logits_per_kernel = Rearrange('k1 k2 b1 b2 -> (k1 k2 b1) b2')(self.output)
            #embed_i = self.output_embed_i
            
            embed_corner = self.net_g.module._embed_i.detach().cpu()[:, :, positions[:,0], positions[:, 1]]
            logit_scale = self.net_g.module.logit_scale.detach().cpu().item()
            logit_embed = logit_scale * torch.matmul(embed_corner.transpose(-1,-2), embed_corner)
            #import pdb; pdb.set_trace()
            #output_internal = 
            prob_embed = F.softmax(logit_embed, dim=1).numpy()
            
            for i_img, prob_i in enumerate(prob_kernel):
                save_img_dir = osp.join(savedir, f'Debug_{str(idx)}_{str(i_img)}_images.png')
                img_i = lq_kernel_clone[i_img]
                img_i = torchvision.utils.draw_bounding_boxes((255*img_i).byte(),boxes, width=3)
                torchvision.utils.save_image((img_i/255.).float(),save_img_dir)
                
                save_fig_dir = osp.join(savedir, f'Debug_{str(idx)}_{str(i_img)}_confusion_gt.png')
                
                df_cm = pd.DataFrame(prob_i, index = [i for i in range(n_positions)], columns = [i for i in range(n_positions)])
                confusion = sn.heatmap(df_cm, annot=True, annot_kws={"size": 4})
                figure = confusion.get_figure()
                figure.savefig(save_fig_dir, dpi=400)

                figure.clear()
                
                save_fig_dir = osp.join(savedir, f'Debug_{str(idx)}_{str(i_img)}_confusion_pred.png')
                prob_pred = prob_embed[i_img]
                df_cm = pd.DataFrame(prob_pred, index = [i for i in range(n_positions)], columns = [i for i in range(n_positions)])
                confusion = sn.heatmap(df_cm, annot=True, annot_kws={"size": 4})
                figure = confusion.get_figure()
                figure.savefig(save_fig_dir, dpi=400)

                figure.clear()

            cnt += 1
            
            if rank == 0:
                for _ in range(world_size):
                    pbar.update(1)
                    pbar.set_description(f'Test {idx}')
        
        self.metric_results['l_jsd'] /= cnt
        self._log_validation_metric_values(current_iter, dataloader.dataset.opt['name'],
                                                   tb_logger, self.metric_results)

        if rank == 0:
            pbar.close()
        
        return 0.

    def nondist_validation(self, *args, **kwargs):
        logger = get_root_logger()
        logger.warning('nondist_validation is not implemented. Run dist_validation.')
        self.dist_validation(*args, **kwargs)


    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger, metric_dict):
        log_str = f'Validation {dataset_name}, \t'
        for metric, value in metric_dict.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)

        log_dict = OrderedDict()
        # for name, value in loss_dict.items():
        for metric, value in metric_dict.items():
            log_dict[f'm_{metric}'] = value

        self.log_dict = log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
