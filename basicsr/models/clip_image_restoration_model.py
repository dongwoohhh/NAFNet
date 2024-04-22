# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import importlib
import torch
import torch.nn.functional as F
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.dist_util import get_dist_info

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')

class CLIPImageRestorationModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(CLIPImageRestorationModel, self).__init__(opt)

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
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            percep_type = train_opt['perceptual_opt'].pop('type')
            cri_perceptual_cls = getattr(loss_module, percep_type)
            self.cri_perceptual = cri_perceptual_cls(
                **train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')
        
        if train_opt.get('embedding_opt'):
            embedding_type = train_opt['embedding_opt'].pop('type')
            cri_embedding_cls = getattr(loss_module, embedding_type)
            self.cri_embedding = cri_embedding_cls(
                **train_opt['embedding_opt']).to(self.device)
        else:
            self.cri_embedding = None

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        self.hyper_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
        #         if k.startswith('module.offsets') or k.startswith('module.dcns'):
        #             optim_params_lowlr.append(v)
        #         else:
                if k.find('mlp_res_block')>0 or k.find('fc_hyper')>0:
                    self.hyper_params.append(v)
                else:
                    optim_params.append(v)
            # else:
            #     logger = get_root_logger()
            #     logger.warning(f'Params {k} will not be optimized.')
        # print(optim_params)
        # ratio = 0.1
    
        optim_type = train_opt['optim_g'].pop('type')
        self.lr_hyper = train_opt['optim_g'].pop('lr_hyper')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam([{'params': optim_params},
                                                 ], #{'params': hyper_params, 'lr':lr_hyper}
                                                **train_opt['optim_g'])
        elif optim_type == 'SGD':
            self.optimizer_g = torch.optim.SGD(optim_params,
                                               **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW([{'params': optim_params},
                                                  ], #{'params': hyper_params, 'lr':lr_hyper}
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

    def grids_overlap(self):
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

        num_row = 1 + (num_row -1)*2
        num_col = 1 + (num_col -1)*2

        
        """
        import math
        step_j = crop_size_w if num_col == 1 else math.ceil((w - crop_size_w) / (num_col - 1) - 1e-8)
        step_i = crop_size_h if num_row == 1 else math.ceil((h - crop_size_h) / (num_row - 1) - 1e-8)

        scale = self.scale
        step_i = step_i//scale*scale
        step_j = step_j//scale*scale
        """
        scale = self.scale
        stride_scale = int(self.opt['val'].get('stride_scale'))
        
        step_j = crop_size_h // stride_scale
        step_i = crop_size_w // stride_scale
        
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
            preds[0, :, i: i + crop_size_h, j: j + crop_size_w] += self.output[cnt]
            count_mt[0, 0, i: i + crop_size_h, j: j + crop_size_w] += 1.

        self.output = (preds / count_mt).to(self.device)
        self.lq = self.origin_lq

    def grids_inverse_overlap(self):
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
        
        stride_scale = int(self.opt['val'].get('stride_scale'))
        half_size_h = crop_size_h // stride_scale
        half_size_w = crop_size_w // stride_scale
        
        #import pdb; pdb.set_trace()
        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            #print(i, j, i, i + crop_size_h, j, j + crop_size_w)
        
            if i==0 and j==0:
                preds[0, :, 0:half_size_h+half_size_h//2, 0:half_size_w+half_size_w//2] = self.output[cnt, :, 0:half_size_h+half_size_h//2, 0:half_size_w+half_size_w//2]
            elif i+crop_size_h >= h and j+crop_size_w >= w:

                preds[0, :, i+half_size_h//2:, j+half_size_w//2:] = self.output[cnt, :,  half_size_h//2:, half_size_w//2:]
            elif i==0:
                if j+crop_size_w >= w:
                    preds[0, :, 0:half_size_h+half_size_h//2, j+half_size_w//2:] = self.output[cnt, :, 0:half_size_h+half_size_h//2, half_size_w//2:]    
                else: 
                    preds[0, :, 0:half_size_h+half_size_h//2, j+half_size_w//2:j+half_size_w//2+half_size_w] = self.output[cnt, :, 0:half_size_h+half_size_h//2, half_size_w//2:half_size_w//2+half_size_w]
            elif j==0:
                if i+crop_size_h >= h:
                    preds[0, :, i+half_size_h//2:, 0:half_size_w+half_size_w//2] = self.output[cnt, :, half_size_h//2:, 0:half_size_w//2+half_size_w] 
                else:
                    preds[0, :, i+half_size_h//2:i+half_size_h//2+half_size_h, 0:half_size_w+half_size_w//2] = self.output[cnt, :, half_size_h//2:half_size_h//2+half_size_h, 0:half_size_w+half_size_w//2]
            elif j+crop_size_w >= w:
                preds[0, :, i+half_size_h//2:i+half_size_h//2+half_size_h, j+half_size_w//2:] = self.output[cnt, :, half_size_h//2:half_size_h//2+half_size_h, half_size_w//2:]
            elif i+crop_size_h >= h:
                preds[0, :, i+half_size_h//2:, j+half_size_w//2:j+half_size_w//2+half_size_w] = self.output[cnt, :, half_size_h//2:, half_size_w//2:half_size_w//2+half_size_w] 
            else:
                preds[0, :, i+half_size_h//2:i+half_size_h//2+half_size_h, j+half_size_w//2:j+half_size_w//2+half_size_w] = self.output[cnt, :, half_size_h//2:half_size_h//2+half_size_h, half_size_w//2:half_size_w//2+half_size_w] 
                

            
            #preds[0, :, i: i + crop_size_h, j: j + crop_size_w] += self.output[cnt]
            #count_mt[0, 0, i: i + crop_size_h, j: j + crop_size_w] += 1.

        #self.output = (preds / count_mt).to(self.device)
        self.output = preds.to(self.device)
        self.lq = self.origin_lq
    def grids_inverse_overlap_avg(self):
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
        
        stride_scale = int(self.opt['val'].get('stride_scale'))
        half_size_h = crop_size_h // stride_scale
        half_size_w = crop_size_w // stride_scale
        
        #import pdb; pdb.set_trace()
        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            #print(i, j, i, i + crop_size_h, j, j + crop_size_w)
            """
            if i==0 and j==0:
                preds[0, :, 0:half_size_h+half_size_h//2, 0:half_size_w+half_size_w//2] = self.output[cnt, :, 0:half_size_h+half_size_h//2, 0:half_size_w+half_size_w//2]
            elif i+crop_size_h >= h and j+crop_size_w >= w:

                preds[0, :, i+half_size_h//2:, j+half_size_w//2:] = self.output[cnt, :,  half_size_h//2:, half_size_w//2:]
            elif i==0:
                if j+crop_size_w >= w:
                    preds[0, :, 0:half_size_h+half_size_h//2, j+half_size_w//2:] = self.output[cnt, :, 0:half_size_h+half_size_h//2, half_size_w//2:]    
                else: 
                    preds[0, :, 0:half_size_h+half_size_h//2, j+half_size_w//2:j+half_size_w//2+half_size_w] = self.output[cnt, :, 0:half_size_h+half_size_h//2, half_size_w//2:half_size_w//2+half_size_w]
            elif j==0:
                if i+crop_size_h >= h:
                    preds[0, :, i+half_size_h//2:, 0:half_size_w+half_size_w//2] = self.output[cnt, :, half_size_h//2:, 0:half_size_w//2+half_size_w] 
                else:
                    preds[0, :, i+half_size_h//2:i+half_size_h//2+half_size_h, 0:half_size_w+half_size_w//2] = self.output[cnt, :, half_size_h//2:half_size_h//2+half_size_h, 0:half_size_w+half_size_w//2]
            elif j+crop_size_w >= w:
                preds[0, :, i+half_size_h//2:i+half_size_h//2+half_size_h, j+half_size_w//2:] = self.output[cnt, :, half_size_h//2:half_size_h//2+half_size_h, half_size_w//2:]
            elif i+crop_size_h >= h:
                preds[0, :, i+half_size_h//2:, j+half_size_w//2:j+half_size_w//2+half_size_w] = self.output[cnt, :, half_size_h//2:, half_size_w//2:half_size_w//2+half_size_w] 
            else:
                preds[0, :, i+half_size_h//2:i+half_size_h//2+half_size_h, j+half_size_w//2:j+half_size_w//2+half_size_w] = self.output[cnt, :, half_size_h//2:half_size_h//2+half_size_h, half_size_w//2:half_size_w//2+half_size_w] 
            """ 

            
            preds[0, :, i: i + crop_size_h, j: j + crop_size_w] += self.output[cnt]
            count_mt[0, 0, i: i + crop_size_h, j: j + crop_size_w] += 1.

        self.output = (preds / count_mt).to(self.device)
        #self.output = preds.to(self.device)
        self.lq = self.origin_lq
    def optimize_parameters(self, current_iter, tb_logger):
        self.optimizer_g.zero_grad()

        if self.opt['train'].get('mixup', False):
            self.mixup_aug()

        preds = self.net_g(self.lq)
        if not isinstance(preds, list):
            preds = [preds]

        self.output = preds[-1]

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = 0.
            for pred in preds:
                l_pix += self.cri_pix(pred, self.gt)

            # print('l pix ... ', l_pix)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
        #
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style
        if self.cri_embedding:
            embedding_input = self.net_g.module.b_encoder(self.lq)
            embedding_gt = self.net_g.module.b_encoder(self.gt)
            embedding_deblur = self.net_g.module.b_encoder(self.output)

            l_embedding_input = self.cri_embedding(embedding_input, embedding_gt)
            l_embedding = self.cri_embedding(embedding_deblur, embedding_gt)
            
            l_total += l_embedding
            loss_dict['l_embedding'] = l_embedding
            loss_dict['l_embedding_lq'] = l_embedding_input


        l_total = l_total + 0. * sum(p.sum() for p in self.net_g.parameters())

        l_total.backward()
        use_grad_clip = self.opt['train'].get('use_grad_clip', True)
        if use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()


        self.log_dict = self.reduce_loss_dict(loss_dict)

        if current_iter >= self.opt['train'].get('start_hyper_iter') and len(self.optimizer_g.param_groups)==1:
            self.optimizer_g.add_param_group({'params': self.hyper_params, 'lr':self.lr_hyper})

        #    self.net_g.module.fc_hyper5.weight.requires_grad = True
        #    self.net_g.module.fc_hyper5.bias.requires_grad = True


    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            n = len(self.lq)
            outs = []
            m = self.opt['val'].get('max_minibatch', n)
            i = 0
            while i < n:
                j = i + m
                if j >= n:
                    j = n
                pred = self.net_g(self.lq[i:j])
                if isinstance(pred, list):
                    pred = pred[-1]
                outs.append(pred.detach().cpu())
                i = j

            self.output = torch.cat(outs, dim=0)
        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }

        rank, world_size = get_dist_info()
        if rank == 0:
            pbar = tqdm(total=len(dataloader), unit='image')

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            if idx % world_size != rank:
                continue

            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data, is_val=True)
            if self.opt['val'].get('grids', False):
                #self.grids_overlap()
                self.grids()

            self.test()

            if self.opt['val'].get('grids', False):
                #self.grids_inverse_overlap_avg()
                self.grids_inverse()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if sr_img.shape[2] == 6:
                    L_img = sr_img[:, :, :3]
                    R_img = sr_img[:, :, 3:]

                    # visual_dir = osp.join('visual_results', dataset_name, self.opt['name'])
                    visual_dir = osp.join(self.opt['path']['visualization'], dataset_name)

                    imwrite(L_img, osp.join(visual_dir, f'{img_name}_L.png'))
                    imwrite(R_img, osp.join(visual_dir, f'{img_name}_R.png'))
                else:
                    if self.opt['is_train']:

                        save_img_path = osp.join(self.opt['path']['visualization'],
                                                 img_name,
                                                 f'{img_name}_{current_iter}.png')

                        save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                                 img_name,
                                                 f'{img_name}_{current_iter}_gt.png')
                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}.png')
                        save_gt_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_gt.png')

                    imwrite(sr_img, save_img_path)
                    imwrite(gt_img, save_gt_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            cnt += 1
            if rank == 0:
                for _ in range(world_size):
                    pbar.update(1)
                    pbar.set_description(f'Test {img_name}')
        if rank == 0:
            pbar.close()

        # current_metric = 0.
        collected_metrics = OrderedDict()
        if with_metrics:
            for metric in self.metric_results.keys():
                collected_metrics[metric] = torch.tensor(self.metric_results[metric]).float().to(self.device)
            collected_metrics['cnt'] = torch.tensor(cnt).float().to(self.device)

            self.collected_metrics = collected_metrics
        
        keys = []
        metrics = []
        for name, value in self.collected_metrics.items():
            keys.append(name)
            metrics.append(value)
        metrics = torch.stack(metrics, 0)
        torch.distributed.reduce(metrics, dst=0)
        if self.opt['rank'] == 0:
            metrics_dict = {}
            cnt = 0
            for key, metric in zip(keys, metrics):
                if key == 'cnt':
                    cnt = float(metric)
                    continue
                metrics_dict[key] = float(metric)

            for key in metrics_dict:
                metrics_dict[key] /= cnt

            self._log_validation_metric_values(current_iter, dataloader.dataset.opt['name'],
                                               tb_logger, metrics_dict)
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
