# -*- encoding: utf-8 -*-

import os
import time
import torch
import logging
import datetime
import argparse
import os.path as osp
import torch.nn as nn
from model import BiSeNet
from loss import OhemCELoss
from evaluate import evaluate
from logger import setup_logger
from optimizer import Optimizer
import torch.distributed as dist
from hair_dataset import HairMask
from torch.utils.data import DataLoader

respth = './res/cp'
if not osp.exists(respth):
    os.makedirs(respth)
logger = logging.getLogger()


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument(
        '--local_rank',
        dest='local_rank',
        type=int,
        default=-1,
    )
    return parse.parse_args()


def train():
    args = parse_args()
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:33241',
        world_size=torch.cuda.device_count(),
        rank=args.local_rank
    )
    setup_logger(respth)

    # dataset
    n_classes = 2
    n_img_per_gpu = 16
    n_workers = 8
    crop_size = [448, 448]
    data_root = './data/HAIR'

    dataset = HairMask(data_root, crop_size=crop_size, mode='train')
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    data_loader = DataLoader(dataset,
                             batch_size=n_img_per_gpu,
                             shuffle=False,
                             sampler=sampler,
                             num_workers=n_workers,
                             pin_memory=True,
                             drop_last=True)

    # model
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    net.train()
    net = nn.parallel.DistributedDataParallel(net,
                                              device_ids=[args.local_rank, ],
                                              output_device=args.local_rank
                                              )

    score_thres = 0.7
    n_min = n_img_per_gpu * crop_size[0] * crop_size[1] // 16
    LossP = OhemCELoss(thresh=score_thres, n_min=n_min)
    Loss2 = OhemCELoss(thresh=score_thres, n_min=n_min)
    Loss3 = OhemCELoss(thresh=score_thres, n_min=n_min)

    ## optimizer
    momentum = 0.9
    weight_decay = 5e-4
    lr_start = 1e-2
    max_iter = 80000
    power = 0.9
    warmup_steps = 1000
    warmup_start_lr = 1e-5

    optim = Optimizer(
        model=net.module,
        lr0=lr_start,
        momentum=momentum,
        wd=weight_decay,
        warmup_steps=warmup_steps,
        warmup_start_lr=warmup_start_lr,
        max_iter=max_iter,
        power=power)

    ## train loop
    msg_iter = 50
    loss_avg = []
    st = glob_st = time.time()
    data_iter = iter(data_loader)
    epoch = 0

    for it in range(max_iter):
        try:
            im, label = next(data_iter)
            if not im.size()[0] == n_img_per_gpu:
                raise StopIteration
        except StopIteration:
            epoch += 1
            sampler.set_epoch(epoch)
            data_iter = iter(data_loader)
            im, label = next(data_iter)

        im = im.cuda()
        label = label.cuda()
        label = torch.squeeze(label, 1)

        optim.zero_grad()
        out, out16, out32 = net(im)
        lossp = LossP(out, label)
        loss2 = Loss2(out16, label)
        loss3 = Loss3(out32, label)
        loss = lossp + loss2 + loss3

        loss.backward()
        optim.step()

        loss_avg.append(loss.item())

        if (it + 1) % msg_iter == 0:
            loss_avg = sum(loss_avg) / len(loss_avg)
            lr = optim.lr
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            eta = int((max_iter - it) * (glob_t_intv / it))
            eta = str(datetime.timedelta(seconds=eta))
            msg = ', '.join([
                'it: {it}/{max_it}',
                'lr: {lr:4f}',
                'loss: {loss:.4f}',
                'eta: {eta}',
                'time: {time:.4f}',
            ]).format(
                it=it + 1,
                max_it=max_iter,
                lr=lr,
                loss=loss_avg,
                time=t_intv,
                eta=eta
            )
            logger.info(msg)
            loss_avg = []
            st = ed

        if dist.get_rank() == 0:
            if (it + 1) % 5000 == 0:
                state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
                torch.save(state, './res/cp/{}_iter.pth'.format(it))
                evaluate(dspth='./data/HAIR/img_dir/val', cp='{}_iter.pth'.format(it),)

    #  dump the final model
    save_pth = osp.join(respth, 'model_final_diss.pth')
    # net.cpu()
    state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
    if dist.get_rank() == 0:
        torch.save(state, save_pth)
    logger.info('training done, model saved to: {}'.format(save_pth))


if __name__ == "__main__":
    train()
