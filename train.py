from __future__ import absolute_import, division, print_function

import os
import signal
from os.path import join as pjoin

from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_

from torchvision.transforms import Compose, Lambda

from torch.utils.data import ConcatDataset, DataLoader
from dataset import fold_split, ADNIDataset
from transforms import transform_presets

from models import Plane, resnet11, resnet19, resnet35, resnet51
from utils import Summary, SimpleTimer, ScoreReport
from utils import save_checkpoint, print_model_parameters
from config import train_args, argument_report


def main():
    # option flags
    FLG = train_args()

    # torch setting
    device = torch.device('cuda:{}'.format(FLG.devices[0]))
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(FLG.devices[0])

    # create summary and report the option
    visenv = FLG.model
    summary = Summary(port=10001, env=visenv)
    summary.viz.text(argument_report(FLG, end='<br>'),
                     win='report'+str(FLG.running_fold))
    train_report = ScoreReport()
    valid_report = ScoreReport()
    timer = SimpleTimer()
    fold_str = 'fold'+str(FLG.running_fold)
    best_score = dict(epoch=0, loss=1e+100, accuracy=0)

    #### create dataset ###
    # kfold split
    target_dict = np.load(pjoin(FLG.data_root, 'target_dict.pkl'))
    trainblock, validblock, ratio = fold_split(
        FLG.fold, FLG.running_fold, FLG.labels,
        np.load(pjoin(FLG.data_root, 'subject_indices.npy')),
        target_dict)

    def _dataset(block, transform):
        return ADNIDataset(FLG.labels, pjoin(FLG.data_root, FLG.modal),
                           block, target_dict, transform=transform)

    # create train set
    trainset = _dataset(trainblock, transform_presets(FLG.augmentation))

    # create normal valid set
    validset = _dataset(
        validblock, transform_presets(
            'nine crop' if FLG.augmentation == 'random crop' else 'no augmentation'))

    # each loader
    trainloader = DataLoader(trainset, batch_size=FLG.batch_size, shuffle=True,
                             num_workers=4, pin_memory=True)
    validloader = DataLoader(validset, num_workers=4, pin_memory=True)

    # data check
    # for image, _ in trainloader:
    #     summary.image3d('asdf', image)

    # create model
    def kaiming_init(tensor):
        return kaiming_normal_(tensor, mode='fan_out', nonlinearity='relu')
    if 'plane' in FLG.model:
        model = Plane(len(FLG.labels), name=FLG.model,
                      weights_initializer=kaiming_init)
    elif 'resnet11' in FLG.model:
        model = resnet11(len(FLG.labels), FLG.model,
                         weights_initializer=kaiming_init)
    elif 'resnet19' in FLG.model:
        model = resnet19(len(FLG.labels), FLG.model,
                         weights_initializer=kaiming_init)
    elif 'resnet35' in FLG.model:
        model = resnet35(len(FLG.labels), FLG.model,
                         weights_initializer=kaiming_init)
    elif 'resnet51' in FLG.model:
        model = resnet51(len(FLG.labels), FLG.model,
                         weights_initializer=kaiming_init)
    else:
        raise NotImplementedError(FLG.model)

    print_model_parameters(model)
    model = torch.nn.DataParallel(model, FLG.devices)
    model.to(device)

    # criterion
    train_criterion = torch.nn.CrossEntropyLoss(
        weight=torch.Tensor(
            list(map(lambda x: x*2, reversed(ratio))))).to(device)
    valid_criterion = torch.nn.CrossEntropyLoss().to(device)

    # TODO resume
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=FLG.lr,
                                 weight_decay=FLG.l2_decay)
    # scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, FLG.lr_gamma)

    start_epoch = 0
    global_step = start_epoch*len(trainloader)
    pbar = None
    for epoch in range(1, FLG.max_epoch+1):
        timer.tic()
        scheduler.step()
        summary.scalar('lr', fold_str,
                       epoch-1, optimizer.param_groups[0]['lr'],
                       ytickmin=0, ytickmax=FLG.lr)

        # train()
        torch.set_grad_enabled(True)
        model.train(True)
        train_report.clear()
        if pbar is None:
            pbar = tqdm(total=len(trainloader)*FLG.validation_term,
                        desc='Epoch {:<3}-{:>3} train'.format(
                epoch, epoch+FLG.validation_term-1))
        for images, targets in trainloader:
            images = images.cuda(device, non_blocking=True)
            targets = targets.cuda(device, non_blocking=True)

            optimizer.zero_grad()

            outputs = model(images)
            loss = train_criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_report.update_true(targets)
            train_report.update_score(F.softmax(outputs, dim=1))

            summary.scalar('loss', 'train '+fold_str,
                           global_step/len(trainloader), loss.item(),
                           ytickmin=0, ytickmax=1)

            pbar.update()
            global_step += 1

        if epoch % FLG.validation_term != 0:
            timer.toc()
            continue
        pbar.close()

        # valid()
        torch.set_grad_enabled(False)
        model.eval()
        valid_report.clear()
        pbar = tqdm(total=len(validloader),
                    desc='Epoch {:>3} valid'.format(epoch))
        for images, targets in validloader:
            true = targets
            npatchs = 1
            if len(images.shape) == 6:
                _, npatchs, c, x, y, z = images.shape
                images = images.view(-1, c, x, y, z)
                targets = torch.cat([
                    targets for _ in range(npatchs)]).squeeze()
            images = images.cuda(device, non_blocking=True)
            targets = targets.cuda(device, non_blocking=True)

            output = model(images)
            loss = valid_criterion(output, targets)

            valid_report.loss += loss.item()

            if npatchs == 1:
                score = F.softmax(output, dim=1)
            else:
                score = torch.mean(F.softmax(output, dim=1),
                                   dim=0, keepdim=True)
            valid_report.update_true(true)
            valid_report.update_score(score)

            pbar.update()
        pbar.close()

        # report
        vloss = valid_report.loss/len(validloader)
        summary.scalar('accuracy', 'train '+fold_str,
                       epoch, train_report.accuracy,
                       ytickmin=-0.05, ytickmax=1.05)

        summary.scalar('loss', 'valid '+fold_str,
                       epoch, vloss,
                       ytickmin=0, ytickmax=0.8)
        summary.scalar('accuracy', 'valid '+fold_str,
                       epoch, valid_report.accuracy,
                       ytickmin=-0.05, ytickmax=1.05)

        is_best = False
        if best_score['loss'] > vloss:
            best_score['loss'] = vloss
            best_score['epoch'] = epoch
            best_score['accuracy'] = valid_report.accuracy
            is_best = True

        print('Best Epoch {}: validation loss {} accuracy {}'.format(
            best_score['epoch'], best_score['loss'], best_score['accuracy']))

        # save
        if isinstance(model, torch.nn.DataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()

        save_checkpoint(
            dict(epoch=epoch,
                 best_score=best_score,
                 state_dict=state_dict,
                 optimizer_state_dict=optimizer.state_dict()),
            FLG.checkpoint_root, FLG.running_fold,
            FLG.model,
            is_best)
        pbar = None
        timer.toc()
        print('Time elapse {}h {}m {}s'.format(*timer.total()))


if __name__ == '__main__':
    main()
