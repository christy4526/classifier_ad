from __future__ import absolute_import, division, print_function

from os.path import join as pjoin
from tqdm import tqdm
import numpy as np
import nibabel as nib
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from models import Plane, resnet11, resnet19, resnet35, resnet51
from config import test_args

from transforms import transform_presets, ToFloatTensor, CenterCrop
from torchvision.transforms import Compose

import pickle
from dataset import fold_split, ADNIDataset
from utils import load_checkpoint, save_checkpoint
from utils import Summary, SimpleTimer, ScoreReport, save_checkpoint
from sklearn import metrics


def original_load(validblock, target_dict, transformer, device):
    originalset = ADNIDataset(FLG.labels, pjoin(FLG.data_root, 'spm_normalized'),
                              validblock, target_dict, transform=transformer)
    originloader = DataLoader(originalset, pin_memory=True)
    for image, target in originloader:
        if len(image.shape) == 6:
            _, npatches, c, x, y, z = image.shape
            image = image.view(-1, c, x, y, z)
            target = torch.stack(
                [target for _ in range(npatches)]).squeeze()
        image = image.cuda(device, non_blocking=True)
        target = target.cuda(device, non_blocking=True)
        break
    return image, target

def test(FLG):
    device = torch.device('cuda:{}'.format(FLG.devices[0]))
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(FLG.devices[0])
    report = [ScoreReport() for _ in range(FLG.fold)]
    overall_report = ScoreReport()
    target_dict = np.load(pjoin(FLG.data_root, 'target_dict.pkl'))

    with open(FLG.model+'_stat.pkl', 'rb') as f:
        stat = pickle.load(f)
    summary = Summary(port=10001, env=str(FLG.model)+'CAM')

    class Feature(object):
        def __init__(self):
            self.blob = None
        def capture(self, blob):
            self.blob = blob

    if 'plane' in FLG.model:
        model = Plane(len(FLG.labels), name=FLG.model)
    elif 'resnet11' in FLG.model:
        model = resnet11(len(FLG.labels), FLG.model)
    elif 'resnet19' in FLG.model:
        model = resnet19(len(FLG.labels), FLG.model)
    elif 'resnet35' in FLG.model:
        model = resnet35(len(FLG.labels), FLG.model)
    elif 'resnet51' in FLG.model:
        model = resnet51(len(FLG.labels), FLG.model)
    else:
        raise NotImplementedError(FLG.model)
    model.to(device)

    ad_h=[]
    nl_h=[]
    adcams = np.zeros((4,3,112,144,112), dtype="f8")
    nlcams = np.zeros((4,3,112,144,112), dtype="f8")
    sb=[9.996e-01, 6.3e-01, 1.001e-01]
    for running_fold in range(FLG.fold):
        _, validblock, _ = fold_split(
            FLG.fold, running_fold, FLG.labels,
            np.load(pjoin(FLG.data_root, 'subject_indices.npy')),
            target_dict)
        validset = ADNIDataset(FLG.labels, pjoin(FLG.data_root, FLG.modal),
                               validblock, target_dict,
                               transform=transform_presets(FLG.augmentation))
        validloader = DataLoader(validset, pin_memory=True)

        epoch, _ = load_checkpoint(model, FLG.checkpoint_root,
                                   running_fold, FLG.model, None, True)
        model.eval()
        feature = Feature()

        def hook(mod, inp, oup): return feature.capture(oup.data.cpu().numpy())
        _ = model.layer4.register_forward_hook(hook)
        fc_weights = model.fc.weight.data.cpu().numpy()

        transformer = Compose([CenterCrop((112,144,112)), ToFloatTensor()])
        im, _ = original_load(validblock, target_dict, transformer, device)

        for image, target in validloader:
            true = target
            npatches = 1
            if len(image.shape) == 6:
                _, npatches, c, x, y, z = image.shape
                image = image.view(-1, c, x, y, z)
                target = torch.stack(
                    [target for _ in range(npatches)]).squeeze()
            image = image.cuda(device, non_blocking=True)
            target = target.cuda(device, non_blocking=True)

            output = model(image)

            if npatches == 1:
                score = F.softmax(output, dim=1)
            else:
                score = torch.mean(F.softmax(output, dim=1),
                                   dim=0, keepdim=True)

            report[running_fold].update_true(true)
            report[running_fold].update_score(score)

            overall_report.update_true(true)
            overall_report.update_score(score)

            print(target)
            if FLG.cam:
                s=0
                cams=[]
                if target[0] == 0:
                    s = score[0][0]
                    #s = s.cpu().numpy()[()]
                    cams = adcams
                else:
                    sn = score[0][1]
                    #s = s.cpu().numpy()[()]
                    cams = nlcams
                if s > sb[0]:
                    cams[0] = summary.cam3d(FLG.labels[target], im, feature.blob, fc_weights, target,
                                              cams[0], s, num_images=5)
                elif s > sb[1]:
                    cams[1] = summary.cam3d(FLG.labels[target], im, feature.blob, fc_weights, target,
                                              cams[1], s, num_images=5)
                elif s > sb[2]:
                    cams[2] = summary.cam3d(FLG.labels[target], im, feature.blob, fc_weights, target,
                                              cams[2], s, num_images=5)
                else:
                    cams[3] = summary.cam3d(FLG.labels[target], im, feature.blob, fc_weights, target,
                                              cams[3], s, num_images=5)
                #ad_h += [s]
                #nl_h += [sn]

        print('At {}'.format(epoch))
        print(metrics.classification_report(
            report[running_fold].y_true,
            report[running_fold].y_pred,
            target_names=FLG.labels, digits=4))
        print('accuracy {}'.format(report[running_fold].accuracy))

    #print(np.histogram(ad_h))
    #print(np.histogram(nl_h))

    print('over all')
    print(metrics.classification_report(
        overall_report.y_true,
        overall_report.y_pred,
        target_names=FLG.labels, digits=4))
    print('accuracy {}'.format(overall_report.accuracy))

    with open(FLG.model+'_stat.pkl', 'wb') as f:
        pickle.dump(report, f, pickle.HIGHEST_PROTOCOL)


def test_cam():
    with open(FLG.model+'_stat.pkl', 'rb') as f:
        stat = pickle.load(f)
    summary = Summary(port=10001, env=str(FLG.model)+'CAM')

    class Feature(object):
        def __init__(self):
            self.blob = None
        def capture(self, blob):
            self.blob = blob

    # TODO: create model
    device = torch.device('cuda:{}'.format(FLG.devices[0]))
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(FLG.devices[0])
    report = [ScoreReport() for _ in range(FLG.fold)]
    target_dict = np.load(pjoin(FLG.data_root, 'target_dict.pkl'))

    model = Plane(len(FLG.labels), name=FLG.model)
    model.to(device)

    transformer = Compose([CenterCrop((112,144,112)), ToFloatTensor()])

    def original_load(validblock):
        originalset = ADNIDataset(FLG.labels, pjoin(FLG.data_root, 'spm_normalized'),
                               validblock, target_dict, transform=transformer)
        originloader = DataLoader(originalset, pin_memory=True)
        for image, target in originloader:
            if len(image.shape) == 6:
                _, npatches, c, x, y, z = image.shape
                image = image.view(-1, c, x, y, z)
                target = torch.stack(
                    [target for _ in range(npatches)]).squeeze()
            image = image.cuda(device, non_blocking=True)
            target = target.cuda(device, non_blocking=True)
            break
        return image, target

    hadcams = np.zeros((3,112,144,112), dtype="f8")
    madcams = np.zeros((3,112,144,112), dtype="f8")
    sadcams = np.zeros((3,112,144,112), dtype="f8")
    zadcams = np.zeros((3,112,144,112), dtype="f8")

    nlcams = np.zeros((4,3,112,144,112), dtype="f8")
    sb=[4.34444371e-16, 1.67179015e-18, 4.08813312e-23]
    #im, _ = original_load(validblock)
    for running_fold in range(FLG.fold):
        # validset
        _, validblock, _ = fold_split(FLG.fold, running_fold, FLG.labels,
                                      np.load(pjoin(FLG.data_root, 'subject_indices.npy')), target_dict)
        validset = ADNIDataset(FLG.labels, pjoin(FLG.data_root, FLG.modal),
                               validblock, target_dict, transform=transformer)
        validloader = DataLoader(validset, pin_memory=True)

        load_checkpoint(model, FLG.checkpoint_root, running_fold, FLG.model,
                       epoch=None, is_best=True)
        model.eval()
        feature = Feature()

        def hook(mod, inp, oup): return feature.capture(oup.data.cpu().numpy())
        _ = model.layer4.register_forward_hook(hook)
        fc_weights = model.fc.weight.data.cpu().numpy()

        im, _ = original_load(validblock)
        ad_s=[]
        for image, target in validloader:
            true = target
            npatches = 1
            if len(image.shape) == 6:
                _, npatches, c, x, y, z = image.shape
                image = image.view(-1, c, x, y, z)
                target = torch.stack(
                    [target for _ in range(npatches)]).squeeze()
            image = image.cuda(device, non_blocking=True)
            target = target.cuda(device, non_blocking=True)

            #_ = model(image.view(*image.shape))
            output = model(image)

            if npatches == 1:
                score = F.softmax(output, dim=1)
            else:
                score = torch.mean(F.softmax(output, dim=1),
                                   dim=0, keepdim=True)

            sa = score[0][1]
            #name = 'k'+str(running_fold)
            sa = sa.cpu().numpy()[()]
            print(score, score.shape)
            if true == torch.tensor([ 0]):
                if sa > sb[0]:
                    hadcams = summary.cam3d(FLG.labels[target], im, feature.blob, fc_weights, target,
                                              hadcams, sa, num_images=5)
                elif sa > sb[1]:
                    madcams = summary.cam3d(FLG.labels[target], im, feature.blob, fc_weights, target,
                                              madcams, sa, num_images=5)
                elif sa > sb[2]:
                    sadcams = summary.cam3d(FLG.labels[target], im, feature.blob, fc_weights, target,
                                              sadcams, sa, num_images=5)
                else:
                    zadcams = summary.cam3d(FLG.labels[target], im, feature.blob, fc_weights, target,
                                              zadcams, sa, num_images=5)
            else:
                if s > sb[0]:
                    nlcams[0] = summary.cam3d(FLG.labels[target], im, feature.blob, fc_weights, target,
                                              nlcams[0], sr, num_images=5)
                elif sr > sb[1]:
                    nlcams[1] = summary.cam3d(FLG.labels[target], im, feature.blob, fc_weights, target,
                                              nlcams[1], sr, num_images=5)
                elif sr > sb[2]:
                    nlcams[2] = summary.cam3d(FLG.labels[target], im, feature.blob, fc_weights, target,
                                              nlcams[2], sr, num_images=5)
                else:
                    nlcams[3] = summary.cam3d(FLG.labels[target], im, feature.blob, fc_weights, target,
                                              nlcams[3], sr, num_images=5)
            ad_s += [sr]
        print('histogram',  np.histogram(ad_s))


def test_plot():
    with open(FLG.model+'_stat.pkl', 'rb') as f:
        reports = pickle.load(f)
    summary = Summary(port=10001, env=FLG.model)

    over_y_true = []
    over_y_score = []
    for running_fold in range(FLG.fold):
        report = reports[running_fold]
        over_y_true += report.y_true
        over_y_score += report.y_score
        summary.precision_recall_curve(report.y_true, report.y_score,
                                       true_label=0, name='fold'+str(running_fold))
        summary.roc_curve(report.y_true, report.y_score,
                          true_label=0, name='fold'+str(running_fold))

    summary.precision_recall_curve(over_y_true, over_y_score,
                                   true_label=0, name='over all')
    summary.roc_curve(over_y_true, over_y_score,
                      true_label=0, name='over all')

if __name__ == '__main__':
    FLG = test_args()
    if FLG.plot:
        test_plot()
    else:
        test(FLG)
