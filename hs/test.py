from __future__ import absolute_import, division, print_function

from os.path import join as pjoin
from tqdm import tqdm
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from models import Plane, resnet11, resnet19, resnet35, resnet51
from config import test_args

from transforms import transform_presets

import pickle
from dataset import fold_split, ADNIDataset
from utils import load_checkpoint, save_checkpoint
from utils import Summary, SimpleTimer, ScoreReport, save_checkpoint
from sklearn import metrics


def test(FLG):
    device = torch.device('cuda:{}'.format(FLG.devices[0]))
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(FLG.devices[0])
    report = [ScoreReport() for _ in range(FLG.fold)]
    overall_report = ScoreReport()
    target_dict = np.load(pjoin(FLG.data_root, 'target_dict.pkl'))

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

        print('At {}'.format(epoch))
        print(metrics.classification_report(
            report[running_fold].y_true,
            report[running_fold].y_pred,
            target_names=FLG.labels, digits=4))
        print('accuracy {}'.format(report[running_fold].accuracy))

    print('over all')
    print(metrics.classification_report(
        overall_report.y_true,
        overall_report.y_pred,
        target_names=FLG.labels, digits=4))
    print('accuracy {}'.format(overall_report.accuracy))

    with open(FLG.model+'_stat.pkl', 'wb') as f:
        pickle.dump(report, f, pickle.HIGHEST_PROTOCOL)


def test_plot():
    with open(FLG.model+'_stat.pkl', 'rb') as f:
        reports = pickle.load(f)
    summary = Summary(port=39199, env=FLG.model)

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


def test_cam():
    FLAGS.noadapt = True
    summary = Summary(port=39199, env=FLAGS.visdom_env)
    with open(FLAGS.model+'_stat.pkl', 'rb') as f:
        stat = pickle.load(f)

    class Feature(object):
        def __init__(self):
            self.blob = None

        def capture(self, blob):
            self.blob = blob

    # TODO: cropped cam?
    transformer = Compose([
        CenterCrop((112, 144, 112)),
        ToFloatTensor()])

    model = create_model()
    model = model.cuda(FLAGS.devices[0])
    for running_k in range(FLAGS.kfold):
        load_checkpoint(model, FLAGS.checkpoint_root, running_k,
                        FLAGS.model, epoch=None, is_best=True)
        model.eval()

        feature = Feature()

        def hook(mod, inp, oup): return feature.capture(oup.data.cpu().numpy())
        _ = model.layer4.register_forward_hook(hook)
        fc_weights = model.fc.weight.data.cpu().numpy()

        _, validset, _ =\
            make_kfold_dataset(FLAGS.kfold, running_k,
                               np.load(pjoin(FLAGS.root, 'sids.npy')),
                               np.load(pjoin(FLAGS.root, 'diagnosis.npz')),
                               FLAGS.labels, [FLAGS.dataset_root],
                               validset_loader=ScaledLoader(
                                   (112, 144, 112), False),
                               valid_transform=transformer)

        i, p = stat[running_k]['pos_best']['index'], stat[running_k]['pos_best']['p']
        i = 8
        image, target = validset[i]
        _ = model(Variable(image.view(1, *image.shape).cuda(FLAGS.devices[0])))

        name = 'k'+str(running_k)

        summary.cam3d(name+FLAGS.labels[target]+'_pos_best_'+str(p),
                      image, feature.blob, fc_weights, target, num_images=10)

        i, p = stat[running_k]['neg_best']['index'], stat[running_k]['neg_best']['p']
        i = 45
        image, target = validset[i]
        _ = model(Variable(image.view(1, *image.shape).cuda(FLAGS.devices[0])))

        summary.cam3d(name+FLAGS.labels[target]+'_neg_best_'+str(p),
                      image, feature.blob, fc_weights, target, num_images=10)


if __name__ == '__main__':
    FLG = test_args()

    if FLG.cam:
        test_cam()
    elif FLG.plot:
        test_plot()
    else:
        test(FLG)
