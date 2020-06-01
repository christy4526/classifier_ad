from __future__ import absolute_import, division, print_function

from os.path import join as pjoin
import argparse


def argument_report(arg, end='\n'):
    d = arg.__dict__
    keys = d.keys()
    report = '{:15}    {}'.format('running_fold', d['running_fold'])+end
    report += '{:15}    {}'.format('memo', d['memo'])+end
    for k in sorted(keys):
        if k == 'running_fold' or k == 'memo':
            continue
        report += '{:15}    {}'.format(k, d[k])+end
    return report


def base_parser():
    _base_parser = argparse.ArgumentParser(add_help=False)

    _base_parser.add_argument('model', type=str)
    _base_parser.add_argument('--data_root', type=str,
                              default=pjoin('data', 'ADNI'))

    _base_parser.add_argument('--modal', type=str, default='spm_normalized')
    _base_parser.add_argument('--augmentation', type=str,
                              default='random crop')
    _base_parser.add_argument('--checkpoint_root', type=str,
                              default='checkpoint')

    _base_parser.add_argument('--devices', type=int,
                              nargs='+', default=(6, 7, 8, 9))

    _base_parser.add_argument('--labels', type=str,
                              nargs='+', default=('AD', 'NL'))

    _base_parser.add_argument('--fold', type=int, default=5)
    _base_parser.add_argument('--running_fold', type=int, default=0)
    _base_parser.add_argument('--slient', action='store_true')
    _base_parser.add_argument('--memo', type=str, default='')

    return _base_parser


def train_args():
    _base_parser = base_parser()
    _train_parser = argparse.ArgumentParser(parents=[_base_parser])

    _train_parser.add_argument('--max_epoch', type=int, default=300)
    _train_parser.add_argument('--batch_size', type=int, default=84)
    _train_parser.add_argument('--l2_decay', type=float, default=0.0005)
    _train_parser.add_argument('--lr', type=float, default=0.001)
    _train_parser.add_argument('--lr_gamma', type=float, default=0.98)
    _train_parser.add_argument('--validation_term', type=int, default=5)

    args = _train_parser.parse_args()
    return args


def test_args():
    _base_parser = base_parser()
    _test_parser = argparse.ArgumentParser(parents=[_base_parser])

    _test_parser.add_argument('--plot', action='store_true')
    _test_parser.add_argument('--cam', action='store_true')

    arg = _test_parser.parse_args()
    return arg
