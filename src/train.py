"""
script for training a model
"""

import argparse
from datetime import datetime

from pathlib import Path
from typing import Tuple, Dict

import torch
from monai.losses import DiceCELoss, DiceLoss, DiceFocalLoss
from monai.networks.nets import BasicUNet, SwinUNETR

from torch.utils.tensorboard import SummaryWriter

from src.ResNetUNet import Model
from src.trainer import Trainer, params
from src.utils import get_dataset

torch.manual_seed(1679818121452)


def main(args: Dict):
    """
    starts training of model
    :param args: dict with parameter for training
    :return: None
    """
    # setup tensor board
    train_log_dir = f'{Path(__file__).parent.absolute()}/../logs/runs/' + args['name']
    print(f"{train_log_dir=}")

    writer = SummaryWriter(train_log_dir)

    params.update(args)

    # setup Dataset
    folder = "split" if params['dataset'] == 'picai' else 'Inhouse_split'
    train_dataset = get_dataset(f'{Path(__file__).parent.absolute()}/../data/{folder}/train', dataset=params['dataset'],
                                sequences=['t2w', 'adc', 'hbv'],
                                train_mode=True)
    valid_dataset = get_dataset(f'{Path(__file__).parent.absolute()}/../data/{folder}/valid', dataset=params['dataset'],
                                sequences=['t2w', 'adc', 'hbv'],
                                train_mode=False)
    test_dataset = get_dataset(f'{Path(__file__).parent.absolute()}/../data/{folder}/valid', dataset=params['dataset'],
                               sequences=['t2w', 'adc', 'hbv'],
                               train_mode=False)

    print('\nthe datasets:')
    print(f"{len(train_dataset)=}")
    print(f"{len(valid_dataset)=}")
    print(f"{len(test_dataset)=}")

    # setup src
    if params['model'] == 'basic':
        model = BasicUNet(spatial_dims=3, in_channels=3, features=(32, 32, 64, 128, 256, 32))
    elif params['model'] == 'resnet':
        model = Model(3, 2)
    elif params['model'] == 'transformer':
        model = SwinUNETR(img_size=(128, 128, 32), in_channels=3, out_channels=2, depths=(2, 2, 2, 2),
                          num_heads=(3, 6, 12, 24), feature_size=24)
    else:
        raise Exception('no model with with name!')

    # setup src
    if args['loss'] == 'dice':
        loss = DiceLoss(include_background=False, to_onehot_y=True)
    elif args['loss'] == 'dicece':
        loss = DiceCELoss(include_background=False, to_onehot_y=True, ce_weight=torch.tensor(params['ce_weights']))
    elif args['loss'] == 'dicefocal':
        loss = DiceFocalLoss(include_background=False, to_onehot_y=True,
                             focal_weight=torch.tensor(params['ce_weights']))
    else:
        raise Exception('no model with with name!')

    # setup Trainer
    trainer = Trainer(model=model, loss_function=loss, writer=writer,
                      train_dataset=train_dataset, valid_dataset=valid_dataset,
                      config=params, device=params['cuda'])

    # trainer.load('basic_dicefocal_lr4_wd2_dice')

    # train
    results = trainer.train(epochs=params['epochs'])

    return results


def get_args() -> argparse.Namespace:
    """
    get args for training the model
    :return: args object
    """
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--epochs', '-e', metavar='EPOCHS', default=1, type=int, help='number of epochs to train')
    parser.add_argument('--name', '-n', metavar='NAME', type=str, default=datetime.now().strftime("%Y%m%d-%H%M%S"),
                        help='name of run in tensorboard')
    parser.add_argument('--model', '-m', dest='model', default='basic', type=str,
                        help='type of model to train')
    parser.add_argument('--loss', '-l', dest='loss', default='dice', type=str,
                        help='loss function for training')
    parser.add_argument('--batch-size', '-b', dest='batch_size', default=params['batch_size'], metavar='B', type=int,
                        help='Batch size')
    parser.add_argument('--learning-rate', '-lr', metavar='LR', default=params['lr'], type=float, help='Learning rate',
                        dest='lr')
    parser.add_argument('--weight_decay', '-wd', metavar='WD', default=params['weight_decay'], type=float,
                        help='weight_decay', dest='weight_decay')
    parser.add_argument('--gamma', '-g', metavar='gamma', default=params['gamma'], type=float,
                        help='gamma for exponential lr scheduler', dest='gamma')
    parser.add_argument('--ce_weights', '-cw', default=params['ce_weights'], metavar='ce_weights',
                        type=Tuple[float, float], help='weights for classes in CE loss', dest='ce_weights')
    parser.add_argument('--cuda-device', '-c', dest='cuda', type=int, default=0,
                        help='Cuda device string')
    parser.add_argument('--worker', '-w', dest='worker', type=int, default=params['worker'],
                        help='Number of workers for Dataloader')
    parser.add_argument('--dataset', '-d', dest='dataset', type=str, default='picai',
                        help='Name of the dataset to train with (\'picai\', \'inhouse\')')

    return parser.parse_args()


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    args = vars(get_args())

    main(args)
