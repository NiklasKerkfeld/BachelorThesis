import csv
import json
import os
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from monai.losses import DiceLoss, DiceCELoss, DiceFocalLoss
from monai.networks.nets import BasicUNet, SwinUNETR

from src.ResNetUNet import Model
from src.utils import get_dataset
from src.trainer import params, Trainer

torch.manual_seed(1679818121452)


def main():
    """
    executes the next training run from the runs-json file
    :return:
    """
    keys = ['name', 'epochs', 'model', 'loss', 'lr', 'weight_decay', 'part']

    # read parameter
    with open(f"{Path(__file__).parent.absolute()}/../data/run_count.txt", "r") as f:
        run_number = int(f.read())

    with open(f"{Path(__file__).parent.absolute()}/../data/run_count.txt", "w") as f:
        f.write(str(run_number + 1))

    with open(f'{Path(__file__).parent.absolute()}/../data/runs.json', 'r') as f:
        args = {k: v for k, v in zip(keys, json.load(f)[run_number])}

    # start training
    params.update(args)
    print(f"start model {args['name']}")

    # setup tensor board
    train_log_dir = f'{Path(__file__).parent.absolute()}/../logs/runs/' + args['name']
    print(f"{train_log_dir=}")

    writer = SummaryWriter(train_log_dir)

    # setup Datasets
    train_dataset = get_dataset(f'{Path(__file__).parent.absolute()}/../data/split/train', dataset='picai',
                                   sequences=['t2w', 'adc', 'hbv'], part=params['part'], train_mode=True)
    valid_dataset = get_dataset(f'{Path(__file__).parent.absolute()}/../data/split/valid', dataset='picai',
                                   sequences=['t2w', 'adc', 'hbv'], train_mode=False)

    print('\nthe datasets:')
    print(f"{len(train_dataset)=}")
    print(f"{len(valid_dataset)=}")

    # setup src
    if params['model'] == 'basic':
        model = BasicUNet(spatial_dims=3, in_channels=3, features=(32, 32, 64, 128, 256, 32))
    elif params['model'] == 'mymodel':
        model = Model(3, 2)
    elif params['model'] == 'transformer':
        model = SwinUNETR(img_size=(128, 128, 32), in_channels=3, out_channels=2, depths=(2, 2, 2, 2),
                          num_heads=(3, 6, 12, 24), feature_size=24)
    else:
        raise Exception('no model with with name!')

    # setup src
    if params['loss'] == 'dice':
        loss = DiceLoss(include_background=False, to_onehot_y=True)
    elif params['loss'] == 'dicece':
        loss = DiceCELoss(include_background=False, to_onehot_y=True)
    elif params['loss'] == 'dicefocal':
        loss = DiceFocalLoss(include_background=False, to_onehot_y=True)
    elif params['loss'] == 'ce':
        loss = DiceCELoss(include_background=False, to_onehot_y=True, lambda_dice=0.0)
    else:
        raise Exception('no loss function with with name!')

    # setup Trainer
    trainer = Trainer(model=model, loss_function=loss, writer=writer,
                      train_dataset=train_dataset,
                      valid_dataset=valid_dataset,
                      config=params)

    # train
    results = trainer.train(epochs=params['epochs'])

    # save results
    dice_results = [*args.values(), 'dice', *results['dice'].values()]
    ap_results = [*args.values(), 'AP', *results['ap'].values()]

    if not os.path.exists(f'{Path(__file__).parent.absolute()}/../logs/results.csv'):
        with open(f'{Path(__file__).parent.absolute()}/../logs/results.csv', 'x', newline='') as file:
            writer = csv.writer(file, delimiter='|')
            writer.writerow(['name', 'epochs', 'model', 'loss', 'lr', 'weight_decay', 'part', 'best', 'loss', 'dice',
                             'dice_median', 'dice_std', 'iou', 'cross_entropy', 'accuracy', 'volume', 'AP',
                             'AUROC', 'score', 'step', 'time'])

    with open(f'{Path(__file__).parent.absolute()}/../logs/results.csv', 'a', newline='') as file:
        writer = csv.writer(file, delimiter='|')
        writer.writerow(dice_results)
        writer.writerow(ap_results)


if __name__ == '__main__':
    main()
