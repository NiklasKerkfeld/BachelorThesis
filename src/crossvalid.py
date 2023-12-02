import argparse
import csv
import glob
import os
from datetime import datetime
from os.path import isfile
from pathlib import Path
from pprint import pprint

from monai.data import CacheDataset
from monai.losses import DiceLoss, DiceCELoss, DiceFocalLoss
from monai.networks.nets import BasicUNet, SwinUNETR
from monai.transforms import Compose, LoadImaged, RandRotated, RandShiftIntensityd, RandScaleIntensityd, \
    RandSpatialCropd, ConcatItemsd, ToTensord, SpatialPadd, CenterSpatialCropd
from torch.utils.tensorboard import SummaryWriter

from src.CustomTransforms import RandomDropSequenced
from src.ResNetUNet import Model

from src.trainer import params, Trainer

CROP = (128, 128, 32)

train_transform = Compose([
    # load data
    LoadImaged(keys=['t2w', 'adc', 'hbv', 'label', 'prostate'], ensure_channel_first=True),
    # randomly rotate in plane
    RandRotated(keys=['t2w', 'adc', 'hbv', 'label', 'prostate'], range_x=0.0, range_y=0.0, range_z=0.17, prob=0.2,
                keep_size=True, mode=['bilinear'] * 3 + ['nearest', 'nearest']),
    # randomly shift intensity by 0.1
    RandShiftIntensityd(keys=['t2w', 'adc', 'hbv'], offsets=0.1, prob=0.2),
    # randomly scale intensity by 1 +- 0.1
    RandScaleIntensityd(keys=['t2w', 'adc', 'hbv'], factors=0.1, prob=0.2),
    # randomly crop images
    RandSpatialCropd(keys=['t2w', 'adc', 'hbv', 'label', 'prostate'], roi_size=CROP, random_size=False),
    # randomly drop the adc and hbv sequences
    RandomDropSequenced(keys=['t2w', 'adc', 'hbv'], probability=(0.0, 0.1, 0.2)),
    # combine all image sequences to one tensor
    ConcatItemsd(keys=['t2w', 'adc', 'hbv'], name='tensor'),
    # create torch tensors
    ToTensord(keys=['tensor', 'label', 'prostate'])
])
test_transform = Compose([
    # load data
    LoadImaged(keys=['t2w', 'adc', 'hbv', 'label', 'prostate'], ensure_channel_first=True),
    # ensure image is at least 224 x 224 x 32 in size
    SpatialPadd(keys=['t2w', 'adc', 'hbv', 'label', 'prostate'], spatial_size=(224, 224, 32)),
    # cut out crop around the center
    CenterSpatialCropd(keys=['t2w', 'adc', 'hbv', 'label', 'prostate'], roi_size=(224, 224, 32)),
    # combine all image sequences to one tensor
    ConcatItemsd(keys=['t2w', 'adc', 'hbv'], name='tensor'),
    # create torch tensors
    ToTensord(keys=['tensor', 'label', 'prostate'])
])


def get_split(data, testfold: int):
    # Calculate the size of each fold
    fold_size = len(data) // 5

    # Calculate the indices for each fold
    indices = [(i * fold_size, (i + 1) * fold_size) for i in range(5)]

    # Adjust the last fold to include any leftover elements
    indices[-1] = (indices[-1][0], len(data))

    print(f"{indices=}")

    splits = [data[start: end] for start, end in indices]

    traindata = [d for i, d in enumerate(splits) if i != testfold]
    traindata = [item for sublist in traindata for item in sublist]

    testdata = splits[testfold]

    return traindata, testdata


def get_datasets(path: str, testfold: int):
    """

    :param testfold:
    :return:
    """
    """
    creates a list of dicts with all file paths
    :param path: path to folder with data
    :return: List of dicts
    """
    positives = []
    folders = [x for x in glob.glob(f"{path}/*/positive/*") if not isfile(x)]
    for folder in folders:
        item = {}
        item['adc'] = [os.path.normpath(x) for x in glob.glob(f"{folder}/ADC_roi.mha")][0]
        item['hbv'] = [os.path.normpath(x) for x in glob.glob(f"{folder}/DWI_b1500_roi.mha")][0]
        item['t2w'] = [os.path.normpath(x) for x in glob.glob(f"{folder}/T2_tra_roi.mha")][0]
        item['label'] = [os.path.normpath(x) for x in glob.glob(f"{folder}/lesion_roi.nii.gz")][0]
        item['prostate'] = [os.path.normpath(x) for x in glob.glob(f"{folder}/prostate_roi.nii.gz")][0]
        item['case'] = 'positive'
        item['folder'] = folder
        positives.append(item)

    negatives = []
    folders = [x for x in glob.glob(f"{path}/*/negative/*") if not isfile(x)]
    for folder in folders:
        item = {}
        item['adc'] = [os.path.normpath(x) for x in glob.glob(f"{folder}/ADC_roi.mha")][0]
        item['hbv'] = [os.path.normpath(x) for x in glob.glob(f"{folder}/DWI_b1500_roi.mha")][0]
        item['t2w'] = [os.path.normpath(x) for x in glob.glob(f"{folder}/T2_tra_roi.mha")][0]
        item['label'] = [os.path.normpath(x) for x in glob.glob(f"{folder}/lesion_roi.nii.gz")][0]
        item['prostate'] = [os.path.normpath(x) for x in glob.glob(f"{folder}/prostate_roi.nii.gz")][0]
        item['case'] = 'negative'
        item['folder'] = folder
        negatives.append(item)

    positives_train, positives_test = get_split(positives, testfold)
    negatives_train, negatives_test = get_split(negatives, testfold)

    traindata = positives_train + negatives_train
    testdata = positives_test + negatives_test

    return CacheDataset(traindata, train_transform),  CacheDataset(testdata, test_transform)


def main(args):
    for i in range(5):
        name = args['name'] + f"_fold_{i}"
        # setup tensor board
        train_log_dir = f'{Path(__file__).parent.absolute()}/../logs/runs/' + name
        print(f"{train_log_dir=}")
        writer = SummaryWriter(train_log_dir)

        train_dataset, test_dataset = get_datasets(f"{Path(__file__).parent.absolute()}/../data/Inhouse_split", i)
        print(f"{len(train_dataset)=}")
        print(f"{len(test_dataset)=}")

        # setup src
        if args['model'] == 'basic':
            model = BasicUNet(spatial_dims=3, in_channels=3, features=(32, 32, 64, 128, 256, 32))
        elif args['model'] == 'mymodel':
            model = Model(3, 2)
        elif args['model'] == 'transformer':
            model = SwinUNETR(img_size=(128, 128, 32), in_channels=3, out_channels=2, depths=(2, 2, 2, 2),
                              num_heads=(3, 6, 12, 24), feature_size=24)
        else:
            raise Exception('no model with with name!')

        # setup src
        if args['loss'] == 'dice':
            loss = DiceLoss(include_background=False, to_onehot_y=True)
        elif args['loss'] == 'dicece':
            loss = DiceCELoss(include_background=False, to_onehot_y=True)
        elif args['loss'] == 'dicefocal':
            loss = DiceFocalLoss(include_background=False, to_onehot_y=True)
        elif args['loss'] == 'ce':
            loss = DiceCELoss(include_background=False, to_onehot_y=True, lambda_dice=0.0)
        else:
            raise Exception('no loss function with with name!')

        # setup Trainer
        trainer = Trainer(model=model, loss_function=loss, writer=writer,
                          train_dataset=train_dataset,
                          valid_dataset=test_dataset,
                          config=args)

        # train
        results = trainer.train(epochs=args['epochs'])

        # save results
        dice_results = [name, args['epochs'], args['model'], args['loss'], args['lr'], args['weight_decay'],
                        args['part'], 'dice', *results['dice'].values()]
        ap_results = [name, args['epochs'], args['model'], args['loss'], args['lr'], args['weight_decay'],
                      args['part'], 'AP', *results['ap'].values()]

        if not os.path.exists(f'{Path(__file__).parent.absolute()}/../logs/resultsCrossvalid.csv'):
            with open(f'{Path(__file__).parent.absolute()}/../logs/resultsCrossvalid.csv', 'x', newline='') as file:
                writer = csv.writer(file, delimiter='|')
                writer.writerow(
                    ['name', 'epochs', 'model', 'loss', 'lr', 'weight_decay', 'part', 'best', 'loss', 'dice',
                     'dice_median', 'dice_std', 'iou', 'cross_entropy', 'accuracy', 'volume', 'AP',
                     'AUROC', 'score', 'step', 'time'])

        with open(f'{Path(__file__).parent.absolute()}/../logs/resultsCrossvalid.csv', 'a', newline='') as file:
            writer = csv.writer(file, delimiter='|')
            writer.writerow(dice_results)
            writer.writerow(ap_results)


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
    parser.add_argument('--pretrained', '-p', dest='pretrained', default=False, type=bool,
                        help='load a pretrained model before training')

    return parser.parse_args()


if __name__ == '__main__':
    args = vars(get_args())

    if args['model'] == 'basic':
        args['lr'] = 1e-4
        args['weight_decay'] = 1e-2
        args['loss'] = 'dicefocal'

    elif args['model'] == 'mymodel':
        args['lr'] = 1e-3
        args['weight_decay'] = 1e-2
        args['loss'] = 'dicece'

    elif args['model'] == 'transformer':
        args['lr'] = 1e-2
        args['weight_decay'] = 1e-2
        args['loss'] = 'dice'

    else:
        raise Exception("given src unknown!")

    if args['pretrained']:
        args['lr'] *= 0.1

    params.update(args)
    pprint(params)
    print("\n")

    main(params)
