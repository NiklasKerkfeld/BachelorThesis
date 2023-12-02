import glob
import os

from os.path import isfile
from typing import List

from monai.data import CacheDataset
from monai.transforms import (Compose, LoadImaged, ToTensord, ConcatItemsd, RandRotated, CenterSpatialCropd,
                              SpatialPadd, RandSpatialCropd, RandShiftIntensityd, RandScaleIntensityd)

from src.CustomTransforms import RandomDropSequenced, DropSequenced

N_POSITIVE = 7 # number of times the positive examples are in the trainingsset
CROP = (128, 128, 32)


def get_dataset(path: str, sequences: List[str], dataset: str = 'picai', part: float=1.0, sequence_drop: List[str] = [],
    train_mode: bool = False):
    """
    creates a CacheDataset with data in path
    :param sequence_drop: list of sequences that are replaced with zeros while testing
    :param path: path to data
    :param sequences: sequences to use for dataset
    :param part: part of the dataset to use
    :param train_mode: used augmentations if true
    :return: CacheDataset
    """
    if train_mode:
        transform = Compose([
            # load data
            LoadImaged(keys=sequences + ['label', 'prostate'], ensure_channel_first=True),
            # randomly rotate in plane
            RandRotated(keys=sequences + ['label', 'prostate'], range_x=0.0, range_y=0.0, range_z=0.17, prob=0.2,
                        keep_size=True, mode=['bilinear'] * len(sequences) + ['nearest', 'nearest']),
            # randomly shift intensity by +-0.1
            RandShiftIntensityd(keys=sequences, offsets=0.1, prob=0.2),
            # randomly scale intensity by 1 +- 0.1
            RandScaleIntensityd(keys=sequences, factors=0.1, prob=0.2),
            # randomly crop images
            RandSpatialCropd(keys=sequences + ['label', 'prostate'], roi_size=CROP, random_size=False),
            # randomly drop the adc and hbv sequences
            RandomDropSequenced(keys=sequences, probability=(0.0, 0.1, 0.2)[:len(sequences)]),
            # combine all image sequences to one tensor
            ConcatItemsd(keys=sequences, name='tensor'),
            # create torch tensors
            ToTensord(keys=['tensor', 'label', 'prostate'])
        ])
    else:
        transform = Compose([
            # load data
            LoadImaged(keys=sequences + ['label', 'prostate'], ensure_channel_first=True),
            # ensure image is at least 224 x 224 x 32 in size
            SpatialPadd(keys=sequences + ['label', 'prostate'], spatial_size=(224, 224, 32)),
            # cut out crop around the center
            CenterSpatialCropd(keys=sequences + ['label', 'prostate'], roi_size=(224, 224, 32)),
            # replace given sequences with zeros
            DropSequenced(keys=sequences, drop=sequence_drop),
            # combine all image sequences to one tensor
            ConcatItemsd(keys=sequences, name='tensor'),
            # create torch tensors
            ToTensord(keys=['tensor', 'label', 'prostate'])
        ])

    if dataset == 'picai':
        data = get_picai_data(path, part=part, train_mode=train_mode)
    else:
        data = get_inhouse_data(path)

    return CacheDataset(data, transform)


def get_picai_data(path: str, train_mode: bool, part: float = 1.0):
    """
    creates a list of dicts with all file paths
    :param path: path to folder with data
    :param train_mode: positiv cases are N_POSITIV times in dataset
    :param part of the dataset to use
    :return: List of dicts
    """
    positive_data = []
    folders = [x for x in glob.glob(f"{path}/positive/*") if not isfile(x)]
    limit = int(len(folders) * part)
    for folder in folders:
        pathes = glob.glob(f"{folder}/*.mha")
        item = {path[-11:-8]: os.path.normpath(path) for path in pathes if path[-11:-8] in ['adc', 'hbv', 't2w']}
        item['label'] = [os.path.normpath(x) for x in glob.glob(f"{folder}/*lesion_roi.nii.gz")][0]
        item['prostate'] = [os.path.normpath(x) for x in glob.glob(f"{folder}/*prostate_roi.nii.gz")][0]
        item['case'] = 'positive'
        positive_data.append(item)

    if train_mode:
        positive_data = positive_data[:limit]
        positive_data *= N_POSITIVE

    negative_data = []
    folders = [x for x in glob.glob(f"{path}/negative/*") if not isfile(x)]
    limit = int(len(folders) * part)
    for folder in folders:
        pathes = glob.glob(f"{folder}/*.mha")
        item = {path[-11:-8]: os.path.normpath(path) for path in pathes if path[-11:-8] in ['adc', 'hbv', 't2w']}
        item['label'] = [os.path.normpath(x) for x in glob.glob(f"{folder}/*lesion_roi.nii.gz")][0]
        item['prostate'] = [os.path.normpath(x) for x in glob.glob(f"{folder}/*prostate_roi.nii.gz")][0]
        item['case'] = 'negative'
        negative_data.append(item)

    if train_mode:
        negative_data = negative_data[:limit]

    data = positive_data + negative_data
    return data


def get_inhouse_data(path: str):
    """
    creates a list of dicts with all file paths
    :param path: path to folder with data
    :return: List of dicts
    """
    data = []
    folders = [x for x in glob.glob(f"{path}/positive/*") if not isfile(x)]
    for folder in folders:
        item = {}
        item['adc'] = [os.path.normpath(x) for x in glob.glob(f"{folder}/ADC_roi.mha")][0]
        item['hbv'] = [os.path.normpath(x) for x in glob.glob(f"{folder}/DWI_b1500_roi.mha")][0]
        item['t2w'] = [os.path.normpath(x) for x in glob.glob(f"{folder}/T2_tra_roi.mha")][0]
        item['label'] = [os.path.normpath(x) for x in glob.glob(f"{folder}/lesion_roi.nii.gz")][0]
        item['prostate'] = [os.path.normpath(x) for x in glob.glob(f"{folder}/prostate_roi.nii.gz")][0]
        item['case'] = 'positive'
        item['folder'] = folder
        data.append(item)

    folders = [x for x in glob.glob(f"{path}/negative/*") if not isfile(x)]
    for folder in folders:
        item = {}
        item['adc'] = [os.path.normpath(x) for x in glob.glob(f"{folder}/ADC_roi.mha")][0]
        item['hbv'] = [os.path.normpath(x) for x in glob.glob(f"{folder}/DWI_b1500_roi.mha")][0]
        item['t2w'] = [os.path.normpath(x) for x in glob.glob(f"{folder}/T2_tra_roi.mha")][0]
        item['label'] = [os.path.normpath(x) for x in glob.glob(f"{folder}/lesion_roi.nii.gz")][0]
        item['prostate'] = [os.path.normpath(x) for x in glob.glob(f"{folder}/prostate_roi.nii.gz")][0]
        item['case'] = 'negative'
        item['folder'] = folder
        data.append(item)

    return data


if __name__ == '__main__':
    from pathlib import Path

    data = get_picai_data(f'{Path(__file__).parent.absolute()}/../data/split/train', True, 1.0)
    print(len(data))