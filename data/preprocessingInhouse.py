"""
Preprocesses the Data and saves them
"""
import json
from pathlib import Path
from tqdm import tqdm
from pprint import pprint

import numpy as np

from monai.transforms import (
    Compose,
    LoadImaged,
    ScaleIntensityRangePercentilesd,
    NormalizeIntensityd,
    ResampleToMatchd,
    Spacingd,
    SaveImage,
    MapLabelValued,
    DivisiblePadd,
)

from src.CustomTransforms import CutRoiBySegmentationd

SPACING = (0.3, 0.3, 1.0)
CROPS = (128, 128, 32)
np.random.seed(42)

# define monai transformations
preprocess = Compose([
    # Load all imges and add a Channel at beginning (C, X, Y, Z)
    LoadImaged(keys=['adc', 'hbv', 't2w', 'lesion', 'prostate'], ensure_channel_first=True, allow_missing_keys=True),

    # Clip intensity between 1. and 99. percentile and scale it between 0, 1
    ScaleIntensityRangePercentilesd(keys=['adc', 'hbv', 't2w'], lower=1, upper=99, b_min=0, b_max=1,
                                    allow_missing_keys=True),

    # Normalize intensity
    NormalizeIntensityd(keys=['adc', 'hbv', 't2w'], allow_missing_keys=True),

    # ensure label and prostate only have 0 and 1 values
    MapLabelValued(keys=['lesion'], orig_labels=[0, 1, 2, 3, 4, 5, 6, 7],
                   target_labels=[0, 1, 1, 1, 1, 1, 1, 1], allow_missing_keys=True),

    MapLabelValued(keys=['prostate'], orig_labels=[0, 1, 2, 3, 4, 5, 6, 7],
                   target_labels=[0, 1, 1, 1, 1, 1, 1, 1], allow_missing_keys=True),

    # resample images to t2w so all images of patient have same size
    ResampleToMatchd(keys=['adc', 'hbv', 'lesion', 'prostate'], key_dst='t2w',
                     mode=['bilinear', 'bilinear', 'nearest', 'nearest'], allow_missing_keys=True),

    # ensure all images of one patient have same spacing
    Spacingd(keys=['adc', 'hbv', 't2w', 'lesion', 'prostate'], pixdim=SPACING,
             mode=("bilinear", "bilinear", "bilinear", "nearest", "nearest"), allow_missing_keys=True),

    # cut out Region of prostate of the images
    CutRoiBySegmentationd(keys=['adc', 'hbv', 't2w', 'lesion', 'prostate'], segmentation_key='prostate',
                          edge=(CROPS[0] // 8, CROPS[1] // 8, CROPS[2] // 4)),

    # make sure size is dividable by 32
    DivisiblePadd(keys=['t2w', 'adc', 'hbv', 'lesion', 'prostate'], k=(32, 32, 32), constant_values=0,
                  allow_missing_keys=True),
])


def preprocessing(patient: str, category: str, positive: bool = False):
    """
    preprocessing mri images with preprocessing transformation adding lesion mask with zeros if patient is negative
    :param patient: folder name of the patient
    :param category: category (train, valid, test) for saving in
    :param positive: true if patient is positive casePCa > 1
    :return:
    """
    item = {
        'adc': f'{Path(__file__).parent.absolute()}/../../../Data/InhouseDataset/{patient}/ADC.mha',
        'hbv': f'{Path(__file__).parent.absolute()}/../../../Data/InhouseDataset/{patient}/DWI_b1500.mha',
        't2w': f'{Path(__file__).parent.absolute()}/../../../Data/InhouseDataset/{patient}/T2_tra.mha',
        'prostate': f'{Path(__file__).parent.absolute()}/../../../Data/InhouseDataset/{patient}/Zones.nii.gz',
        'lesion': f'{Path(__file__).parent.absolute()}/../../../Data/InhouseDataset/{patient}/lesion.nii.gz'
    }

    # preprocess images
    item = preprocess(item)

    # save image sequences
    part = 'positive' if positive else 'negative'
    save = SaveImage(output_dir=f'{Path(__file__).parent.absolute()}/Inhouse_split/{category}/{part}/{patient}',
                     dtype=np.float32, output_postfix='roi', output_ext='.mha', resample=False, print_log=False,
                     separate_folder=False)

    for key in ['t2w', 'hbv', 'adc']:
        save(item[key])

    # save segmentations
    save = SaveImage(output_dir=f'{Path(__file__).parent.absolute()}/Inhouse_split/{category}/{part}/{patient}',
                     output_postfix='roi', output_ext='.nii.gz', resample=False, print_log=False,
                     separate_folder=False)

    for key in ['lesion', 'prostate']:
        save(item[key])


def main():
    """
    preprocesses the data and saves the results according to their train, valid, test split
    :return:
    """
    with open(f'{Path(__file__).parent.absolute()}/../../../Data/InhouseDataset/split_new.json', ) as f:
        split = json.load(f)

    pprint(split)

    positive = split['positive']

    for patient in tqdm(positive['train'], desc='preprocessing positive training'):
        preprocessing(patient, 'train', True)

    for patient in tqdm(positive['test'], desc='preprocessing positive test'):
        preprocessing(patient, 'test', True)

    for patient in tqdm(positive['valid'], desc='preprocessing positive valid'):
        preprocessing(patient, 'valid', True)

    negatives = split['negative']

    for patient in tqdm(negatives['train'], desc='preprocessing negative training'):
        preprocessing(patient, 'train', False)

    for patient in tqdm(negatives['test'], desc='preprocessing negatives test'):
        preprocessing(patient, 'test', False)

    for patient in tqdm(negatives['valid'], desc='preprocessing negatives valid'):
        preprocessing(patient, 'valid', False)


if __name__ == '__main__':
    main()
