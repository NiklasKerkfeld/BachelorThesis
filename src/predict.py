"""
script for prediction of lesions in bpMRI
"""
import argparse
import glob
import os
from os.path import isfile

import numpy as np
import torch
from termcolor import colored

from monai.networks.nets import BasicUNet
from monai.transforms import Compose, LoadImaged, SpatialPadd, CenterSpatialCropd, ConcatItemsd, ToTensord, \
    ScaleIntensityRangePercentilesd, NormalizeIntensityd, MapLabelValued, ResampleToMatchd, Spacingd, DivisiblePadd, \
    SaveImage
from torch import nn
from tqdm import tqdm

from src.CustomTransforms import CutRoiBySegmentationd
from src.postprocessing import Postprocessing

print(colored(f"This model is not for medical use! It's not even close to be good enough!\n\n"
              f"Please contact a doctor if you have a medical problem!\n\n\n", 'red'))

SPACING = (0.3, 0.3, 1.0)
MODEL = "models/basic_dicece_lr3_wd2_e30_dice.pth"

model = BasicUNet(spatial_dims=3, in_channels=3, features=(32, 32, 64, 128, 256, 32))
model.load_state_dict(torch.load(MODEL))

postprocessing = Postprocessing()

preprocess = Compose([
    LoadImaged(keys=['adc', 'hbv', 't2w', 'prostate'], ensure_channel_first=True, allow_missing_keys=True),

    # Clip intensity between 1. and 99. percentile and scale it between 0, 1
    ScaleIntensityRangePercentilesd(keys=['adc', 'hbv', 't2w'], lower=1, upper=99, b_min=0, b_max=1,
                                    allow_missing_keys=True),

    # Normalize intensity
    NormalizeIntensityd(keys=['adc', 'hbv', 't2w'], allow_missing_keys=True),

    MapLabelValued(keys=['prostate'], orig_labels=[0, 1, 2, 3, 4, 5, 6, 7],
                   target_labels=[0, 1, 1, 1, 1, 1, 1, 1], allow_missing_keys=True),

    # resample images to t2w so all images of patient have same size
    ResampleToMatchd(keys=['adc', 'hbv', 'prostate'], key_dst='t2w',
                     mode=['bilinear', 'bilinear', 'nearest'], allow_missing_keys=True),

    # ensure all images of one patient have same spacing
    Spacingd(keys=['adc', 'hbv', 't2w', 'prostate'], pixdim=SPACING,
             mode=("bilinear", "bilinear", "bilinear", "nearest"), allow_missing_keys=True),

    # cut out Region of prostate of the images
    CutRoiBySegmentationd(keys=['adc', 'hbv', 't2w', 'prostate'], segmentation_key='prostate',
                          edge=(16, 16, 8)),

    # make sure size is dividable by 32
    DivisiblePadd(keys=['t2w', 'adc', 'hbv', 'prostate'], k=(32, 32, 32), constant_values=0,
                  allow_missing_keys=True),

    # ensure image is at least 224 x 224 x 32 in size
    SpatialPadd(keys=['t2w', 'adc', 'hbv', 'prostate'], spatial_size=(224, 224, 32)),
    # cut out crop around the center
    CenterSpatialCropd(keys=['t2w', 'adc', 'hbv', 'prostate'], roi_size=(224, 224, 32)),
    # combine all image sequences to one tensor
    ConcatItemsd(keys=['t2w', 'adc', 'hbv'], name='tensor'),
    # create torch tensors
    ToTensord(keys=['tensor', 'prostate'])
])

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Using cuda:0 for prediction!")
else:
    device = torch.device("cpu")
    print("No cuda-device found! Using cpu for prediction!")

model.to(device)


def predict(item):
    """
    predicts lesions from a single patient
    :param item: dict with paths to t2w, adc and hbv images
    :return: prediction tensor, metadata
    """
    # preprocess image
    item = preprocess(item)

    # predict lesions
    pred = model(item['tensor'][None, :].to(device))
    pred = nn.functional.softmax(pred, dim=1)

    # detach
    pred = pred.cpu().detach()[:, 1]

    # postprocess prediction
    pred = postprocessing(pred, item['prostate'])

    # make pred a meta tensor
    prediction = torch.zeros_like(item['t2w'])
    prediction[:] = pred

    prediction.meta['filename_or_obj'] = prediction.meta['filename_or_obj'].split(os.sep)[-1][:-8]

    return prediction


def predict_and_save(item, path, extra_folder: bool = False, post_fix: str = 'prediction'):
    """
    predicts given item and saves results as NIFTI file
    :param item: Dict with image data
    :param path: path to folder to save the prediction in
    :param extra_folder: creates an extra 'prediction' folder to save the prediction in
    :param post_fix: added to the filename to create a name that can be identified as prediction
    :return:
    """
    # predict lesions
    pred = predict(item)

    # save prediction
    if extra_folder:
        path = f"{path}/prediction"
        os.makedirs(path, exist_ok=True)

    save = SaveImage(output_dir=f'{path}', dtype=np.float32, output_postfix=post_fix, output_ext='.nii.gz',
                     resample=False, print_log=False, separate_folder=False)

    save(pred)


def predict_one(path: str, extra_folder: bool = False, post_fix: str = 'prediction'):
    """
    predicts lesions from a single patient from given path
    :param path: path to folder with images
    :param extra_folder: creates extra folder for predictions if Ture
    :param post_fix: added to the filename to create a name that can be identified as prediction
    :return:
    """
    # create dict
    item = {
        'adc': glob.glob(f'{path}{os.sep}*_adc.mha')[0],
        'hbv': glob.glob(f'{path}{os.sep}*_hbv.mha')[0],
        't2w': glob.glob(f'{path}{os.sep}*_t2w.mha')[0],
        'prostate': glob.glob(f'{path}{os.sep}*_prostate.nii.gz')[0]
    }
    print('predicting...')
    predict_and_save(item, path, extra_folder=extra_folder, post_fix=post_fix)


def predict_folder(path: str, extra_folder: bool = False, post_fix: str = 'prediction'):
    """
    predicts lesions for all patient folders in given path
    :param path: path to folder with patient data
    :param extra_folder: creates extra folder for predictions if Ture
    :param post_fix: added to the filename to create a name that can be identified as prediction
    :return:
    """
    folder = [x for x in glob.glob(f"{path}/*") if not isfile(x)]

    data = []
    for folder in folder:
        try:
            item = (
                {
                    'adc': glob.glob(f'{folder}{os.sep}*_adc.mha')[0],
                    'hbv': glob.glob(f'{folder}{os.sep}*_hbv.mha')[0],
                    't2w': glob.glob(f'{folder}{os.sep}*_t2w.mha')[0],
                    'prostate': glob.glob(f'{folder}{os.sep}*_prostate.nii.gz')[0]
                },
                folder
            )
            data.append(item)
        except IndexError:
            pass

    print(f"found {len(data)} folder with correct image data!\n")

    for item, path in tqdm(data, desc='predicting images'):
        predict_and_save(item, path, extra_folder=extra_folder, post_fix=post_fix)


def get_args() -> argparse.Namespace:
    """
    get args for training the model
    :return: args object
    """
    parser = argparse.ArgumentParser(description='predict')
    parser.add_argument('--source', '-s', type=str, help='path to image data')
    parser.add_argument('--multi', '-m', type=bool, default=False,
                        help='if True prediction for all folders in source')
    parser.add_argument('--extra_folder', '-e', default=False, type=bool,
                        help='create extra folder for prediction?')
    parser.add_argument('--post_fix', '-p', default='prediction', type=str,
                        help='path to image data')

    return parser.parse_args()


if __name__ == '__main__':
    args = vars(get_args())

    if args['multi']:
        predict_folder(args['source'], extra_folder=args['extra_folder'], post_fix=args['post_fix'])
    else:
        predict_one(args['source'], extra_folder=args['extra_folder'], post_fix=args['post_fix'])