import glob
import os
from os.path import isfile

import torch
from monai.transforms import LoadImaged, ConcatItemsd, ToTensord, Compose
from tqdm import tqdm

CROP = 128, 128, 32

def get_data(folder):
    item = {}
    sequences = ['t2w', 'adc', 'hbv', 'prostate', 'label']
    files = ['t2w_roi.mha', 'adc_roi.mha', 'hbv_roi.mha', 'prostate_roi.nii.gz', 'lesion_roi.nii.gz']
    for seq, file in zip(sequences, files):
        # print(f"{folder}/*{file}")
        item[seq] = [os.path.normpath(x) for x in glob.glob(f"{folder}/*{file}")][0]

    return item


def testData():
    transfromation = Compose([LoadImaged(keys=['t2w', 'adc', 'hbv', 'label'], ensure_channel_first=True),
                              ConcatItemsd(keys=['t2w', 'adc', 'hbv'], name='tensor'),
                              ToTensord(keys=['tensor', 'label'])
                              ])

    folders = [x for x in glob.glob('preprocessed/*') if not isfile(x)]

    # iterate over all given patients
    t = tqdm(folders, desc='testing')
    for folder in t:
        # get dict with paths to the specific images
        item = get_data(folder)
        file = item['t2w']

        # preprocess images
        item = transfromation(item)

        # random choice of crop center by given probabilities
        label = item['label'][0]
        label[:CROP[0] // 2 + 1] = 3
        label[-CROP[0] // 2 + 1:] = 3

        label[:, :CROP[1] // 2 + 1] = 3
        label[:, -CROP[1] // 2 + 1:] = 3

        label[:, :, :CROP[2] // 2 + 1] = 3
        label[:, :, -CROP[2] // 2 + 1:] = 3

        # random choice center with crop class
        count_background = torch.count_nonzero((label == 0)).item()
        count_lesion = torch.count_nonzero((label == 1)).item()

        assert count_background > 0, f"got no background pixel to select for crop-center"
        assert count_lesion > 0, f"got no lesion pixel to select for crop-center"
        _, x, y, z = item['t2w'].shape
        assert x % 32 == 0 and y % 32 == 0 and z % 32 == 0, \
            f"shape of roi is not dividable 32 got {item['t2w'].shape} at image {file}"


if __name__ == '__main__':
    testData()
