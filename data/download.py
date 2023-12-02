"""
script for downloading dataset with annotations
"""

import glob
import json
import os
from os.path import isfile
from typing import List, Tuple

import requests
from tqdm import tqdm
import urllib.request
from pathlib import Path
import zipfile


def extract_zip_in_folder(file_list: List[str]):
    """
    list with paths of zip-files to extract
    :param file_list:
    :return:
    """
    for file in tqdm(file_list, desc='extracting'):
        with zipfile.ZipFile(file, 'r') as zipper:
            folder = Path(f'{Path(__file__).parent.absolute()}/full/')
            folder.mkdir(exist_ok=True)
            zipper.extractall(path=folder)


def download(patient_id: str, filename: str, lesion: bool = True, prostate: bool = True):
    """
    downloads annotations
    :param patient_id: study the patient is in
    :param filename: filename of the patient
    :param lesion: download lesion annotation if True
    :param prostate: download prostate annotations if True
    :return: None
    """

    if lesion:
        url = f"https://github.com/DIAGNijmegen/picai_labels/raw/main/csPCa_lesion_delineations/human_expert/resampled/{filename}.nii.gz"
        r = requests.get(url)

        with open(os.path.join(Path(__file__).parent.absolute(), 'full', patient_id, f"{filename}_lesion.nii.gz"), 'wb') as f:
            f.write(r.content)

    if prostate:
        url = f"https://github.com/DIAGNijmegen/picai_labels/raw/main/anatomical_delineations/whole_gland/AI/Bosma22b/{filename}.nii.gz"
        r = requests.get(url)

        with open(os.path.join(Path(__file__).parent.absolute(), 'full', patient_id, f"{filename}_prostate.nii.gz"), 'wb') as f:
            f.write(r.content)


def download_annotations():
    """
    downloads the annotations from GitHub
    :return: None
    """
    files = [x.split(os.sep)[-1][:-8] for x in glob.glob(f'{Path(__file__).parent.absolute()}/full/*/*_t2w.mha')]
    files = [(x.split('_')[0], x) for x in files]

    t = tqdm(files, desc='downloading')
    for pat, filename in t:
        download(pat, filename)
        t.set_description(desc=f'downloading: {pat}')


def delete_segmentation():
    """
    deletes all .gz files from data
    :return: None
    """
    files = [x for x in glob.glob(f"{Path(__file__).parent.absolute()}/full/**/*.gz") if isfile(x)]

    for file in files:
        os.remove(file)


def load_and_store(file_list: List[Tuple[str, str]]):
    """
    downloads PIC-AI zip files
    :param file_list: list of files
    :return:
    """
    for online_path, local_path in tqdm(file_list, desc='downloading'):
        urllib.request.urlretrieve(online_path, local_path)


def main():
    """
    downloading PI-CAI dataset with all image and annotations for lesion and prostate
    :return:
    """
    data_location = 'https://zenodo.org/record/6624726'

    files = [f"picai_public_images_fold{i}.zip" for i in range(0,5)]
    picai_url = [(f'{data_location}/files/{file}', f'{Path(__file__).parent.absolute()}/full/{file}') for file in files]

    print("downloading... this will take a while. Please be patient.")
    Path(f"{Path(__file__).parent.absolute()}/full/").mkdir(exist_ok=True)
    load_and_store(picai_url)
    print("download ok")

    print("Extracting...")
    local_zips = list(purl[1] for purl in picai_url)
    extract_zip_in_folder(local_zips)

    print("deleting zip files...")
    for file in local_zips:
        os.remove(file)

    print("download annotations...")
    download_annotations()


if __name__ == '__main__':
    main()