import csv
import os
from pathlib import Path
from pprint import pprint
from typing import List

import monai.data
import numpy as np
from monai.transforms import SaveImage
from tqdm import tqdm

from monai.networks.nets import BasicUNet, SwinUNETR
from monai.data import DataLoader

import torch
from torch import nn

from picai_eval import evaluate
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import dice

import matplotlib.pyplot as plt

from src.lesion_dice import analyse_lesions
from src.ResNetUNet import Model
from src.postprocessing import Postprocessing
from src.train import get_dataset

postprocessing = Postprocessing()


def save_image(data: torch.Tensor, path: str, name: str, reference: monai.data.MetaTensor, postfix):
    save = SaveImage(output_dir=path, dtype=np.float32, output_postfix=postfix,
                     output_ext='.nii.gz', resample=False, print_log=False, separate_folder=False)

    image = torch.zeros_like(reference)
    image[:] = data

    image.meta['filename_or_obj'] = name

    save(image)


def _validate(model, dataloader, device, save_path: str, threshold: float = 0.6):
    """
    validates with given dataloader returns different metrics
    :param dataloader: valid- or test-dataloader
    :return: loss, dice, iou, cross_entropy, accuracy, volume, average precision
    """
    model.to(device)

    logs = []

    n = len(dataloader)
    j = 0  # counter for positive cases
    dices = torch.zeros(n)
    ious = torch.zeros(n)
    volumes_pred = torch.zeros(n)
    volumes_truth = torch.zeros(n)
    accuracy = torch.zeros(n)
    probabilities = []
    predictions = []
    targets = []

    for i, batch in tqdm(enumerate(dataloader), total=n, desc='testing'):
        name = batch['t2w'].meta['filename_or_obj'][0].split(os.sep)[-1][:-12]
        log = {'name': name, 'case': batch['case'][0]}

        # predict
        x = batch['tensor'].to(device)
        prob = model(x)
        prob = nn.functional.softmax(prob, dim=1)
        prob = prob.detach().cpu()[:, 1]

        # get label and prostate mask
        prostate = batch['prostate'][:, 0]
        y = batch['label'][:, 0]

        # postprocess
        detection_map, lesions = postprocessing(prob, prostate=prostate)

        # append to lists
        probabilities.append(prob[0].numpy())
        predictions.append(detection_map.numpy())
        targets.append(y[0].numpy())

        # make discrete
        discrete_lesions = detection_map > threshold
        lesions *= discrete_lesions

        # calc number of predicted lesions
        num_lesions = len(np.unique(lesions)) - 1
        log['num_pred_lesions'] = num_lesions

        # calc volume
        volumes_pred[i] = torch.sum(discrete_lesions)
        volumes_truth[i] = torch.sum(y)
        log['pred_volume'] = volumes_pred[i].item()
        log['truth_volume'] = volumes_truth[i].item()

        # accuracy
        accuracy[i] = accuracy_score(y.flatten(), discrete_lesions.flatten())
        log['accuracy'] = accuracy[i].item()

        # calc dice and iou for positive cases
        if batch['case'][0] == 'positive' and y.sum() != 0:
            dices[j] = 1 - dice(discrete_lesions.flatten(), y.flatten())
            ious[j] = dices[j] / (2 - dices[j])
            log['dice'] = dices[j].item()
            log['iou'] = ious[j].item()
            j += 1

        else:
            log['dice'] = None
            log['iou'] = None

        # lesion based statistics
        stat = analyse_lesions(discrete_lesions.float(), y[0])
        log['num_pred_lesions'] = stat[1]
        log['num_truth_lesions'] = stat[2]
        log['found_lesions'] = stat[3]
        log['pred_with_lesion'] = stat[4]
        log['false_positives'] = stat[5]
        log['false_negatives'] = stat[6]

        # save predictions as NIFTI
        save_image(prob, save_path, name, batch['t2w'][0], 'probability_map')
        save_image(detection_map, save_path, name, batch['t2w'][0], 'detection_map')
        save_image(lesions * discrete_lesions, save_path, name, batch['t2w'][0], 'predicted_lesions')

        logs.append(log)

        del x, y, prostate, prob, detection_map, discrete_lesions, lesions, stat
        torch.cuda.empty_cache()

    # calc means
    acc_mean = torch.mean(accuracy).item()
    dice_mean = torch.mean(dices[:j]).item()
    dice_median = torch.median(dices[:j]).item()
    dice_std = torch.std(dices[:j]).item()
    iou_mean = torch.mean(ious[:j]).item()

    # calc volume differences
    volume = (torch.mean(volumes_pred) / (torch.mean(volumes_truth) + 1e-9)).item()

    # PICAI metrics
    metrics = evaluate(
        y_det=predictions,
        y_true=targets,
    )

    ap = metrics.AP
    auroc = metrics.auroc
    score = (ap + auroc) / 2

    # plot recision-Recall (PR) curve
    disp = PrecisionRecallDisplay(precision=metrics.precision, recall=metrics.recall, average_precision=ap)
    disp.plot()
    plt.savefig(f'{save_path}/PrecisionRecallDisplay.png')

    # plot Receiver Operating Characteristic (ROC) curve
    disp = RocCurveDisplay(fpr=metrics.case_FPR, tpr=metrics.case_TPR, roc_auc=auroc)
    disp.plot()
    plt.savefig(f'{save_path}/RocCurveDisplay.png')

    # plot Free-Response Receiver Operating Characteristic (FROC) curve
    f, ax = plt.subplots()
    disp = RocCurveDisplay(fpr=metrics.lesion_FPR, tpr=metrics.lesion_TPR)
    disp.plot(ax=ax)
    ax.set_xlim(0.001, 5.0)
    ax.set_xscale('log')
    ax.set_xlabel("False positives per case")
    ax.set_ylabel("Sensitivity")
    plt.savefig(f'{save_path}/FROC.png')

    header = ['name', 'case', 'dice', 'iou', 'pred_volume', 'truth_volume', 'accuracy', 'num_pred_lesions',
              'num_truth_lesions', 'found_lesions', 'pred_with_lesion', 'false_positives', 'false_negatives']
    with open(f'{save_path}/ImageBasedResults.csv', 'w', encoding='UTF8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(logs)

    return {'dice': dice_mean, 'dice_median': dice_median, 'dice_std': dice_std, 'iou': iou_mean,
            'accuracy': acc_mean, 'volume': volume, 'AP': ap, 'AUROC': auroc, 'score': score}


def main(model_name: str, drop_sequences: List[str] = []):
    # create a path for saving prediction results
    save_path = f"{Path(__file__).parent.absolute()}/../logs/testingresults2/{model_name}"
    # update save_path if sequences are dropped
    if drop_sequences:
        save_path += f"_without_{'_'.join(drop_sequences)}"

    print(f"{save_path=}")

    # init model and load weights from savepoint
    # model = BasicUNet(spatial_dims=3, in_channels=3, features=(32, 32, 64, 128, 256, 32))
    # model = src(3, 2)
    model = SwinUNETR(img_size=(128, 128, 32), in_channels=3, out_channels=2, depths=(2, 2, 2, 2),
                      num_heads=(3, 6, 12, 24), feature_size=24)
    model.load_state_dict(torch.load(f"models/{model_name}.pth"))
    print(f"model (models/{model_name}.pth) loaded!")

    # init test dataset
    test_dataset = get_dataset(f'{Path(__file__).parent.absolute()}/../data/split/test',
                               sequences=['t2w', 'adc', 'hbv'], sequence_drop=drop_sequences, train_mode=False)

    print(f"{len(test_dataset)=}\n")

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=1)

    # validate test data
    results = _validate(model, dataloader=test_loader, device=torch.device(f'cuda:0'), save_path=save_path)

    # print results
    print("\nresults:")
    pprint(results)

    with open(f'{save_path}/results.txt', 'w', encoding='UTF8') as f:
        f.write(f"Results for {model_name}:\n")
        for key, value in results.items():
            f.write(f"{key} = {value}\n")


if __name__ == '__main__':
    main("transformer_dice_lr2_wd2_dice")
