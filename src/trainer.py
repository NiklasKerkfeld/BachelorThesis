"""
class for training-process
"""

import os
import time
from pathlib import Path
from pprint import pprint
from typing import Dict
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import torch.multiprocessing

from monai.data import Dataset, DataLoader

from picai_eval import evaluate
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import dice

from src.postprocessing import Postprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

params = {
    'batch_size': 16,
    'lr': 1e-3,
    'weight_decay': 1e-5,
    'gamma': 0.98,
    'ce_weights': (1.0, 1.0),
    'save_every': 5,
    'warmup_epochs': 10,
    'threshold': 0.8,
    'theta_1': 0.4,
    'theta_2': 0.6,
    'worker': 8,
    'part': 1.0
}


class Trainer:
    """
    trainer for training model on biMRI Images of Prostate
    """
    def __init__(self, model: nn.Module, loss_function: nn.Module, writer,
                 train_dataset: Dataset, valid_dataset: Dataset,
                 config: dict, device: int = 0, silence: bool = False):
        """
        trainer for training model on biMRI Images of Prostate
        :param model: model to train
        :param name: name of the model for saving
        :param train_dataset: Dataset for training
        :param valid_dataset: Dateset for validation
        :param config: dict with trainings parameter
        :param silence: does not show progress if true
        """
        self.config = config
        params.update(config)
        self.ce_weights = torch.tensor(params['ce_weights'])

        # init variables
        self.epoch = 0
        self.step = 0
        self.time = 0

        self.best_dice = 0.0
        self.dice_step = 0
        self.best_ap = 0
        self.ap_step = 0
        self.results = {'dice': {'loss': None, 'dice': None, 'dice_median': None, 'dice_std': None, 'iou': None,
                                 'cross_entropy': None, 'accuracy': None,  'volume': None, 'AP': None,
                                 'AUROC': None, 'score': None, 'step': None},
                        'ap':   {'loss': None, 'dice': None, 'dice_median': None, 'dice_std': None, 'iou': None,
                                 'cross_entropy': None, 'accuracy': None, 'volume': None, 'AP': None,
                                 'AUROC': None, 'score': None, 'step': None}}

        self.name = params['name']
        self.save_every = params['save_every']
        self.threshold = params['threshold']
        self.silence = silence
        self.writer = writer

        # check for cuda
        if torch.cuda.is_available():
            if not silence:
                print(f"using cuda:{device}!")
            self.device = torch.device(f'cuda:{device}')
        else:
            if not silence:
                print(f"no cuda found using cpu!")
            self.device = torch.device('cpu')

        # init model, optimizer and loss function
        self.model = model
        self.optimizer = AdamW(self.model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

        # defining lr schedule with warmup and exponential lr decay
        lr_schedule = lambda epoch: min((epoch + 1) / params['warmup_epochs'], 1) * \
                                    min(params['gamma'] ** ((epoch + 1) - params['warmup_epochs']), 1)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_schedule)

        self.loss_fn = loss_function

        # setup dataloader
        self.train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, drop_last=True,
                                       num_workers=params['worker'])
        self.valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, drop_last=False,
                                       num_workers=params['worker'])

        self.postprocessing = Postprocessing(theta_1=params['theta_1'], theta_2=params['theta_2'])

        # images for logging in tensorboard
        example = valid_dataset[0]
        self.valid_target = example['label']
        self.valid_image = example['tensor']
        self.valid_prostate = example['prostate']
        self.valid_t2w = example['t2w']
        self.valid_adc = example['adc']
        self.valid_hbv = example['hbv']

        self.log_dim = torch.argmax(torch.sum(self.valid_target, dim=(0, 1, 2))).item()

        # log images
        self.writer.add_image("image/t2w", self.scale_image(self.valid_t2w[:, :, :, self.log_dim]), global_step=0)
        self.writer.add_image("image/adc", self.scale_image(self.valid_adc[:, :, :, self.log_dim]), global_step=0)
        self.writer.add_image("image/hbv", self.scale_image(self.valid_hbv[:, :, :, self.log_dim]), global_step=0)
        self.writer.add_image("image/target", self.valid_target[:, :, :, self.log_dim].float(), global_step=0)

        self.writer.add_text("params/batch_size", str(params['batch_size']), global_step=0)
        self.writer.add_text("params/learning_rate", str(params['lr']), global_step=0)
        self.writer.add_text("params/weight_decay", str(params['weight_decay']), global_step=0)
        self.writer.add_text("params/gamma", str(params['gamma']), global_step=0)
        self.writer.add_text("params/ce_weights", str(params['ce_weights']), global_step=0)
        self.writer.flush()

    def train(self, epochs: int):
        """
        trains model for given number of epochs
        :param epochs: number of epochs to train
        :return: None
        """
        print('start training with this params:')
        pprint(params)
        print()

        self.writer.add_text("params/epochs", str(epochs), global_step=0)

        self.model.to(self.device)
        self.loss_fn.to(self.device)

        self.time = time.time()
        for e in range(1, epochs + 1):
            self.epoch = e
            self._train_epoch()

        return self.results

    def _train_epoch(self):
        """
        trains model for one epoch
        :return: None
        """
        self.model.train()

        for batch in tqdm(self.train_loader, desc=f'epoch {self.epoch}', disable=self.silence):
            x = batch['tensor'].to(self.device)
            y = batch['label'].to(self.device)

            # do training step
            self.optimizer.zero_grad()
            pred = self.model(x)
            pred = nn.functional.softmax(pred, dim=1)
            loss = self.loss_fn(pred, y)

            loss.backward()
            self.optimizer.step()

            ce_losses = F.cross_entropy(pred, y.long().squeeze())

            # logging
            self.writer.add_scalar(f'training/loss', loss.detach().cpu().item(), global_step=self.step)
            self.writer.add_scalar(f'training/cross entropy', ce_losses, global_step=self.step)

            self.writer.add_scalar('process/epoch', self.epoch, global_step=self.step)
            self.writer.add_scalar(f'process/learning rate', self.optimizer.param_groups[0]["lr"],
                                   global_step=self.step)
            self.writer.add_scalar(f'process/weight decay', self.optimizer.param_groups[0]["weight_decay"],
                                   global_step=self.step)
            self.writer.flush()

            # delete tensors on cuda-device
            del x, y, pred, loss
            torch.cuda.empty_cache()

            # update step and valid
            self.step += 1

        self.valid()

        # save model
        if self.epoch % self.save_every == 0:
            self.save(self.name)

        # step scheduler
        self.scheduler.step()

    def valid(self):
        """
        validates model on validation data
        :return: None
        """
        self.model.eval()

        results = self._validate(self.valid_loader)

        # logging
        self.log('valid', results)

        # predict sample image
        pred = self.model(self.valid_image[None, :].to(self.device)).cpu().detach()
        prob = nn.functional.softmax(pred, dim=1)[0, 1]

        self.writer.add_image("image/prob", prob[None, :, :, self.log_dim].float(), global_step=self.step)
        detection_map, _ = self.postprocessing(prob, self.valid_prostate)
        self.writer.add_image("image/pred", detection_map[None, :, :, self.log_dim].float(), global_step=self.step)
        self.writer.flush()

        # early stopping
        if self.best_dice < results['dice']:
            self.best_dice = results['dice']
            self.dice_step = self.step
            self.results['dice'] = results
            self.save(f"{self.name}_dice")

        if self.best_ap < results['AP']:
            self.best_ap = results['AP']
            self.ap_step = self.step
            self.results['ap'] = results
            self.save(f"{self.name}_ap")

        del results
        torch.cuda.empty_cache()

        self.model.train()

    def _validate(self, dataloader):
        """
        validates with given dataloader returns different metrics
        :param dataloader: valid- or test-dataloader
        :return: loss, dice, iou, cross_entropy, accuracy, volume, average precision
        """
        n = len(dataloader)
        j = 0
        losses = torch.zeros(n)
        dices = torch.zeros(30)
        ious = torch.zeros(30)
        ce_losses = torch.zeros(n)
        volumes_pred = torch.zeros(n)
        volumes_truth = torch.zeros(n)
        accuracy = torch.zeros(n)
        predictions = []
        targets = []

        for i, batch in tqdm(enumerate(dataloader), total=n, desc='testing', disable=self.silence):
            x = batch['tensor'].to(self.device)
            y = batch['label'].to(self.device)
            prostate = batch['prostate'][:, 0]

            # predict
            prob = self.model(x)
            prob = nn.functional.softmax(prob, dim=1)

            # metrics
            losses[i] = self.loss_fn(prob, y).cpu().detach()
            ce_losses[i] = F.cross_entropy(prob, y[:, 0].long()).cpu().detach()

            # detach
            prob = prob.cpu().detach()[:, 1]
            y = y.cpu().detach()[:, 0]

            detection_map, _ = self.postprocessing(prob, prostate=prostate)

            # append to list
            predictions.append(detection_map.numpy())
            targets.append(y[0].numpy())

            discrete_lesions = detection_map > self.threshold
            volumes_pred[i] = (torch.sum(discrete_lesions) - torch.sum(y))
            volumes_truth[i] = torch.sum(y)

            # calc dice and iou for positive cases
            if batch['case'][0] == 'positive':
                dices[j] = 1 - dice(discrete_lesions.flatten(), y.flatten())
                ious[j] = dices[j] / (2 - dices[j])
                j += 1

            # accuracy
            accuracy[i] = accuracy_score(y.flatten(), discrete_lesions.flatten())

            del x, y, prostate, prob, detection_map, discrete_lesions
            torch.cuda.empty_cache()

        # calc means
        loss = torch.mean(losses).item()
        cross_entropy = torch.mean(ce_losses).item()
        accuracy = torch.mean(accuracy).item()

        # get dice and iou and reset
        dice_value = torch.mean(dices[:j]).item()
        dice_median = torch.median(dices[:j]).item()
        dice_std = torch.std(dices[:j]).item()
        iou_value = torch.mean(ious[:j]).item()

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

        return {'loss': loss, 'dice': dice_value, 'dice_median': dice_median, 'dice_std': dice_std, 'iou': iou_value,
                'cross_entropy': cross_entropy, 'accuracy': accuracy, 'volume': volume, 'AP': ap, 'AUROC': auroc,
                'score': score, 'step': self.step, 'time': self.convert_time(time.time() - self.time)}

    def log(self, category: str, results: Dict[str, float]):
        """
        log the given data on tensorboard
        :param category: category to log in ('valid' or 'test')
        :param results: dict with the results from validate
        :return:
        """
        self.writer.add_scalar(f'{category}/loss', results['loss'], global_step=results['step'])
        self.writer.add_scalar(f'{category}/dice', results['dice'], global_step=results['step'])
        self.writer.add_scalar(f'{category}/dice(median)', results['dice_median'], global_step=results['step'])
        self.writer.add_scalar(f'{category}/dice(std)', results['dice_std'], global_step=results['step'])
        self.writer.add_scalar(f'{category}/iou', results['iou'], global_step=results['step'])
        self.writer.add_scalar(f'{category}/cross_entropy', results['cross_entropy'], global_step=results['step'])
        self.writer.add_scalar(f'{category}/accuracy', results['accuracy'], global_step=results['step'])
        self.writer.add_scalar(f'{category}/volume', results['volume'], global_step=results['step'])
        self.writer.add_scalar(f'{category}/AP', results['AP'], global_step=results['step'])
        self.writer.add_scalar(f'{category}/AUROC', results['AUROC'], global_step=results['step'])
        self.writer.add_scalar(f'{category}/score', results['score'], global_step=results['step'])
        self.writer.flush()

    def save(self, name: str):
        """
        saves model
        :param name: name for the save file
        :return: None
        """
        os.makedirs(f'{Path(__file__).parent.absolute()}/../models/', exist_ok=True)
        torch.save(
            self.model.state_dict(),
            os.path.join(f'{Path(__file__).parent.absolute()}/../models/{name}.pth'),
        )

    def load(self, name: str):
        """
        loads model
        :param name: name of the model to load
        :return: None
        """
        self.model.load_state_dict(torch.load(f'{Path(__file__).parent.absolute()}/../models/{name}.pth'))
        print(f"model form {Path(__file__).parent.absolute()}/../models/{name}.pth loaded!")

    @staticmethod
    def scale_image(image: torch.Tensor) -> torch.Tensor:
        """
        scales image between 0 and 1
        :param image: image to be scaled
        :return: scaled image
        """
        image = image - image.min()
        return image / image.max()

    @staticmethod
    def convert_time(milsec):
        sec = int(milsec)
        sec = sec % 86400
        hour = sec // 3600
        sec %= 3600
        min = sec // 60
        sec %= 60
        return "%02d:%02d:%02d" % (hour, min, sec)
