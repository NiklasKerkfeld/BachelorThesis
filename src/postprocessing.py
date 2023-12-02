"""
script for postprocessing the model prediction
"""

import numpy as np
import torch
from monai.transforms import RemoveSmallObjects, Compose, AsDiscrete, FillHoles
from report_guided_annotation import extract_lesion_candidates


class Postprocessing:
    def __init__(self, theta_1: float = 0.4, theta_2: float = 0.6, min_size: int = 250):
        """
        :param theta_1: threshold for removing small objects and filling holes
        :param theta_2: threshold for PI-CAI extraction of lesion candidates
        :param min_size: minimal size of lesions
        """
        self.theta_1 = theta_1
        self.theta_2 = theta_2
        self.min_size = min_size

    def __call__(self, prediction: torch.Tensor, prostate: torch.Tensor):
        """
        postprocessing for a given probability map
        :param prediction: probability map for lesions
        :param prostate: segmentation mask for prostate
        :return:
        """
        post = Compose([
            AsDiscrete(threshold=self.theta_1),
            RemoveSmallObjects(min_size=self.min_size),
            FillHoles(applied_labels=1)
        ])

        mask = post(prediction)

        prediction = prediction * mask * prostate

        detection_map, lesions = self.extract_lesions(prediction[0])

        return detection_map, lesions

    def extract_lesions(self, prediction: np.array):
        """
        extracts lesion with method from PI-CAI
        :param prediction: heatmap of prediction
        :return: map ove lesions (value is probability)
        """
        all_hard_blobs, confidences, blobs_index = extract_lesion_candidates(prediction.numpy(), threshold=self.theta_2)

        confidenc_dict = {x: y for x, y in confidences}
        confidenc_dict.update({0: 0.0})
        replace = np.vectorize(lambda x: confidenc_dict[x])

        return torch.tensor(replace(blobs_index)), torch.tensor(blobs_index)
