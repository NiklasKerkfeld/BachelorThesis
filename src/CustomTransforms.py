"""
multiple monai-transforms for different purposes in preprocessing and augmentation
"""

from copy import deepcopy
from typing import Tuple, Mapping, Hashable, Union, Sequence, Dict, List

import numpy as np
import torch
from monai.config import KeysCollection
from monai.transforms import MapTransform, SpatialCrop


class DropSequenced(MapTransform):
    def __init__(self, keys: KeysCollection, drop: Sequence[str]):
        """
        randomly sets input sequence to zero
        :param drop: list of bool that indicate what sequence to drop
        """
        super().__init__(keys)
        self.drop = drop

    def __call__(self, data: Mapping[Hashable, torch.Tensor]):
        """
        crops data with roi around the given segmentation
        :param data: key: images dict to preprocess
        :return: key: images dict
        """
        # create d as dict of data
        d = dict(data)
        # crop the data
        for key in self.key_iterator(d):
            if key in self.drop:
                d[key] = torch.zeros_like(d[key])

        return d


class RandomDropSequenced(MapTransform):
    def __init__(self, keys: KeysCollection, probability: Sequence[float]):
        """
        randomly sets input sequence to zero
        :param probability: probability for ever sequence
        """
        super().__init__(keys)
        self.probability = probability

    def __call__(self, data: Mapping[Hashable, torch.Tensor]):
        """
        crops data with roi around the given segmentation
        :param data: key: images dict to preprocess
        :return: key: images dict
        """
        # create d as dict of data
        d = dict(data)
        # crop the data
        for i, key in enumerate(self.key_iterator(d)):
            if torch.rand(1).item() < self.probability[i]:
                d[key] = torch.zeros_like(d[key])

        return d


class CutRoiBySegmentationd(MapTransform):
    def __init__(self, keys: KeysCollection, segmentation_key: str,
                 edge: Tuple[int, int, int] = (0, 0, 0), kdiv: int = 32):
        """
        crops data with roi around the given segmentation
        :param keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        :param segmentation_key: key of the segmentation the roi should be around
        :param edge: min number of pixels between segmentation and roi (default 0)
        :param kdiv: every dim of roi is dividable by kdiv (default 32)
        """
        super().__init__(keys)
        self.segmentation_key = segmentation_key
        self.edge = edge
        self.kdiv = kdiv

    def __call__(self, data: Mapping[Hashable, torch.Tensor]):
        """
        crops data with roi around the given segmentation
        :param data: key: images dict to preprocess
        :return: key: images dict
        """
        # create d as dict of data
        d = dict(data)

        # calculate roi and create cropper
        roi_center, roi_size = get_roi(data[self.segmentation_key], self.edge, self.kdiv)
        cropper = SpatialCrop(roi_center=roi_center, roi_size=roi_size)

        # crop the data
        for key in self.key_iterator(d):
            d[key] = cropper(d[key])

        return d


class SystematicCropSegmentd(MapTransform):
    def __init__(self, keys: KeysCollection, segment_key: str, spatial_size: Union[Sequence[int], int],
                 mode: str = "arange",
                 crops: Tuple[int, int, int] = (128, 128, 96),
                 space: Tuple[int, int, int] = (10, 10, 5),
                 nr_per_dim: Tuple[int, int, int] = (5, 5, 3),
                 edges: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]] = ((0, 0), (0, 0), (0, 0)),
                 allow_missing_keys: bool = False
                 ) -> None:
        """
        systematically crops the bounding box given segment
        :param keys: keys to crop
        :param segment_key: segment where the crops should center in
        :param spatial_size: size of the crops
        :param space: space between the crops
        :param edges: extra space for the bounding box
        allow_missing_keys: don't raise exception if key is missing.
        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.segment_key = segment_key
        self.spatial_size = spatial_size
        self.mode = mode
        self.crops = crops
        self.space = space
        self.nr_per_dim = nr_per_dim
        self.edges = edges

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, List[torch.Tensor]]:
        """
        systematically crops the bounding box given segment
        :param data: key: images dict to preprocess
        :return: key: images dict
        """
        d = dict(data)
        crops = get_crops(d[self.segment_key], crops=self.crops, mode=self.mode, space=self.space, nr_per_dim=self.nr_per_dim,
                          edges=self.edges)
        ret: List[Dict[Hashable, torch.Tensor]] = [dict(data) for _ in range(len(crops))]

        for i in range(len(crops)):
            for key in set(data.keys()).difference(set(self.keys)):
                ret[i][key] = deepcopy(data[key])

        for key in self.key_iterator(dict(data)):
            for i, center in enumerate(crops):
                ret[i][key] = SpatialCrop(roi_center=center, roi_size=self.spatial_size)(d[key])

        return ret


class DistributionRandCropd(MapTransform):
    def __init__(self, keys: KeysCollection, segment_key: str, spatial_size: Union[Sequence[int], int],
                 number_classes: int,
                 probabilities: Sequence[float],
                 allow_missing_keys: bool = False,
                 ) -> None:
        """
        systematically crops the bounding box given segment
        :param keys: keys to crop
        :param segment_key: segment where the crops should center in
        :param spatial_size: size of the crops
        :param probabilities: probabilities of each class to be in the crop center
        allow_missing_keys: don't raise exception if key is missing.
        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.segment_key = segment_key
        self.spatial_size = spatial_size
        self.number_classes = number_classes
        self.probabilities = np.array(probabilities)
        self.probabilities = self.probabilities / self.probabilities.sum()

        assert self.number_classes == len(
            self.probabilities), f"number of classes does not match number of given probabilities"

        assert sum(self.probabilities) == 1, f"probabilities does not sum up to 1!"

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, List[torch.Tensor]]:
        """
        systematically crops the bounding box given segment
        :param data: key: images dict to preprocess
        :return: key: images dict
        """
        # create d as dict of data
        d = dict(data)

        label = deepcopy(d[self.segment_key][0])
        label[:self.spatial_size[0] // 2 + 1] = self.number_classes + 1
        label[-self.spatial_size[0] // 2 + 1:] = self.number_classes + 1

        label[:, :self.spatial_size[1] // 2 + 1] = self.number_classes + 1
        label[:, -self.spatial_size[1] // 2 + 1:] = self.number_classes + 1

        label[:, :, :self.spatial_size[2] // 2 + 1] = self.number_classes + 1
        label[:, :, -self.spatial_size[2] // 2 + 1:] = self.number_classes + 1

        # cropping lesion center only works on images with lesions
        if d['case'] == 'positive':
            # random choice of crop center by given probabilities
            crop_class = np.random.choice(self.number_classes, 1, p=self.probabilities).item()
        else:
            crop_class = 0

        # random choice center with crop class
        count = torch.count_nonzero((label == crop_class)).item()
        if count == 0:
            print(f"{d['case']=}")
            print(f"{label.shape=}")
            print(f"{d['t2w_meta_dict']['filename_or_obj']=}")
            print(f"{count=}, {d['case']=}, {crop_class=}")
        index = torch.randint(0, count, (1,)).item()
        index = (label == crop_class).nonzero()[index]

        # create cropper
        cropper = SpatialCrop(roi_center=index, roi_size=self.spatial_size)

        # crop the data
        for key in self.key_iterator(d):
            image = cropper(d[key])
            if image.shape != (1, 128, 128, 32):
                print(f"image got wrong shape! got {image.shape}!")
                print(f"{d['case']=}")
                print(f"{label.shape=}")
                print(f"{d['t2w_meta_dict']['filename_or_obj']=}")
                print(f"{count=}, {d['case']=}, {crop_class=}")
                print(f"{d[key].shape=}")
            d[key] = image

        return d


def get_segment_size(segment: torch.Tensor) -> Tuple[int, int, int, int, int, int]:
    """
    calcs upper and lower bound of segment in x, y and z dim
    :param segment: tensor with segment
    :return: xmin, xmax, ymin, ymax, zmin, zmax
    """

    x = torch.sum(segment, dim=(0, 2, 3))
    y = torch.sum(segment, dim=(0, 1, 3))
    z = torch.sum(segment, dim=(0, 1, 2))

    xmin, xmax = torch.where(x)[0][[0, -1]]
    ymin, ymax = torch.where(y)[0][[0, -1]]
    zmin, zmax = torch.where(z)[0][[0, -1]]

    return xmin, xmax, ymin, ymax, zmin, zmax


def get_roi(segment: torch.Tensor, edge: Tuple[int, int, int] = (0, 0, 0),
            kdiv: int = 32) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """
    calculates the center and the size of a roi given the segmentations of the prostate
    :param segment: tensor with the segmentation
    :param edge: space to add between segmentation and edge of roi
    :param kdiv: ensures size of roi is dividable by kdiv
    :return: coordinates of center and size of roi
    """
    xmin, xmax, ymin, ymax, zmin, zmax = get_segment_size(segment)

    center = xmin + ((xmax - xmin) // 2), ymin + ((ymax - ymin) // 2), zmin + ((zmax - zmin) // 2)

    size = torch.tensor([xmax - xmin + 2 * edge[0], ymax - ymin + 2 * edge[1], zmax - zmin + 2 * edge[2]])
    size += kdiv - (size % kdiv)

    return center, tuple(size)


def get_crops(segment: torch.Tensor,
              mode: str = "arange",
              crops: Tuple[int, int, int] = (128, 128, 96),
              space: Tuple[int, int, int] = (5, 5, 5),
              nr_per_dim: Tuple[int, int, int] = (5, 5, 3),
              edges: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]] = ((0, 0), (0, 0), (0, 0))
              ) -> Sequence[Tuple[int, int, int]]:
    """
    calcs the center of systematic crops over the given segment
    :param mode: 'arange' for fixed spaces between crops, 'linspace' for fixed number of crops
    :param crops: size of crops
    :param segment: segment to be cropped
    :param space: space between crops in dim (x, y, z)
    :param edges: extra space added to segmentation roi
    :return: List of centers
    """
    # get sizes
    _, x, y, z = segment.shape

    # get extensions of segment
    xmin, xmax, ymin, ymax, zmin, zmax = get_segment_size(segment)

    # systematic points over x, y and z dim
    function = torch.arange if mode == "arange" else torch.linspace
    param = space if mode == "arange" else nr_per_dim
    x_pos = function(max(xmin - edges[0][0], (crops[0] // 2 + 1)), min(xmax + edges[0][1], x - (crops[0] // 2 + 1)),
                     param[0])
    y_pos = function(max(ymin - edges[1][0], (crops[1] // 2 + 1)), min(ymax + edges[1][1], y - ((crops[1] // 2) + 1)),
                     param[1])
    z_pos = function(max(zmin - edges[2][0], (crops[2] // 2 + 1)), min(zmax + edges[2][1], z - ((crops[2] // 2) + 1)),
                     param[2])

    # calc positions
    pos = torch.stack(torch.meshgrid([x_pos, y_pos, z_pos])).reshape(3, -1).T

    return pos