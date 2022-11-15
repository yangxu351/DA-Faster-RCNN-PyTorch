# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .coco import COCODataset
from .voc import PascalVOCDataset
from .concat_dataset import ConcatDataset
# tag: yang added
from .real_wdt import RealWDT
from .synthetic_wdt import SyntheticWDT

# __all__ = ["COCODataset", "ConcatDataset", "PascalVOCDataset"]
# tag: yang added
__all__ = ["COCODataset", "ConcatDataset", "PascalVOCDataset", "RealWDT", "SyntheticWDT"]
