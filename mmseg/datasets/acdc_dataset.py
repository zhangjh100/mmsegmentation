# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class acdcDataset(BaseSegDataset):
    """ISPRS Potsdam dataset.

    In segmentation map annotation for Potsdam dataset, 0 is the ignore index.
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """
    METAINFO = dict(
        classes=('LV', 'Myo', 'RV'),
        palette=[[171, 171, 171], [114, 114, 114], [57, 57, 57]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
