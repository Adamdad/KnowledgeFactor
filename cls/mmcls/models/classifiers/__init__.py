# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseClassifier
from .image import (ImageBranchClassifier, ImageClassifier,
                    ImagePseudoMultiTaskClassifier, MultiTaskImageClassifier)
from .kd import KDImageClassifier
from .kf import KFImageClassifier


__all__ = ['BaseClassifier', 'ImageClassifier',
           'KDImageClassifier', 'KFactorImageClassifier', 'KFactorImageClassifier2',
           'KFactorImageClassifier_Additive', 'KFactorImageClassifier_Ortho_Additive',
           'KFactorImageClassifier_no_Additive', 'KDDeepClassifier',
           'KnowledgeRelationClassifier', 'KFMulHeadImageClassifier',
           'ImagePseudoMultiTaskClassifier', 'MultiTaskImageClassifier',
           'KFPMTImageClassifier', 'KFPMTSeriesImageClassifier', 'KFPMTParralImageClassifier',
           'KFactorParallelImageClassifier', 'KFactorParallelAttnImageClassifier', 'KFactorParallelAddImageClassifier',
           'KFactorBranchImageClassifier', 'KFactorSeriesImageClassifier', 'KFactorBranchImageClassifier2',
           'KFactorBranchImageClassifier_cat', 'KFactorBranchImageClassifier_ConvCat',
           'KFTaskParallelImageClassifier', 'ImageBranchClassifier', 'ImageVibClassifier',
           'KFTaskParallelImageClassifierv2', 'KDDeepClassifier_InfoMax','KDImageClassifier_Select']
