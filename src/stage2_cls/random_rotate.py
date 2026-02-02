
import mmcv
import numpy as np
from mmcv.transforms import BaseTransform
from mmengine.registry import TRANSFORMS

@TRANSFORMS.register_module()
class RandomRotate(BaseTransform):
    def __init__(self, prob=0.5, angle=(-180, 180)):
        self.prob = prob
        self.angle = angle

    def transform(self, results):
        if np.random.rand() < self.prob:
            angle = np.random.uniform(self.angle[0], self.angle[1])
            results['img'] = mmcv.imrotate(results['img'], angle)
        return results
