"""
目标检测器公共接口模块：定义所有检测模型必须实现的推理方法
"""
from abc import ABC, abstractmethod



class BaseDetector(ABC):
    @abstractmethod
    def predict(self, pil_image, conf_thresh=0.72, iou_thresh=0.2):
        pass
