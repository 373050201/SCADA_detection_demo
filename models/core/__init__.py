"""
目标检测模型公共接口模块
"""

from models.core.data_structure import Detection
from models.core.base_detector import BaseDetector


__all__ = ["BaseDetector", "Detection"]
