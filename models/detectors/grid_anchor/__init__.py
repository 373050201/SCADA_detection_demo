"""
GridAnchor自研目标检测器模块：提供统一的检测器创建入口
"""
from models.detectors.grid_anchor.adapter import GridAnchorDetectorAdapter



def create_detector(config_path, model_path):
    return GridAnchorDetectorAdapter(config_path, model_path)
