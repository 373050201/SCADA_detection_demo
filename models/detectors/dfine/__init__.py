"""
D-FINE-S目标检测器模块：提供统一的检测器创建入口
"""
from models.detectors.dfine.adapter import DFineDetectorAdapter



def create_detector(config_path, model_path):
    return DFineDetectorAdapter(config_path, model_path)
