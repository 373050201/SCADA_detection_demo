"""
YOLO11目标检测器模块：提供统一的检测器创建入口
"""
from models.detectors.yolo11.adapter import YOLO11DetectorAdapter



def create_detector(config_path, model_path):
    return YOLO11DetectorAdapter(config_path, model_path)
