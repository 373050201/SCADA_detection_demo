"""
目标检测器加载模块：根据配置动态加载指定检测器
"""
from importlib import import_module



def load_detector(detector_name, config_path, model_path):
    module = import_module(f"models.detectors.{detector_name}")
    if not hasattr(module, "create_detector"):
        raise RuntimeError(f"目标检测器{detector_name}未提供create_detector入口")
    return module.create_detector(config_path, model_path)
