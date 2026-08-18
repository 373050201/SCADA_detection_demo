"""
YOLO11目标检测器适配模块：加载模型并将推理结果转换为统一检测格式
"""
import yaml
from ultralytics import YOLO
from models.core.base_detector import BaseDetector
from models.core.data_structure import Detection



class YOLO11DetectorAdapter(BaseDetector):
    def __init__(self, config_path, model_path):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        self.cls_list = config["cls_list"]
        self.input_size = config["input_size"]# 输入图像尺寸
        self.device = config["device"]# 推理设备
        self.agnostic_nms = config["agnostic_nms"]# 是否使用跨类别NMS
        self.max_det = config["max_det"]# 单张图像最大检测数量

        self.detector = YOLO(model_path)
        model_cls_list = self.detector.names
        if isinstance(model_cls_list, dict):
            model_cls_list = [model_cls_list[idx] for idx in sorted(model_cls_list)]
        if model_cls_list != self.cls_list:
            raise RuntimeError("YOLO11模型类别与配置文件中的cls_list不一致")

    def predict(self, pil_image, conf_thresh=0.72, iou_thresh=0.2):
        """
        预处理、推理、NMS全部在此完成，返回统一目标检测结果列表
        :param pil_image: 原始图像
        :param conf_thresh: 若conf大于此值，判断为正样本
        :param iou_thresh: 若iou大于此值，判断为重叠框
        """
        # 1. 预处理、推理、NMS
        results = self.detector.predict(
            source=pil_image,
            conf=conf_thresh,
            iou=iou_thresh,
            imgsz=self.input_size,
            device=self.device,
            agnostic_nms=self.agnostic_nms,
            max_det=self.max_det,
            verbose=False
        )
        # 2. 解析预测结果
        boxes = results[0].boxes
        if boxes is None:
            return []
        xyxys = boxes.xyxy.cpu().tolist()
        scores = boxes.conf.cpu().tolist()
        clses = boxes.cls.cpu().tolist()
        # 3. 将预测结果转换为统一检测格式
        detections = []# 统一目标检测结果列表
        for xyxy, score, cls in zip(xyxys, scores, clses):
            x1, y1, x2, y2 = xyxy
            detections.append(Detection(x1,y1,x2,y2,score,int(cls)))
        return detections
