"""
视觉服务模块：用目标检测模型定位主界面目标位置
"""
import yaml
from models.loader import load_detector
from rapidocr import RapidOCR



class ModelService:
    def __init__(self, model_path=None):
        with open("configs/vision.yaml") as f:
            config = yaml.safe_load(f)
        self.cls_list = config["cls_list"]
        self.text_class_id = config["ocr"]["text_class_id"]
        detector_config = config["detector"]
        if model_path is None:
            model_path = detector_config["model_path"]
        self.detector = load_detector(detector_config["name"],detector_config["config_path"],model_path)

        self.ocr_engine = RapidOCR(# 初始化文本识别引擎
        params={
            "EngineConfig.onnxruntime.providers": ["CUDAExecutionProvider",],
            "EngineConfig.onnxruntime.cuda_ep_cfg": {"device_id": 0}
        }
)
    
    def predict(self, pil_image, conf_thresh=0.72, iou_thresh=0.2):
        """
        预处理、推理、NMS全部在此完成，返回目标信息列表 [(px, py, cls, text), ...]
        :param pil_image: 原始图像
        :param conf_thresh: 若conf大于此值，判断为正样本
        :param iou_thresh: 若iou大于此值，判断为重叠框
        """
        detections = self.detector.predict(pil_image,conf_thresh,iou_thresh)
        targets = []# 目标信息列表
        for detection in detections:
            # 目标在主界面客户区的中心坐标
            px,py = detection.center
            cls = detection.class_id
            # bbox左上角和右下角坐标，用于裁剪
            x1 = int(detection.x1)
            y1 = int(detection.y1)
            x2 = int(detection.x2)
            y2 = int(detection.y2)
            # 文本内容
            text = "" # 默认为空
            if cls == self.text_class_id: # 仅对文本类别做文字识别
                # 裁剪子图
                crop_box = (x1, y1, x2, y2)
                cropped_img=pil_image.crop(crop_box)
                # 通过OCR引擎
                result = self.ocr_engine(cropped_img)
                if result and len(result.txts)>0:
                    text = result.txts[0].strip() # 默认取第一个预测结果
                    
            targets.append((px, py, cls,text))
        return targets