"""
D-FINE-S目标检测器适配模块：加载模型并将推理结果转换为统一检测格式
"""
import torch
import yaml
from transformers import DFineConfig, DFineForObjectDetection, RTDetrImageProcessor
from models.core.base_detector import BaseDetector
from models.core.data_structure import Detection



class DFineDetectorAdapter(BaseDetector):
    def __init__(self, config_path, model_path):
        with open(config_path,"r",encoding="utf-8") as f:
            config=yaml.safe_load(f)
        self.cls_list=config["cls_list"]
        self.input_size=config["input_size"]# 输入图像尺寸
        self.device_id=config["device"]# 推理设备
        self.max_det=config["max_det"]# 单张图像最大检测数量
        if not torch.cuda.is_available():
            raise RuntimeError("D-FINE-S检测器需要CUDA设备，但当前未检测到可用的CUDA设备")
        self.device=torch.device(f"cuda:{self.device_id}")

        # 1. 从单文件权重恢复模型结构、类别与参数
        checkpoint=torch.load(model_path,map_location="cpu",weights_only=True)
        checkpoint_cls_list=checkpoint.get("cls_list")
        if checkpoint_cls_list!=self.cls_list:
            raise RuntimeError("D-FINE-S模型类别与配置文件中的cls_list不一致")
        if checkpoint.get("input_size")!=self.input_size:
            raise RuntimeError("D-FINE-S模型输入尺寸与配置文件中的input_size不一致")
        model_config=DFineConfig.from_dict(checkpoint["model_config"])
        if model_config.num_labels!=len(self.cls_list):
            raise RuntimeError("D-FINE-S模型类别数量与配置文件中的cls_list不一致")
        self.detector=DFineForObjectDetection(model_config)
        self.detector.load_state_dict(checkpoint["model_state_dict"])
        self.detector.to(self.device)
        self.detector.eval()

        # 2. 使用与训练阶段一致的D-FINE图像预处理配置
        self.image_processor=RTDetrImageProcessor(
            do_resize=True,
            size={"height":self.input_size,"width":self.input_size},
            do_rescale=True,
            rescale_factor=1/255,
            do_normalize=False,
            do_pad=False,
            format="coco_detection",
            do_convert_annotations=True
        )

    def predict(self, pil_image, conf_thresh=0.72, iou_thresh=0.2):
        """
        预处理、推理与结果转换全部在此完成，返回统一目标检测结果列表
        :param pil_image: 原始图像
        :param conf_thresh: 若conf大于此值，判断为正样本
        :param iou_thresh: 为兼容公共接口而保留，D-FINE-S端到端推理不使用NMS
        """
        _=iou_thresh
        pil_image=pil_image.convert("RGB")
        image_width,image_height=pil_image.size

        # 1. 将原始图像缩放并转换为D-FINE-S输入张量
        inputs=self.image_processor(images=pil_image,return_tensors="pt")
        pixel_values=inputs["pixel_values"].to(self.device)

        # 2. 执行端到端推理并将预测框恢复到原图尺寸
        with torch.inference_mode():
            outputs=self.detector(pixel_values=pixel_values)
        target_sizes=torch.tensor([[image_height,image_width]],device=self.device)
        result=self.image_processor.post_process_object_detection(
            outputs,
            threshold=conf_thresh,
            target_sizes=target_sizes
        )[0]

        # 3. 限制最大检测数量并转换为统一目标检测结果
        boxes=result["boxes"][:self.max_det].detach().cpu().tolist()
        scores=result["scores"][:self.max_det].detach().cpu().tolist()
        labels=result["labels"][:self.max_det].detach().cpu().tolist()
        detections=[]
        for box,score,label in zip(boxes,scores,labels):
            x1,y1,x2,y2=box
            x1=max(0.0,min(float(x1),float(image_width)))
            y1=max(0.0,min(float(y1),float(image_height)))
            x2=max(0.0,min(float(x2),float(image_width)))
            y2=max(0.0,min(float(y2),float(image_height)))
            detections.append(Detection(x1,y1,x2,y2,float(score),int(label)))
        return detections
