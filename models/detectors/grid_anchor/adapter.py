"""
GridAnchor目标检测器适配模块：加载模型并将推理结果转换为统一检测格式
"""
import torch
import yaml
from torchvision import transforms
from models.core.base_detector import BaseDetector
from models.core.data_structure import Detection
from models.detectors.grid_anchor.network import GridAnchorDetector
from models.detectors.grid_anchor.calculate import nms_cross_cls



class GridAnchorDetectorAdapter(BaseDetector):
    def __init__(self, config_path, model_path):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        self.anchors = config["anchors"]
        self.cls_list = config["cls_list"]
        self.input_size = config["input_size"]# 输入图像尺寸

        self.grid_size=self.input_size//32# 网格尺寸
        self.detector = GridAnchorDetector().cuda()
        self.detector.load_state_dict(torch.load(model_path))
        self.detector.eval()

    def predict(self, pil_image, conf_thresh=0.72, iou_thresh=0.2):
        """
        预处理、推理、NMS全部在此完成，返回统一目标检测结果列表
        :param pil_image: 原始图像
        :param conf_thresh: 若conf大于此值，判断为正样本
        :param iou_thresh: 若iou大于此值，判断为重叠框
        """
        # 1. 预处理图像（resize到input_size×input_size，转为tensor）
        transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor()
        ])
        img_tensor = transform(pil_image).unsqueeze(0).cuda()  # [1, 3, input_size, input_size]
        # 2. 推理
        with torch.no_grad():
            pred = self.detector(img_tensor)
        pred = pred[0]  # [grid_size, grid_size, 5, 5+C]
        # 3. 解析预测结果，收集正样本
        bboxes = []
        for grid_y in range(self.grid_size):
            for grid_x in range(self.grid_size):
                for anchor_idx in range(5):
                    anchor_w, anchor_h = self.anchors[anchor_idx]
                    conf = pred[grid_y, grid_x, anchor_idx, 4]
                    if conf > conf_thresh:
                        tx, ty, tw, th = pred[grid_y, grid_x, anchor_idx, :4]
                        cls = torch.argmax(pred[grid_y, grid_x, anchor_idx, 5:]).item()
                        x_c = (grid_x + tx) / float(self.grid_size)
                        y_c = (grid_y + ty) / float(self.grid_size)
                        w = anchor_w * torch.exp(tw)
                        h = anchor_h * torch.exp(th)
                        bboxes.append([x_c.item(), y_c.item(), w.item(), h.item(),
                                        conf.item(), cls])
        # 4. NMS
        bboxes = nms_cross_cls(bboxes, iou_thresh)
        # 5. 将归一化坐标转换为窗口像素坐标
        W, H = pil_image.size  # 注意：截图尺寸等于窗口尺寸
        detections = []# 统一目标检测结果列表
        for x_c, y_c, w, h, score, cls in bboxes:
            x1 = (x_c - w/2) * W
            y1 = (y_c - h/2) * H
            x2 = (x_c + w/2) * W
            y2 = (y_c + h/2) * H
            detections.append(Detection(x1,y1,x2,y2,score,cls))
        return detections
