"""
视觉服务模块：用目标检测模型定位主界面目标位置
"""
import torch
import yaml
from models.model import GridAnchorDetector
from torchvision import transforms
from models.calculate import nms_cross_cls



class ModelService:
    def __init__(self, model_path="models/best_mAP@0.5_model.pth"):
        with open("models/config.yaml") as f:
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
        预处理、推理、NMS全部在此完成，返回像素坐标列表 [(px, py, cls), ...]
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
        # 5. 将归一化的中心坐标转换为窗口像素坐标
        W, H = pil_image.size  # 注意：截图尺寸等于窗口尺寸
        points = []#像素坐标列表，xy是目标在主界面客户区的坐标
        for x_c, y_c, _, _, _, cls in bboxes:
            px = int(x_c * W)
            py = int(y_c * H)
            points.append((px, py, cls))
        return points