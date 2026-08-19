"""
目标检测模型精度评估模块：用于统一比较不同检测器的检测结果
"""
import os
import sys
import time
from PIL import Image



PROJECT_ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,PROJECT_ROOT)
from models.loader import load_detector



DETECTOR_NAME="yolo11"#需要评测的检测器：[grid_anchor, yolo11]

CONFIG_PATH=os.path.join(PROJECT_ROOT,"models","detectors",DETECTOR_NAME,"config.yaml")
MODEL_PATHS={
    "grid_anchor":os.path.join(PROJECT_ROOT,"models","weights","best_GridAnchor.pth"),
    "yolo11":os.path.join(PROJECT_ROOT,"models","weights","best_yolo11s.pt")
}
MODEL_PATH=MODEL_PATHS[DETECTOR_NAME]
DATASET_PATH=os.path.join(PROJECT_ROOT,"datasets","SCADA_yolo")
CONF_THRESH=0.05#收集预测框的置信度下限
NMS_IOU_THRESHOLD=0.45#模型NMS的IoU阈值
IOU_THRESHOLD=0.5



def load_targets(label_path,img_w,img_h):#读取YOLO格式标签，转换为像素坐标xyxy
    targets=[]
    with open(label_path,"r") as f:
        for line in f:
            cls,x_c,y_c,w,h=line.strip().split()
            cls=int(cls)
            x_c=float(x_c)*img_w
            y_c=float(y_c)*img_h
            w=float(w)*img_w
            h=float(h)*img_h
            x1=x_c-w/2
            y1=y_c-h/2
            x2=x_c+w/2
            y2=y_c+h/2
            targets.append([x1,y1,x2,y2,cls])
    return targets



def calculate_iou(bbox1,bbox2):#计算两个xyxy格式bbox的IoU
    x1=max(bbox1[0],bbox2[0])
    y1=max(bbox1[1],bbox2[1])
    x2=min(bbox1[2],bbox2[2])
    y2=min(bbox1[3],bbox2[3])
    inter_w=max(0.0,x2-x1)
    inter_h=max(0.0,y2-y1)
    inter=inter_w*inter_h
    area1=(bbox1[2]-bbox1[0])*(bbox1[3]-bbox1[1])
    area2=(bbox2[2]-bbox2[0])*(bbox2[3]-bbox2[1])
    union=area1+area2-inter
    return inter/union



def calculate_ap(recalls,precisions):#使用全点插值法计算PR曲线下的面积
    recalls=[0.0]+recalls+[1.0]
    precisions=[0.0]+precisions+[0.0]
    for i in range(len(precisions)-2,-1,-1):
        precisions[i]=max(precisions[i],precisions[i+1])
    ap=0.0
    for i in range(len(recalls)-1):
        if recalls[i+1]!=recalls[i]:
            ap+=(recalls[i+1]-recalls[i])*precisions[i+1]
    return ap



def calculate_map(all_preds,all_tgts,nc,iou_threshold=0.5):#计算每个类别的AP与mAP
    results=[]
    for cls_id in range(nc):
        preds=[pred for pred in all_preds if pred[-1]==cls_id]
        tgts=[tgt for tgt in all_tgts if tgt[-1]==cls_id]
        preds.sort(key=lambda pred:pred[5],reverse=True)

        tgts_by_image={}
        for tgt in tgts:
            img_id=tgt[0]
            if img_id not in tgts_by_image:
                tgts_by_image[img_id]=[]
            tgts_by_image[img_id].append(tgt)
        matched={img_id:[False]*len(img_tgts) for img_id,img_tgts in tgts_by_image.items()}

        tp=[0]*len(preds)
        fp=[0]*len(preds)
        for pred_idx,pred in enumerate(preds):
            img_id=pred[0]
            best_iou=0.0
            best_tgt_idx=-1
            for tgt_idx,tgt in enumerate(tgts_by_image.get(img_id,[])):
                if matched[img_id][tgt_idx]:
                    continue
                iou=calculate_iou(pred[1:5],tgt[1:5])
                if iou>best_iou:
                    best_iou=iou
                    best_tgt_idx=tgt_idx
            if best_iou>=iou_threshold:
                tp[pred_idx]=1
                matched[img_id][best_tgt_idx]=True
            else:
                fp[pred_idx]=1

        recalls=[]
        precisions=[]
        tp_cum=0
        fp_cum=0
        for tp_value,fp_value in zip(tp,fp):
            tp_cum+=tp_value
            fp_cum+=fp_value
            recalls.append(tp_cum/len(tgts))
            precisions.append(tp_cum/(tp_cum+fp_cum+1e-10))
        ap=calculate_ap(recalls,precisions)
        results.append({"class_id":cls_id,"gt_count":len(tgts),"pred_count":len(preds),"ap":ap})

    mAP=sum(result["ap"] for result in results)/len(results)
    return mAP,results



def evaluate(detector,image_folder,label_folder,conf_thresh,nms_iou_thresh):#在测试集上收集预测框和真实框
    image_names=[name for name in os.listdir(image_folder) if name.lower().endswith(".png")]
    image_names.sort(key=lambda name:int(os.path.splitext(name)[0]))
    all_preds=[]
    all_tgts=[]

    for img_idx,image_name in enumerate(image_names):
        img_id=os.path.splitext(image_name)[0]
        image_path=os.path.join(image_folder,image_name)
        label_path=os.path.join(label_folder,f"{img_id}.txt")
        with Image.open(image_path) as img:
            img=img.convert("RGB")
            img_w,img_h=img.size
            targets=load_targets(label_path,img_w,img_h)
            detections=detector.predict(img,conf_thresh,nms_iou_thresh)

        for x1,y1,x2,y2,cls in targets:
            all_tgts.append([img_id,x1,y1,x2,y2,cls])
        for detection in detections:
            all_preds.append([
                img_id,
                detection.x1,
                detection.y1,
                detection.x2,
                detection.y2,
                detection.score,
                detection.class_id
            ])

        if (img_idx+1)%50==0 or img_idx+1==len(image_names):
            print(f"已完成推理：{img_idx+1}/{len(image_names)}")

    return all_preds,all_tgts



def main():
    image_folder=os.path.join(DATASET_PATH,"images","test")
    label_folder=os.path.join(DATASET_PATH,"labels","test")

    print(f"正在加载检测器：{DETECTOR_NAME}")
    detector=load_detector(DETECTOR_NAME,CONFIG_PATH,MODEL_PATH)
    print("正在评估测试集...")
    start_time=time.time()#评测计时开始
    all_preds,all_tgts=evaluate(detector,image_folder,label_folder,CONF_THRESH,NMS_IOU_THRESHOLD)
    mAP,results=calculate_map(all_preds,all_tgts,len(detector.cls_list),IOU_THRESHOLD)

    print(f"\n{'Class':<10}{'GT':>8}{'Pred':>8}{'AP@0.5':>12}")
    print("-"*38)
    for result in results:
        cls_name=detector.cls_list[result["class_id"]]
        print(f"{cls_name:<10}{result['gt_count']:>8}{result['pred_count']:>8}{result['ap']:>12.4f}")
    print("-"*38)
    print(f"mAP@0.5：{mAP:.4f}")
    end_time=time.time()#评测计时结束
    sum_time=end_time-start_time#总用时秒数
    print(f"总用时：{sum_time/60:.2f}min")



if __name__=="__main__":
    main()
