"""
D-FINE-S目标检测器训练模块：使用预训练权重在SCADA数据集上进行微调
评估标准：mAP@0.5、mAP@0.5:0.95
"""
import random
import time
from pathlib import Path
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader,Dataset
from transformers import DFineConfig,DFineForObjectDetection,RTDetrImageProcessor



IMAGE_SUFFIXES={".png",".jpg",".jpeg",".bmp"}
MAP_IOU_THRESHOLDS=[round(0.5+0.05*i,2) for i in range(10)]



class YoloDetectionDataset(Dataset):
    def __init__(self,image_folder,label_folder,image_processor,nc):
        self.image_folder=image_folder
        self.label_folder=label_folder
        self.image_processor=image_processor
        self.nc=nc
        self.image_paths=sorted(
            [path for path in image_folder.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES],
            key=lambda path:int(path.stem) if path.stem.isdigit() else path.stem
        )
        if not self.image_paths:
            raise RuntimeError(f"图像目录为空：{image_folder}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self,index):
        image_path=self.image_paths[index]
        label_path=self.label_folder/f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(label_path)

        # 1. 读取原始图像及YOLO归一化标签
        with Image.open(image_path) as image:
            image=image.convert("RGB")
            image_width,image_height=image.size
            annotations=[]
            with open(label_path,"r",encoding="utf-8") as f:
                for annotation_id,line in enumerate(f):
                    cls,x_center,y_center,width,height=line.strip().split()
                    cls=int(cls)
                    x_center=float(x_center)
                    y_center=float(y_center)
                    width=float(width)
                    height=float(height)
                    if cls<0 or cls>=self.nc:
                        raise ValueError(f"标签类别超出范围：{label_path}，class_id={cls}")

                    # 2. 将YOLO的cxcywh转换为图像处理器所需的COCO像素坐标
                    box_width=width*image_width
                    box_height=height*image_height
                    x=(x_center-width/2)*image_width
                    y=(y_center-height/2)*image_height
                    annotations.append({
                        "id":annotation_id,
                        "image_id":index,
                        "category_id":cls,
                        "bbox":[x,y,box_width,box_height],
                        "area":box_width*box_height,
                        "iscrowd":0
                    })

            # 3. 同步缩放图像和标注框，生成D-FINE-S训练输入
            target={"image_id":index,"annotations":annotations}
            encoded=self.image_processor(images=image,annotations=target,return_tensors="pt")
        return encoded["pixel_values"].squeeze(0),encoded["labels"][0]



def collate_fn(batch):
    pixel_values=torch.stack([sample[0] for sample in batch])
    labels=[sample[1] for sample in batch]
    return pixel_values,labels



def move_labels_to_device(labels,device):
    return [{key:value.to(device) if torch.is_tensor(value) else value for key,value in label.items()} for label in labels]



def calculate_iou(bbox1,bbox2):
    x1=max(bbox1[0],bbox2[0])
    y1=max(bbox1[1],bbox2[1])
    x2=min(bbox1[2],bbox2[2])
    y2=min(bbox1[3],bbox2[3])
    inter_width=max(0.0,x2-x1)
    inter_height=max(0.0,y2-y1)
    inter=inter_width*inter_height
    area1=(bbox1[2]-bbox1[0])*(bbox1[3]-bbox1[1])
    area2=(bbox2[2]-bbox2[0])*(bbox2[3]-bbox2[1])
    union=area1+area2-inter
    return inter/union if union>0 else 0.0



def calculate_ap(recalls,precisions):
    recalls=[0.0]+recalls+[1.0]
    precisions=[0.0]+precisions+[0.0]
    for index in range(len(precisions)-2,-1,-1):
        precisions[index]=max(precisions[index],precisions[index+1])
    ap=0.0
    for index in range(len(recalls)-1):
        if recalls[index+1]!=recalls[index]:
            ap+=(recalls[index+1]-recalls[index])*precisions[index+1]
    return ap



def calculate_class_ap(predictions,targets,iou_threshold):
    if not targets:
        return None
    predictions=sorted(predictions,key=lambda prediction:prediction[5],reverse=True)
    targets_by_image={}
    for target in targets:
        targets_by_image.setdefault(target[0],[]).append(target)
    matched={image_id:[False]*len(image_targets) for image_id,image_targets in targets_by_image.items()}

    true_positives=[]
    false_positives=[]
    for prediction in predictions:
        image_id=prediction[0]
        best_iou=0.0
        best_target_index=-1
        for target_index,target in enumerate(targets_by_image.get(image_id,[])):
            if matched[image_id][target_index]:
                continue
            iou=calculate_iou(prediction[1:5],target[1:5])
            if iou>best_iou:
                best_iou=iou
                best_target_index=target_index
        if best_iou>=iou_threshold:
            true_positives.append(1)
            false_positives.append(0)
            matched[image_id][best_target_index]=True
        else:
            true_positives.append(0)
            false_positives.append(1)

    recalls=[]
    precisions=[]
    true_positive_count=0
    false_positive_count=0
    for true_positive,false_positive in zip(true_positives,false_positives):
        true_positive_count+=true_positive
        false_positive_count+=false_positive
        recalls.append(true_positive_count/len(targets))
        precisions.append(true_positive_count/(true_positive_count+false_positive_count))
    return calculate_ap(recalls,precisions)



def calculate_map(predictions,targets,nc):
    class_ap50=[]
    class_ap50_95=[]
    for class_id in range(nc):
        class_predictions=[prediction for prediction in predictions if prediction[-1]==class_id]
        class_targets=[target for target in targets if target[-1]==class_id]
        if not class_targets:
            continue
        aps=[calculate_class_ap(class_predictions,class_targets,iou_threshold) for iou_threshold in MAP_IOU_THRESHOLDS]
        class_ap50.append(aps[0])
        class_ap50_95.append(sum(aps)/len(aps))
    if not class_ap50:
        raise ValueError("验证集中没有可用于评估的真实框")
    return sum(class_ap50)/len(class_ap50),sum(class_ap50_95)/len(class_ap50_95)



def collect_detection_results(outputs,labels,image_processor,conf_thresh,all_predictions,all_targets):
    # 1. 将D-FINE-S预测框恢复到每张验证图像的原始尺寸
    target_sizes=torch.stack([label["orig_size"] for label in labels])
    results=image_processor.post_process_object_detection(
        outputs,
        threshold=conf_thresh,
        target_sizes=target_sizes
    )

    # 2. 收集预测框与真实框，供每个IoU阈值进行一对一匹配
    for result,label in zip(results,labels):
        image_id=int(label["image_id"].item())
        for box,score,class_id in zip(result["boxes"],result["scores"],result["labels"]):
            x1,y1,x2,y2=box.detach().cpu().tolist()
            all_predictions.append([image_id,x1,y1,x2,y2,float(score),int(class_id)])

        original_height,original_width=label["orig_size"].detach().cpu().tolist()
        for box,class_id in zip(label["boxes"],label["class_labels"]):
            x_center,y_center,width,height=box.detach().cpu().tolist()
            x1=(x_center-width/2)*original_width
            y1=(y_center-height/2)*original_height
            x2=(x_center+width/2)*original_width
            y2=(y_center+height/2)*original_height
            all_targets.append([image_id,x1,y1,x2,y2,int(class_id)])



def calculate_validation_metrics(detector,data_loader,device,use_amp,image_processor,conf_thresh,nc):
    detector.eval()
    total_loss=0.0
    all_predictions=[]
    all_targets=[]
    with torch.inference_mode():
        for pixel_values,labels in data_loader:
            pixel_values=pixel_values.to(device,non_blocking=True)
            labels=move_labels_to_device(labels,device)
            with torch.amp.autocast("cuda",enabled=use_amp):
                outputs=detector(pixel_values=pixel_values,labels=labels)
            total_loss+=outputs.loss.item()
            collect_detection_results(outputs,labels,image_processor,conf_thresh,all_predictions,all_targets)
    val_loss=total_loss/len(data_loader)
    map50,map50_95=calculate_map(all_predictions,all_targets,nc)
    return val_loss,map50,map50_95



def save_deployment_model(detector,cls_list,input_size,output_model):
    # 仅保存部署需要的模型结构与参数，不保存优化器等训练状态
    checkpoint={
        "model_state_dict":detector.state_dict(),
        "model_config":detector.config.to_dict(),
        "cls_list":cls_list,
        "input_size":input_size
    }
    output_model.parent.mkdir(parents=True,exist_ok=True)
    torch.save(checkpoint,output_model)



def main():
    print("正在加载训练配置...")
    project_root=Path(__file__).resolve().parents[3]
    config_path=Path(__file__).resolve().with_name("config.yaml")
    with open(config_path,"r",encoding="utf-8") as f:
        config=yaml.safe_load(f)

    dataset_path=(project_root/config["dataset_path"]).resolve()
    pretrained_model=config["pretrained_model"]
    input_size=config["input_size"]
    device_id=config["device"]
    epochs=config["epochs"]
    batch_size=config["batch_size"]
    workers=config["workers"]
    patience=config["patience"]
    learning_rate=config["learning_rate"]
    backbone_learning_rate=config["backbone_learning_rate"]
    weight_decay=config["weight_decay"]
    gradient_clip_norm=config["gradient_clip_norm"]
    metric_conf_thresh=config["metric_conf_thresh"]
    seed=config["seed"]
    output_model=(project_root/config["output_model"]).resolve()
    cls_list=config["cls_list"]
    nc=len(cls_list)

    if not torch.cuda.is_available():
        raise RuntimeError("D-FINE-S训练需要CUDA设备，但当前未检测到可用的CUDA设备")
    device=torch.device(f"cuda:{device_id}")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 1. 使用固定输入尺寸初始化训练与推理共用的图像处理器
    image_processor=RTDetrImageProcessor(
        do_resize=True,
        size={"height":input_size,"width":input_size},
        do_rescale=True,
        rescale_factor=1/255,
        do_normalize=False,
        do_pad=False,
        format="coco_detection",
        do_convert_annotations=True
    )

    # 2. 直接读取现有YOLO数据集，不生成额外的COCO标注文件
    train_dataset=YoloDetectionDataset(
        dataset_path/"images"/"train",
        dataset_path/"labels"/"train",
        image_processor,
        nc
    )
    val_dataset=YoloDetectionDataset(
        dataset_path/"images"/"val",
        dataset_path/"labels"/"val",
        image_processor,
        nc
    )
    train_loader=DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    val_loader=DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # 3. 加载COCO预训练的D-FINE-S，并将分类头替换为SCADA的9个类别
    print("正在加载D-FINE-S预训练模型...")
    id2label={idx:name for idx,name in enumerate(cls_list)}
    label2id={name:idx for idx,name in enumerate(cls_list)}
    model_config=DFineConfig.from_pretrained(pretrained_model)
    model_config.num_labels=nc
    model_config.id2label=id2label
    model_config.label2id=label2id
    detector=DFineForObjectDetection.from_pretrained(
        pretrained_model,
        config=model_config,
        ignore_mismatched_sizes=True
    )
    detector.to(device)

    # 4. 骨干网络使用较小学习率，其余模块使用D-FINE-S主学习率
    backbone_parameters=[]
    other_parameters=[]
    for name,parameter in detector.named_parameters():
        if not parameter.requires_grad:
            continue
        if "backbone" in name:
            backbone_parameters.append(parameter)
        else:
            other_parameters.append(parameter)
    optimizer=torch.optim.AdamW(
        [
            {"params":backbone_parameters,"lr":backbone_learning_rate},
            {"params":other_parameters,"lr":learning_rate}
        ],
        weight_decay=weight_decay,
        betas=(0.9,0.999)
    )
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)
    use_amp=True
    scaler=torch.amp.GradScaler("cuda",enabled=use_amp)

    # 5. 逐轮训练并保存验证集mAP@0.5:0.95最高的部署权重
    print("训练开始...")
    start_time=time.time()#训练计时开始
    best_map50_95=-1.0
    best_epoch=0
    epochs_without_improvement=0
    for epoch in range(epochs):
        detector.train()
        total_train_loss=0.0
        for pixel_values,labels in train_loader:
            pixel_values=pixel_values.to(device,non_blocking=True)
            labels=move_labels_to_device(labels,device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda",enabled=use_amp):
                outputs=detector(pixel_values=pixel_values,labels=labels)
                loss=outputs.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(detector.parameters(),gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            total_train_loss+=loss.item()

        train_loss=total_train_loss/len(train_loader)
        val_loss,map50,map50_95=calculate_validation_metrics(
            detector,
            val_loader,
            device,
            use_amp,
            image_processor,
            metric_conf_thresh,
            nc
        )
        scheduler.step()
        print(
            f"Epoch {epoch+1}/{epochs}，train_loss={train_loss:.4f}，val_loss={val_loss:.4f}，"
            f"mAP50={map50:.4f}，mAP50-95={map50_95:.4f}"
        )

        if map50_95>best_map50_95:
            best_map50_95=map50_95
            best_epoch=epoch+1
            epochs_without_improvement=0
            save_deployment_model(detector,cls_list,input_size,output_model)
            print(f"最佳权重已保存至：{output_model}")
        else:
            epochs_without_improvement+=1
            if epochs_without_improvement>=patience:
                print(f"验证集mAP50-95连续{patience}轮未改善，提前结束训练")
                break

    print("训练结束")
    end_time=time.time()#训练计时结束
    sum_time=end_time-start_time#总用时秒数
    print(f"总用时：{sum_time/3600:.2f}h")
    print(f"最佳训练轮次：{best_epoch}，最佳验证集mAP50-95：{best_map50_95:.4f}")



if __name__=="__main__":
    main()
