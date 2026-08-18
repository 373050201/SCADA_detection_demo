"""
模型训练
评估标准：mAP@0.5
"""
from models.detectors.grid_anchor.loss import Loss
from models.detectors.grid_anchor.network import GridAnchorDetector
from models.detectors.grid_anchor.dataset import YoloDataset
from models.detectors.grid_anchor.calculate import calculate_mAP, calculate_kmeans, nms
import yaml
from torch.utils.data import DataLoader
import torch
import time



print("正在更新锚框...")
train_dataset=YoloDataset(split="train",label_transform="no_transform")
calculate_kmeans(train_dataset,write_yaml=True)

print("正在加载训练/验证集...")
with open("models/detectors/grid_anchor/config.yaml",'r') as f:
    config=yaml.safe_load(f)
anchors=config["anchors"]
nc=config["nc"]
mAP_iou_threshold=config["mAP_iou_threshold"]#mAP评估时的iou阈值
batch_size=config["batch_size"]
learning_rate=config["learning_rate"]
grid_size = config["input_size"] // 32# 网格边长
train_dataset=YoloDataset("train")
train_loader=DataLoader(train_dataset,batch_size,shuffle=True,drop_last=True)
val_dataset=YoloDataset("val")
val_loader=DataLoader(val_dataset,batch_size,shuffle=False)

print("正在初始化模型...")
detector=GridAnchorDetector()
detector=detector.cuda()
loss=Loss()
loss=loss.cuda()
optimizer=torch.optim.Adam(detector.parameters(),lr=learning_rate)

start_time=time.time()#训练计时开始
total_step=0
epoch=100
best_mAP=0#最佳mAP
best_epoch=0#最佳训练轮次
count=0#早停计数
patience=15#最多容忍patience个epoch没有改善
skip_mAP_epochs = 10#前skip_mAP_epochs轮验证阶段不计算mAP
conf_threshold = 0.5 #置信度大于此值,判断为预测框
for i in range(epoch):
    print(f"第{i+1}轮训练开始...")
    detector.train()#训练模式
    total_train_loss=0
    for imgs,labels in train_loader:
        imgs=imgs.cuda()
        labels=labels.cuda()

        preds=detector(imgs)
        train_loss=loss(preds,labels)
        total_train_loss+=train_loss
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        total_step+=1
    print(f"此轮已累计训练了{total_step}次")
    print(f"训练集总损失：{total_train_loss:.6f}")

    detector.eval()#评估模式
    total_val_loss=0
    all_preds=[]#所有预测框
    all_tgts=[]#所有目标框
    img_id=0
    with torch.no_grad():
        for imgs,labels in val_loader:
            imgs=imgs.cuda()
            labels=labels.cuda()

            preds=detector(imgs)
            val_loss=loss(preds,labels)
            total_val_loss+=val_loss

            if i+1<=skip_mAP_epochs:#当前轮次小于skip_mAP_epochs，则不计算mAP
                continue

            #评估mAP
            for b in range(imgs.size(0)):
                init_preds=[]#暂存此图nms前的所有预测框
                for grid_y in range(grid_size):
                    for grid_x in range(grid_size):
                        for anchor_idx in range(5):
                            anchor_w,anchor_h=anchors[anchor_idx]#锚框宽高

                            conf_tgt=labels[b,grid_y,grid_x,anchor_idx,4]#目标框置信度解码
                            if conf_tgt>0.5:#置信度大于此值,判断为真实框
                                cls_tgt=torch.argmax(labels[b,grid_y,grid_x,anchor_idx,5:])#目标框类别解码
                                #目标框bbox解码
                                tx_tgt,ty_tgt,tw_tgt,th_tgt=labels[b,grid_y,grid_x,anchor_idx,:4]
                                x_c_tgt=(grid_x+tx_tgt)/float(grid_size)
                                y_c_tgt=(grid_y+ty_tgt)/float(grid_size)
                                w_tgt=anchor_w*torch.exp(tw_tgt)
                                h_tgt=anchor_h*torch.exp(th_tgt)
                                all_tgts.append([img_id,x_c_tgt,y_c_tgt,w_tgt,h_tgt,conf_tgt,cls_tgt])#加入所有目标框的列表

                            conf_pred=preds[b,grid_y,grid_x,anchor_idx,4]#预测框置信度解码
                            if conf_pred>conf_threshold:
                                cls_pred=torch.argmax(preds[b,grid_y,grid_x,anchor_idx,5:])#预测框类别解码
                                #预测框bbox解码
                                tx_pred,ty_pred,tw_pred,th_pred=preds[b,grid_y,grid_x,anchor_idx,:4]
                                x_c_pred=(grid_x+tx_pred)/float(grid_size)
                                y_c_pred=(grid_y+ty_pred)/float(grid_size)
                                w_pred=anchor_w*torch.exp(tw_pred)
                                h_pred=anchor_h*torch.exp(th_pred)
                                init_preds.append([x_c_pred,y_c_pred,w_pred,h_pred,conf_pred,cls_pred])#存入暂存区
                final_preds=nms(init_preds)#nms处理,去除重叠框
                final_preds=[[img_id]+sublist for sublist in final_preds]#在所有子列表第一个位置添加img_id,用于区分不同图的预测框
                all_preds.extend(final_preds)#将此图处理后的预测框加入所有预测框的列表
                img_id+=1
    print(f"验证集总损失：{total_val_loss:.6f}")

    if i+1<=skip_mAP_epochs:
        print("本轮跳过mAP计算")
        continue

    mAP=calculate_mAP(all_preds,all_tgts,nc,mAP_iou_threshold)#计算mAP
    print(f"mAP@{mAP_iou_threshold}：{mAP:.4f}")
    if mAP>best_mAP:
        best_mAP=mAP
        best_epoch=i+1
        count=0
        torch.save(detector.state_dict(),f"./models/weights/best_GridAnchor.pth")
    else:
        count+=1
        if count>=patience:
            print("早停触发！")
            break

print("训练结束")
end_time=time.time()#训练计时结束
sum_time=end_time-start_time#总用时秒数
print(f"总用时：{sum_time/3600:.2f}h")
print(f"最佳训练轮次：{best_epoch}，最佳验证集mAP@{mAP_iou_threshold}：{best_mAP:.4f}")
