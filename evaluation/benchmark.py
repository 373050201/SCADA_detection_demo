"""
目标检测模型性能测试模块：用于统一记录权重大小、平均延迟与峰值显存
"""
import gc
import os
import sys
import time
import torch
from PIL import Image



PROJECT_ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,PROJECT_ROOT)
from models.loader import load_detector

DATASET_PATH=os.path.join(PROJECT_ROOT,"datasets","SCADA_yolo")
IMAGE_FOLDER=os.path.join(DATASET_PATH,"images","test")
WARMUP_RUNS=10#正式测试前的预热次数
CONF_THRESH=0.72#检测框置信度阈值
NMS_IOU_THRESHOLD=0.2#模型NMS的IoU阈值

DETECTORS={
    "grid_anchor":{
        "config":os.path.join(PROJECT_ROOT,"models","detectors","grid_anchor","config.yaml"),
        "weights":os.path.join(PROJECT_ROOT,"models","weights","best_GridAnchor.pth")
    },
    "yolo11":{
        "config":os.path.join(PROJECT_ROOT,"models","detectors","yolo11","config.yaml"),
        "weights":os.path.join(PROJECT_ROOT,"models","weights","best_yolo11s.pt")
    }
}



def load_rgb_image(image_path):#读取RGB图像，磁盘读取时间不计入平均延迟
    with Image.open(image_path) as image:
        return image.convert("RGB")



def main():
    if not torch.cuda.is_available():
        raise RuntimeError("当前检测器使用GPU推理，但未检测到可用的CUDA设备")

    image_names=[name for name in os.listdir(IMAGE_FOLDER) if name.lower().endswith((".png",".jpg",".jpeg",".bmp"))]
    image_names.sort()
    if not image_names:
        raise RuntimeError(f"测试集图像目录为空：{IMAGE_FOLDER}")

    os.chdir(PROJECT_ROOT)#GridAnchorDetector目前使用相对于项目根目录的配置路径

    print(f"测试设备：{torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"测试图像：{len(image_names)}张，预热次数：{WARMUP_RUNS}次")
    print(f"置信度阈值：{CONF_THRESH}，NMS IoU阈值：{NMS_IOU_THRESHOLD}，batch size：1")

    results=[]
    for detector_name,paths in DETECTORS.items():
        if not os.path.isfile(paths["config"]):
            raise FileNotFoundError(paths["config"])
        if not os.path.isfile(paths["weights"]):
            raise FileNotFoundError(paths["weights"])

        print(f"\n正在测试检测器：{detector_name}")
        detector=None
        try:
            detector=load_detector(detector_name,paths["config"],paths["weights"])
            weight_size_mib=os.path.getsize(paths["weights"])/(1024**2)#权重文件大小，单位MiB

            for image_name in image_names[:WARMUP_RUNS]:#预热CUDA内核与模型延迟初始化
                image_path=os.path.join(IMAGE_FOLDER,image_name)
                detector.predict(load_rgb_image(image_path),CONF_THRESH,NMS_IOU_THRESHOLD)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()#预热结束后重新记录峰值显存

            latencies_ms=[]
            for img_idx,image_name in enumerate(image_names):
                image_path=os.path.join(IMAGE_FOLDER,image_name)
                image=load_rgb_image(image_path)

                torch.cuda.synchronize()#等待之前的CUDA任务完成
                start_time=time.perf_counter()
                detector.predict(image,CONF_THRESH,NMS_IOU_THRESHOLD)
                torch.cuda.synchronize()#等待本次推理任务完成
                latency_ms=(time.perf_counter()-start_time)*1000
                latencies_ms.append(latency_ms)

                if (img_idx+1)%50==0 or img_idx+1==len(image_names):
                    print(f"已完成推理：{img_idx+1}/{len(image_names)}")

            average_latency_ms=sum(latencies_ms)/len(latencies_ms)
            peak_memory_mib=torch.cuda.max_memory_allocated()/(1024**2)
            results.append({
                "detector":detector_name,
                "weight_size_mib":weight_size_mib,
                "average_latency_ms":average_latency_ms,
                "peak_memory_mib":peak_memory_mib
            })
        finally:
            if detector is not None:
                del detector
            gc.collect()
            torch.cuda.empty_cache()#测试下一个模型前释放当前模型显存
            torch.cuda.synchronize()

    print(f"\n{'Detector':<16}{'Weights(MiB)':>16}{'Latency(ms)':>16}{'VRAM(MiB)':>16}")
    print("-"*64)
    for result in results:
        print(f"{result['detector']:<16}{result['weight_size_mib']:>16.2f}{result['average_latency_ms']:>16.2f}{result['peak_memory_mib']:>16.2f}")
    print("\n平均延迟不包含磁盘读取，包含图像预处理、模型推理、NMS与结果转换")
    print("峰值显存为PyTorch记录的最大已分配显存，包含模型占用")



if __name__=="__main__":
    main()
