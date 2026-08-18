"""
YOLO11目标检测器训练模块：使用预训练权重在SCADA数据集上进行微调
评估标准：mAP@0.5、mAP@0.5:0.95
"""
import shutil
from pathlib import Path
import yaml
from ultralytics import YOLO



def main():
    print("正在加载训练配置...")
    project_root = Path(__file__).resolve().parents[3]
    config_path = Path(__file__).resolve().with_name("config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    dataset_config = (project_root / config["dataset_config"]).resolve()
    pretrained_model = config["pretrained_model"]
    cleanup_models = [(project_root / model_path).resolve() for model_path in config["cleanup_models"]]
    input_size = config["input_size"]
    device = config["device"]
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    workers = config["workers"]
    patience = config["patience"]
    project_dir = (project_root / config["project"]).resolve()
    name = config["name"]
    output_model = (project_root / config["output_model"]).resolve()

    print("正在加载YOLO11预训练模型...")
    detector = YOLO(pretrained_model)

    print("训练开始...")
    detector.train(
        data=str(dataset_config),
        epochs=epochs,
        imgsz=input_size,
        batch=batch_size,
        device=device,
        workers=workers,
        patience=patience,
        project=str(project_dir),
        name=name,
        exist_ok=True,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=0.0,
        translate=0.1,
        scale=0.1,
        fliplr=0.0,
        flipud=0.0,
        mosaic=0.0
    )
    print("训练结束")

    # 保存最佳权重并清理本次训练产生的其他文件
    run_dir = Path(detector.trainer.save_dir).resolve()
    best_model = Path(detector.trainer.best).resolve()

    output_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_model, output_model)
    print(f"最佳权重已保存至：{output_model}")

    shutil.rmtree(run_dir)
    cleanup_dir = project_dir
    while cleanup_dir != project_root:
        if not cleanup_dir.is_dir() or any(cleanup_dir.iterdir()):
            break
        cleanup_dir.rmdir()
        cleanup_dir = cleanup_dir.parent
    print(f"已清理本次训练输出目录：{run_dir}")

    for cleanup_model in cleanup_models:
        cleanup_model.unlink(missing_ok=True)
        print(f"已清理训练辅助模型：{cleanup_model}")



if __name__ == "__main__":
    main()
