"""
生成模拟的UI，用于训练模型
"""
import sys
import os
import random
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import Qt, QRectF, QTimer
from PyQt5.QtGui import QPainter, QColor, QFont, QPainterPath, QPen



# ---------- 自动保存相关配置 ----------
SAVE_DIR =  os.path.join("datasets", "SCADA_yolo")        # 数据集根目录
IMAGE_DIR = os.path.join(SAVE_DIR, "images")
LABEL_DIR = os.path.join(SAVE_DIR, "labels")
CLASSES = ["Red Diamond", "Red Square", "Green Diamond", "Green Square"]  # 类别顺序

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_next_counter():
    counter_file = "counter.txt"
    if os.path.exists(counter_file):
        with open(counter_file, "r") as f:
            count = int(f.read().strip())
    else:
        count = 0
    # 写入下一个计数
    with open(counter_file, "w") as f:
        f.write(str(count + 1))
    return count

# -------------------------------------

class MockSCADAScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("工业仿真界面 - 数据集生成器")
        self.resize(1280, 720)
        
        self.COLOR_RED = QColor(220, 50, 50)
        self.COLOR_GREEN = QColor(60, 180, 60)
        self.COLOR_TEXT = QColor(180, 180, 180)
        self.COLOR_BG = QColor(30, 35, 45)
        self.COLOR_GRID = QColor(60, 70, 80)
        self.COLOR_TITLE_BG = QColor(0, 80, 140)
        
        self.targets = []
        self.pipes = []
        self.init_random_data()

        # 使用定时器延迟保存，确保界面完全绘制
        QTimer.singleShot(500, self.auto_save)

    def is_overlap(self, cx, cy, size, existing_targets):
        for _, ex_cx, ex_cy, ex_size in existing_targets:
            dist = ((cx - ex_cx) ** 2 + (cy - ex_cy) ** 2) ** 0.5
            if dist < (size + ex_size) / 2 + 5:
                return True
        return False

    def generate_random_pipe(self):
        pipe_type = random.choice(['horizontal', 'vertical', 'diagonal'])
        w = self.width()
        h = self.height()
        margin = 80
        thickness = random.randint(1, 3)

        if pipe_type == 'horizontal':
            y = random.randint(100, h - margin)
            x1 = random.randint(margin, w // 3)
            x2 = random.randint(w // 2, w - margin)
            return (x1, y, x2, y, thickness)

        elif pipe_type == 'vertical':
            x = random.randint(margin, w - margin)
            y1 = random.randint(100, h // 2)
            y2 = random.randint(h // 2, h - margin)
            return (x, y1, x, y2, thickness)

        else:  # diagonal
            x1 = random.randint(margin, w // 2)
            y1 = random.randint(100, h // 2)
            x2 = random.randint(w // 2, w - margin)
            y2 = random.randint(h // 2, h - margin)
            return (x1, y1, x2, y2, thickness)

    def init_random_data(self):
        # 生成随机目标（5~10个，不重叠）
        num_targets = random.randint(5, 10)
        attempts = 0
        max_attempts = 500
        while len(self.targets) < num_targets and attempts < max_attempts:
            class_id = random.randint(0, 3)
            cx = random.randint(100, self.width() - 100)
            cy = random.randint(120, self.height() - 80)
            size = random.randint(20, 35)
            if not self.is_overlap(cx, cy, size, self.targets):
                self.targets.append((class_id, cx, cy, size))
            attempts += 1

        # 生成随机管线（5~10条）
        num_pipes = random.randint(5, 10)
        self.pipes = [self.generate_random_pipe() for _ in range(num_pipes)]

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 1. 背景
        painter.fillRect(self.rect(), self.COLOR_BG)

        # 2. 顶部标题栏
        title_rect = QRectF(0, 0, self.width(), 55)
        painter.fillRect(title_rect, self.COLOR_TITLE_BG)
        painter.setPen(self.COLOR_TEXT)
        font = QFont("Arial", 12, QFont.Bold)
        painter.setFont(font)
        painter.drawText(QRectF(20, 10, 700, 24), Qt.AlignLeft | Qt.AlignVCenter, "中国三峡 - 柔性直流输电示范工程")

        # 3. 左侧竖排文字
        painter.save()
        painter.translate(40, self.height() / 2)
        painter.rotate(-90)
        painter.setPen(self.COLOR_TEXT)
        font_left = QFont("Arial", 13)
        painter.setFont(font_left)
        painter.drawText(QRectF(-self.height()/2, -20, self.height(), 40), Qt.AlignCenter, "陆上站交流场")
        painter.restore()

        # 4. 网格
        pen_grid = QPen(self.COLOR_GRID, 0.5)
        painter.setPen(pen_grid)
        for x in range(0, self.width(), 40):
            painter.drawLine(int(x), int(60), int(x), int(self.height()))
        for y in range(60, self.height(), 40):
            painter.drawLine(int(0), int(y), int(self.width()), int(y))

        # 5. 随机管线
        for x1, y1, x2, y2, width in self.pipes:
            pen_pipe = QPen(self.COLOR_TEXT, width)
            painter.setPen(pen_pipe)
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        # 6. 核心目标
        for class_id, cx, cy, size in self.targets:
            if class_id in [0, 1]:
                color = self.COLOR_RED
            else:
                color = self.COLOR_GREEN
            pen_outline = QPen(color.lighter(150), 2)
            painter.setPen(pen_outline)
            painter.setBrush(color)
            rect = QRectF(cx - size/2, cy - size/2, size, size)
            if class_id % 2 == 0:
                path = QPainterPath()
                path.moveTo(cx, cy - size/2)
                path.lineTo(cx + size/2, cy)
                path.lineTo(cx, cy + size/2)
                path.lineTo(cx - size/2, cy)
                path.closeSubpath()
                painter.drawPath(path)
            else:
                painter.drawRect(rect)

    def auto_save(self):
        """自动保存截图和标注文件"""
        # 确保目录存在
        ensure_dir(IMAGE_DIR)
        ensure_dir(LABEL_DIR)

        # 获取当前计数
        idx = get_next_counter()
        img_path = os.path.join(IMAGE_DIR, f"{idx}.png")
        label_path = os.path.join(LABEL_DIR, f"{idx}.txt")

        # 截图
        pixmap = self.grab()  # 捕获窗口内容
        pixmap.save(img_path, "PNG")

        # 生成 YOLO 格式标注
        img_w = self.width()
        img_h = self.height()
        lines = []
        for class_id, cx, cy, size in self.targets:
            # 归一化坐标
            x_center = cx / img_w
            y_center = cy / img_h
            # 宽度和高度：目标的外接正方形边长
            w = size / img_w
            h = size / img_h
            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

        with open(label_path, "w") as f:
            f.writelines(lines)

        # 输出信息
        print(f"✅ 已保存：{img_path}")
        print(f"✅ 已保存：{label_path}")
        print(f"目标数量：{len(self.targets)}，管线数量：{len(self.pipes)}")

        # 关闭窗口（可选，方便连续运行）
        self.close()


def main():
    app = QApplication(sys.argv)
    font = QFont("Microsoft YaHei", 10)
    app.setFont(font)
    window = MockSCADAScreen()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()