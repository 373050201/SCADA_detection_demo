"""
生成模拟的UI，用于训练模型
"""
import sys
import os
import random
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import Qt, QRectF, QTimer
from PyQt5.QtGui import QPainter, QColor, QFont, QPainterPath, QPen, QFontMetrics



# ---------- 自动保存的相关配置 ----------
SAVE_DIR =  os.path.join("datasets", "SCADA_yolo")        # 数据集根目录
IMAGE_DIR = os.path.join(SAVE_DIR, "images")
LABEL_DIR = os.path.join(SAVE_DIR, "labels")
CLASSES = ["Red Diamond", "Red Square", "Green Diamond", "Green Square", 
           "Red Diamond Locked", "Red Square Locked", "Green Diamond Locked", "Green Square Locked", 
           "Text"]  # 类别顺序

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
        self.COLOR_TEXT = QColor(0, 0, 0)
        self.COLOR_BG = QColor(255, 255, 255)
        self.COLOR_TITLE_BG = QColor(0, 80, 140)
        self.label_font = QFont("Arial", 10)  # 编号绘制、避让、标注共用同一字体
        
        self.targets = []
        self.pipes = []
        self.init_random_data() # 生成图元targets
        self.locked_indices = self._pick_two_locks() # 带锁图元的索引
        self.text_labels = self._generate_text_labels() # 每个图元的六位数编号
        self.label_boxes = self._place_labels() # 编号的矩形框(tx, ty, tw, th)
        self._generate_pipes() # 生成管线

        # 使用定时器延迟保存，确保界面完全绘制
        QTimer.singleShot(500, self.auto_save)

    def _pick_two_locks(self):
        if len(self.targets) < 2:
            return []
        return random.sample(range(len(self.targets)), 2)

    def _generate_text_labels(self):
        """为每个图元生成一个随机六位数编号"""
        labels = []
        for _ in self.targets:
            # 生成 100000~999999 之间的整数，转为字符串
            num = random.randint(100000, 999999)
            labels.append(str(num))
        return labels

    def _place_labels(self):
        """贪心放置每个图元的编号标签，返回 [(tx, ty, tw, th)]，避开所有图元"""
        placed = []  # 已放置的标签矩形 (tx, ty, tw, th)
        boxes = []
        img_w, img_h = self.width(), self.height()
        font_metrics = QFontMetrics(self.label_font)  # 使用真实字体尺寸计算编号框
        label_padding = 3  # 避让时给编号四周留出安全距离

        # 所有图元的矩形（扩大5像素作为安全间距）
        target_rects = []
        for _, cx, cy, size in self.targets:
            half = size / 2 + 5   # 加5像素缓冲
            target_rects.append((cx - half, cy - half, cx + half, cy + half))

        for idx, (class_id, cx, cy, size) in enumerate(self.targets):
            label = self.text_labels[idx]
            tw = font_metrics.horizontalAdvance(label)  # 使用真实绘制宽度
            th = font_metrics.height()                  # 使用真实绘制高度

            candidates = [
                (cx - tw/2, cy - size/2 - th - 10),       # 正上
                (cx - tw/2, cy + size/2 + 10),            # 正下
                (cx - size/2 - tw - 8, cy - th/2),        # 正左（垂直居中）
                (cx + size/2 + 8, cy - th/2),             # 正右（垂直居中）
            ]
            #增加更多候选位置
            for gap in range(20, 161, 20):
                candidates.extend([
                    (cx - tw/2 - gap, cy - size/2 - th - gap),
                    (cx - tw/2 + gap, cy - size/2 - th - gap),
                    (cx - tw/2 - gap, cy + size/2 + gap),
                    (cx - tw/2 + gap, cy + size/2 + gap),
                    (cx - size/2 - tw - gap, cy - th/2 - gap),
                    (cx - size/2 - tw - gap, cy - th/2 + gap),
                    (cx + size/2 + gap, cy - th/2 - gap),
                    (cx + size/2 + gap, cy - th/2 + gap),
                ])

            best = None
            for tx, ty in candidates:
                # 边界检查
                if tx < 10 or tx + tw > img_w - 10:
                    continue
                if ty < 65 or ty + th > img_h - 10:
                    continue

                # 检查与所有图元矩形是否相交
                overlap_with_target = False
                for rx_min, ry_min, rx_max, ry_max in target_rects:
                    # 标签矩形范围，避让判断使用带安全边距的真实编号框
                    lx_min, ly_min = tx - label_padding, ty - label_padding
                    lx_max, ly_max = tx + tw + label_padding, ty + th + label_padding
                    if not (lx_max <= rx_min or lx_min >= rx_max or ly_max <= ry_min or ly_min >= ry_max):
                        overlap_with_target = True
                        break
                if overlap_with_target:
                    continue

                # 检查与已放置标签的重叠
                overlap_with_label = False
                for px, py, pw, ph in placed:
                    # 编号之间也按安全边距避让
                    if not (tx + tw + label_padding <= px - label_padding or
                            tx - label_padding >= px + pw + label_padding or
                            ty + th + label_padding <= py - label_padding or
                            ty - label_padding >= py + ph + label_padding):
                        overlap_with_label = True
                        break
                if not overlap_with_label:
                    best = (tx, ty)
                    break

            if best is None:
                # 最后兜底也只在不与任意图元/编号重合的位置里选择。
                for ty_try in range(65, int(img_h - th - 10), 5):
                    for tx_try in range(10, int(img_w - tw - 10), 5):
                        # 兜底位置同样使用带安全边距的真实编号框
                        lx_min, ly_min = tx_try - label_padding, ty_try - label_padding
                        lx_max, ly_max = tx_try + tw + label_padding, ty_try + th + label_padding
                        overlap_with_target = any(
                            not (lx_max <= rx_min or lx_min >= rx_max or ly_max <= ry_min or ly_min >= ry_max)
                            for rx_min, ry_min, rx_max, ry_max in target_rects
                        )
                        overlap_with_label = any(
                            not (tx_try + tw + label_padding <= px - label_padding or
                                 tx_try - label_padding >= px + pw + label_padding or
                                 ty_try + th + label_padding <= py - label_padding or
                                 ty_try - label_padding >= py + ph + label_padding)
                            for px, py, pw, ph in placed
                        )
                        if not overlap_with_target and not overlap_with_label:
                            best = (tx_try, ty_try)
                            break
                    if best is not None:
                        break

            tx, ty = best
            placed.append((tx, ty, tw, th))
            boxes.append((tx, ty, tw, th))

        return boxes

    def is_overlap(self, cx, cy, size, existing_targets):
        for _, ex_cx, ex_cy, ex_size in existing_targets:
            dist = ((cx - ex_cx) ** 2 + (cy - ex_cy) ** 2) ** 0.5
            if dist < (size + ex_size) / 2 + 5:
                return True
        return False

    def generate_random_pipe(self):
        """生成一条不与任何编号矩形相交的管线，最多尝试 20 次"""
        max_tries = 20
        for _ in range(max_tries):
            pipe_type = random.choice(['horizontal', 'vertical', 'diagonal'])
            w = self.width()
            h = self.height()
            margin = 80
            thickness = random.randint(1, 3)

            if pipe_type == 'horizontal':
                y = random.randint(100, h - margin)
                x1 = random.randint(margin, w // 3)
                x2 = random.randint(w // 2, w - margin)
                x1, y1, x2, y2 = x1, y, x2, y
            elif pipe_type == 'vertical':
                x = random.randint(margin, w - margin)
                y1 = random.randint(100, h // 2)
                y2 = random.randint(h // 2, h - margin)
                x1, y1, x2, y2 = x, y1, x, y2
            else:  # diagonal
                x1 = random.randint(margin, w // 2)
                y1 = random.randint(100, h // 2)
                x2 = random.randint(w // 2, w - margin)
                y2 = random.randint(h // 2, h - margin)

            # 检查是否与任何编号矩形相交，按管线线宽扩展编号矩形，避免编号与管线边缘重合。
            # 修改：按管线线宽和编号安全边距扩展编号矩形，避免编号与管线重合。
            label_padding = 3
            intersect = any(
                self.line_intersects_rect(x1, y1, x2, y2,
                                          rx - thickness - label_padding,
                                          ry - thickness - label_padding,
                                          rw + (thickness + label_padding) * 2,
                                          rh + (thickness + label_padding) * 2)
                for rx, ry, rw, rh in self.label_boxes
            )
            if not intersect:
                return (x1, y1, x2, y2, thickness)
        return None  # 实在生成不了就跳过

    def init_random_data(self):
        # 只生成 targets
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

    def _generate_pipes(self):
        """生成随机管线"""
        num_pipes = random.randint(5, 10)
        self.pipes = []
        for _ in range(num_pipes):
            pipe = self.generate_random_pipe()
            if pipe is not None:
                self.pipes.append(pipe)

    @staticmethod
    def line_intersects_rect(x1, y1, x2, y2, rx, ry, rw, rh):
        """
        判断线段 (x1,y1)-(x2,y2) 是否与矩形 (rx,ry,rw,rh) 相交
        使用 Cohen-Sutherland 编码的简化版：若两端点都在矩形同一侧，则不相交
        """
        # 计算端点的区域码
        def code(x, y):
            c = 0
            if x < rx:          c |= 1  # left
            elif x > rx + rw:   c |= 2  # right
            if y < ry:          c |= 4  # top
            elif y > ry + rh:   c |= 8  # bottom
            return c
        c1 = code(x1, y1)
        c2 = code(x2, y2)
        # 如果按位与不为0，说明在同一侧外部，肯定不相交
        if c1 & c2 != 0:
            return False
        # 否则认为可能相交（保守判断，即使恰好擦边也视为相交，重试即可）
        return True

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

        # 4. 随机管线
        for x1, y1, x2, y2, width in self.pipes:
            pen_pipe = QPen(self.COLOR_TEXT, width)
            painter.setPen(pen_pipe)
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        # 5. 核心目标
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

        # 6. 绘制黄色锁
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 255, 0))   # 纯黄色
        for idx in self.locked_indices:
            if idx >= len(self.targets):
                continue
            _, cx, cy, size = self.targets[idx]
            lock_size = size * 0.45
            lock_rect = QRectF(cx - lock_size/2, cy - lock_size/2, lock_size, lock_size)
            painter.drawRect(lock_rect)

        # 7. 绘制图元编号
        painter.setFont(self.label_font)  # 绘制字体与避让/标注计算保持一致
        font_metrics = QFontMetrics(self.label_font)  # 用真实字体基线绘制编号
        painter.setPen(QColor(0, 0, 0))  # 黑色
        for idx, (tx, ty, tw, th) in enumerate(self.label_boxes):
            label = self.text_labels[idx]
            # 绘制文本（注意 drawText(x, y, str) 的 y 是基线，所以用 ty+th 作为基线）
            painter.drawText(int(tx), int(ty + font_metrics.ascent()), label)

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
        for idx, (class_id, cx, cy, size) in enumerate(self.targets):
            # 若该图元带锁，类别偏移+4
            if idx in self.locked_indices:
                class_id+=4
            # 归一化坐标
            x_center = cx / img_w
            y_center = cy / img_h
            # 宽度和高度：目标的外接正方形边长
            w = size / img_w
            h = size / img_h
            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

            # 文本区域标注
            tx, ty, tw, th = self.label_boxes[idx]
            text_cx = tx + tw / 2
            text_cy = ty + th / 2
            text_norm_cx = text_cx / img_w
            text_norm_cy = text_cy / img_h
            text_norm_w = tw / img_w
            text_norm_h = th / img_h
            lines.append(f"8 {text_norm_cx:.6f} {text_norm_cy:.6f} {text_norm_w:.6f} {text_norm_h:.6f}\n")

        with open(label_path, "w") as f:
            f.writelines(lines)

        # 输出信息
        print(f"✅ 已保存：{img_path}")
        print(f"✅ 已保存：{label_path}")
        print(f"图元数量：{len(self.targets)}，管线数量：{len(self.pipes)}")

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