"""
用户界面模块：生成固定的模拟UI界面
"""
import sys
import random
import yaml
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QWidget, QDialog, QVBoxLayout,
                             QHBoxLayout, QLabel, QLineEdit, QSpinBox,
                             QPushButton, QMessageBox)
from PyQt5.QtCore import Qt, QRectF, QTimer
from PyQt5.QtGui import QPainter, QColor, QFont, QPainterPath, QPen, QImage, QFontMetrics
from model_svc import ModelService
from strategy_engine import StrategyEngine
from action_executor import ActionExecutor
from locators import QtLocator, OcrLocator



class EditDialog(QDialog):
    """属性编辑对话框"""
    def __init__(self, parent, target_data, target_idx, is_locked):
        """
        target_data: [class_id, cx, cy, size, id_str, status, note]
        """
        super().__init__(parent)
        self.setWindowTitle("编辑属性")
        self.setFixedSize(320, 260)
        self.target_data = target_data  # 引用原始数据，修改后直接更新
        self.target_idx = target_idx  # 记录当前图元索引，用于切换加锁状态
        # 保存父窗口的原始锁状态（用于取消时恢复）
        self.original_is_locked = is_locked
        # 暂存锁状态变更意图：None表示不变，True表示需要切换
        self.pending_lock_change = None

        layout = QVBoxLayout(self)

        # ID
        hbox_id = QHBoxLayout()
        hbox_id.addWidget(QLabel("ID:"))
        self.edit_id = QLineEdit(str(target_data[4]))
        hbox_id.addWidget(self.edit_id)
        layout.addLayout(hbox_id)

        # Status
        hbox_status = QHBoxLayout()
        hbox_status.addWidget(QLabel("Status:"))
        self.spin_status = QSpinBox()
        self.spin_status.setRange(-10000, 10000)  # 可根据需要调整
        self.spin_status.setValue(target_data[5])
        hbox_status.addWidget(self.spin_status)
        layout.addLayout(hbox_status)

        # Note
        hbox_note = QHBoxLayout()
        hbox_note.addWidget(QLabel("Note:"))
        self.edit_note = QLineEdit(str(target_data[6]))
        hbox_note.addWidget(self.edit_note)
        layout.addLayout(hbox_note)

        # 设置输入框对象名称
        self.edit_id.setObjectName("id_input")
        self.spin_status.setObjectName("status_spin")
        self.edit_note.setObjectName("note_input")

        # 按钮
        btn_layout = QHBoxLayout()
        btn_ok = QPushButton("确定")
        btn_cancel = QPushButton("取消")
        btn_ok.setObjectName("ok_button")# 设置确认按钮对象名称
        btn_cancel.setObjectName("cancel_button")# 设置取消按钮对象名称
        btn_layout.addStretch()
        btn_layout.addWidget(btn_ok)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)

        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)

        # 根据当前图元锁定状态显示加锁/解锁按钮，并在点击后实时刷新UI
        self.btn_lock_toggle = QPushButton("解锁" if is_locked else "加锁")
        self.btn_lock_toggle.setObjectName("lock_toggle_button")# 设置加锁/解锁按钮对象名称
        layout.addWidget(self.btn_lock_toggle)
        self.btn_lock_toggle.clicked.connect(self.toggle_lock)

    def toggle_lock(self):
        """切换暂存的锁状态，仅更新按钮文字，不修改父窗口"""
        if self.pending_lock_change is None:
            # 第一次点击：相对于原始状态切换
            self.pending_lock_change = True
        else:
            # 再次点击：取消之前的暂存
            self.pending_lock_change = not self.pending_lock_change
        # 根据暂存状态计算按钮应显示的文字
        if self.pending_lock_change is True: # 如果 pending_lock_change 为 True
            final_locked = not self.original_is_locked # 表示最终状态与原始状态相反
        else: # 否则与原始状态相同
            final_locked = self.original_is_locked
        self.btn_lock_toggle.setText("解锁" if final_locked else "加锁")

    def accept(self):
        # 验证并更新数据
        try:
            new_id = self.edit_id.text().strip()
            new_status = self.spin_status.value()
            new_note = self.edit_note.text().strip()
            # 更新原始数据（注意是引用）
            self.target_data[4] = new_id
            self.target_data[5] = new_status
            self.target_data[6] = new_note

            # 如果有锁的状态变更，则应用
            if self.pending_lock_change:
                # 调用父窗口的 toggle_target_lock 一次，它会翻转当前锁状态
                # 由于我们暂存的是“需要切换一次”，所以正好匹配
                self.parent().toggle_target_lock(self.target_idx)

            super().accept()
        except Exception as e:
            QMessageBox.warning(self, "错误", f"数据无效：{str(e)}")


class DemoWithLines(QWidget):
    def __init__(self):
        super().__init__()
        # 主模块
        self.model_svc = None # 视觉服务
        #self.locator = QtLocator()# 定位器（为执行器服务）
        self.locator = OcrLocator(offset_map={# 定位器（为执行器服务）
            "id_input": (80, 0), # 文字中心向右偏移80px到输入框
            "status_spin": (150, 0),
            "note_input": (100, 0),
            "ok_button": (0, 0), # 不偏移，直接点击文字中心
            "cancel_button": (0, 0)
        })
        self.action_executor = ActionExecutor(locator = self.locator)# 执行器
        self.strategy_engine = StrategyEngine()# 策略引擎

        random.seed(42)
        self.setWindowTitle("mock UI")
        self.resize(1280, 720)

        self.COLOR_RED = QColor(220, 50, 50)
        self.COLOR_GREEN = QColor(60, 180, 60)
        self.COLOR_TEXT = QColor(0, 0, 0)
        self.COLOR_BG = QColor(255, 255, 255)
        self.COLOR_LINE = self.COLOR_TEXT
        self.label_font = QFont("Arial", 10)
        self.COLOR_DETECT_DOT = QColor(0, 0, 255)  # 标记点

        # 生成12个target，每个target为 [cls_id, cx, cy, size, id, status, note]
        # 随机选2个target带锁
        self.targets = self._generate_targets()
        def _assign_locks():
            """随机选择两个图元加上锁，返回索引列表"""
            if len(self.targets) < 2:
                return []
            return random.sample(range(len(self.targets)), 2)
        self.locked_indices = _assign_locks() # 加锁图元的索引
        self.label_boxes = self._place_labels() # 生成编号位置，保证编号避开图元
        self.lines = self._generate_lines()

        #按钮和其他相关成员
        # detect按钮
        self.btn_detect = QPushButton("detect", self)
        self.btn_detect.clicked.connect(self.on_detect_clicked)
        # detect按钮位置（右上角）
        self.btn_detect.move(self.width() - 100, 10)
        self.btn_detect.resize(100, 32)
        # unlock 按钮
        self.btn_unlock = QPushButton("unlock", self)
        self.btn_unlock.clicked.connect(self.on_unlock_clicked)
        # unlock按钮位置
        self.btn_unlock.move(self.width() - 100, 52)
        self.btn_unlock.resize(100, 32)
        # lock by number按钮
        self.btn_lock_by_number = QPushButton("lock by number", self)
        self.btn_lock_by_number.clicked.connect(self.on_lock_by_number_clicked)
        # lock by number按钮位置
        self.btn_lock_by_number.move(self.width()-200, 94)
        self.btn_lock_by_number.resize(200, 32)
        # fill by number按钮
        self.btn_fill_by_number = QPushButton("fill by number", self)
        self.btn_fill_by_number.clicked.connect(self.on_fill_by_number_clicked)
        # fill by number按钮位置
        self.btn_fill_by_number.move(self.width()-200, 136)
        self.btn_fill_by_number.resize(200, 32)
        # 类别列表
        with open("models/config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        self.cls_list = config["cls_list"]
        # 存储检测到的中心点（用于绘制检测点）
        self.detected_points = []  # 每个元素为 (x_pixel, y_pixel)

    def _generate_targets(self):
        targets = []
        # 强制四个类别各一个
        forced = [[0, 150, 200, 28, "000001", 0, "old note"], # 使用三种固定编号
                  [1, 380, 320, 26, "000502", 0, "old note"],
                  [2, 650, 480, 27, "000083", 0, "old note"],
                  [3, 900, 260, 29, "000001", 0, "old note"]]
        targets.extend(forced)

        # 剩余8个随机生成
        while len(targets) < 12:
            class_id = random.randint(0, 3)
            cx = random.randint(80, 1200)
            cy = random.randint(80, 640)
            size = random.randint(22, 33)
            overlap = False
            for t in targets:
                _, ex_cx, ex_cy, ex_size, *_ = t
                dist = ((cx - ex_cx)**2 + (cy - ex_cy)**2)**0.5
                if dist < (size + ex_size)/2 + 8:
                    overlap = True
                    break
            if not overlap:
                id_list = ["000001", "000502", "000083"] # 12个图元只分配这三种编号，且每种至少出现2次
                targets.append([class_id, cx, cy, size, id_list[len(targets) % len(id_list)], 0, "old note"])
        return targets

    def _place_labels(self):
        """生成编号矩形框，避免编号与任意图元重合"""
        placed = []
        boxes = []
        img_w, img_h = self.width(), self.height()
        font_metrics = QFontMetrics(self.label_font)
        label_padding = 3

        target_rects = []
        for t in self.targets:
            _, cx, cy, size, *_ = t
            half = size / 2 + 5
            target_rects.append((cx - half, cy - half, cx + half, cy + half))

        for t in self.targets:
            _, cx, cy, size, label, *_ = t
            tw = font_metrics.horizontalAdvance(label)
            th = font_metrics.height()

            candidates = [
                (cx - tw/2, cy - size/2 - th - 10),
                (cx - tw/2, cy + size/2 + 10),
                (cx - size/2 - tw - 8, cy - th/2),
                (cx + size/2 + 8, cy - th/2),
            ]

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
                if tx < 10 or tx + tw > img_w - 10:
                    continue
                if ty < 65 or ty + th > img_h - 10:
                    continue

                lx_min, ly_min = tx - label_padding, ty - label_padding
                lx_max, ly_max = tx + tw + label_padding, ty + th + label_padding
                overlap_with_target = any(
                    not (lx_max <= rx_min or lx_min >= rx_max or ly_max <= ry_min or ly_min >= ry_max)
                    for rx_min, ry_min, rx_max, ry_max in target_rects
                )
                overlap_with_label = any(
                    not (tx + tw + label_padding <= px - label_padding or
                         tx - label_padding >= px + pw + label_padding or
                         ty + th + label_padding <= py - label_padding or
                         ty - label_padding >= py + ph + label_padding)
                    for px, py, pw, ph in placed
                )
                if not overlap_with_target and not overlap_with_label:
                    best = (tx, ty)
                    break

            tx, ty = best
            placed.append((tx, ty, tw, th))
            boxes.append((tx, ty, tw, th))

        return boxes

    def _generate_lines(self):
        lines = []
        num_lines = random.randint(8, 14)
        max_tries = num_lines * 20  # 避免因为管线碰到编号而生成不足时无限循环
        tries = 0

        while len(lines) < num_lines and tries < max_tries:
            tries += 1
            line_type = random.choice(['h', 'v', 'd'])
            w, h = self.width(), self.height()
            if line_type == 'h':
                y = random.randint(80, h - 40)
                x1 = random.randint(40, w//3)
                x2 = random.randint(w//2, w-40)
                if not self._line_intersects_any_label(x1, y, x2, y):
                    lines.append((x1, y, x2, y))
            elif line_type == 'v':
                x = random.randint(40, w-40)
                y1 = random.randint(80, h//2)
                y2 = random.randint(h//2, h-40)
                if not self._line_intersects_any_label(x, y1, x, y2):
                    lines.append((x, y1, x, y2))
            else:
                x1 = random.randint(40, w//2)
                y1 = random.randint(80, h//2)
                x2 = random.randint(w//2, w-40)
                y2 = random.randint(h//2, h-40)
                if not self._line_intersects_any_label(x1, y1, x2, y2):
                    lines.append((x1, y1, x2, y2))
        return lines

    def _line_intersects_any_label(self, x1, y1, x2, y2):
        """判断管线是否与任意编号框相交"""
        label_padding = 3
        return any(
            self.line_intersects_rect(x1, y1, x2, y2,
                                      rx - label_padding, ry - label_padding,
                                      rw + label_padding * 2, rh + label_padding * 2)
            for rx, ry, rw, rh in self.label_boxes
        )

    @staticmethod
    def line_intersects_rect(x1, y1, x2, y2, rx, ry, rw, rh):
        """判断线段是否与矩形相交"""
        def code(x, y):
            c = 0
            if x < rx:          c |= 1
            elif x > rx + rw:   c |= 2
            if y < ry:          c |= 4
            elif y > ry + rh:   c |= 8
            return c
        c1 = code(x1, y1)
        c2 = code(x2, y2)
        if c1 & c2 != 0:
            return False
        return True

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), self.COLOR_BG)

        # 绘制管线干扰
        pen_line = QPen(self.COLOR_LINE, 1)
        painter.setPen(pen_line)
        for x1, y1, x2, y2 in self.lines:
            painter.drawLine(x1, y1, x2, y2)

        # 绘制所有目标
        for t in self.targets:
            class_id, cx, cy, size = t[:4]
            color = self.COLOR_RED if class_id in [0, 1] else self.COLOR_GREEN
            pen_outline = QPen(color.lighter(150), 2)
            painter.setPen(pen_outline)
            painter.setBrush(color)

            rect = QRectF(cx - size/2, cy - size/2, size, size)
            if class_id % 2 == 0:  # 菱形
                path = QPainterPath()
                path.moveTo(cx, cy - size/2)
                path.lineTo(cx + size/2, cy)
                path.lineTo(cx, cy + size/2)
                path.lineTo(cx - size/2, cy)
                path.closeSubpath()
                painter.drawPath(path)
            else:                  # 方形
                painter.drawRect(rect)

        #绘制检测点及类别文字
        if self.detected_points:
            painter.setPen(Qt.NoPen)  # 无边框
            painter.setBrush(self.COLOR_DETECT_DOT)
            dot_radius = 5
            for (px, py, cls) in self.detected_points:
                # 绘制检测点
                painter.drawEllipse(px - dot_radius, py - dot_radius, dot_radius * 2, dot_radius * 2)
                # 绘制类别文字
                painter.setPen(QColor(0, 0, 255))
                font = QFont("Microsoft YaHei", 8)
                painter.setFont(font)
                cls_name=self.cls_list[cls]
                # 类别文字放在黄点右下方偏移 (6, 6) 像素，避免遮盖
                painter.drawText(px + 6, py + 6, cls_name)

        # 绘制锁（橙色矩形）
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 255, 0))  # 黄色
        for idx in self.locked_indices:
            t = self.targets[idx]
            _, cx, cy, size = t[:4]
            lock_size = size * 0.45  # 锁的大小约为图元的0.45
            lock_rect = QRectF(cx - lock_size/2, cy - lock_size/2, lock_size, lock_size)
            painter.drawRect(lock_rect)

        # 绘制图元编号
        painter.setFont(self.label_font)
        font_metrics = QFontMetrics(self.label_font)
        painter.setPen(self.COLOR_TEXT)
        for idx, (tx, ty, tw, th) in enumerate(self.label_boxes):
            painter.drawText(int(tx), int(ty + font_metrics.ascent()), self.targets[idx][4])

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        pos = event.pos()
        x, y = pos.x(), pos.y()

        # 从后往前遍历（上层图形优先，实际无层级，顺序无所谓）
        for idx, t in enumerate(self.targets):
            class_id, cx, cy, size = t[:4]
            half = size / 2.0
            if class_id % 2 == 0:  # 菱形：使用曼哈顿距离判断
                # 菱形区域：|x-cx|/half + |y-cy|/half <= 1
                dx = abs(x - cx)
                dy = abs(y - cy)
                if dx / half + dy / half <= 1.0:
                    self._open_edit_dialog(idx)
                    return
            else:  # 方形
                if (cx - half <= x <= cx + half) and (cy - half <= y <= cy + half):
                    self._open_edit_dialog(idx)
                    return

    def _open_edit_dialog(self, idx):
        target = self.targets[idx]
        # 计算对话框位置：在图形右侧稍上方
        cx, cy, size = target[1], target[2], target[3]
        dialog_x = cx + size//2 + 15
        dialog_y = cy - 70
        # 防止超出屏幕
        if dialog_x + 320 > self.width():
            dialog_x = cx - size//2 - 335
        if dialog_y < 0:
            dialog_y = 20
        if dialog_y + 270 > self.height():
            dialog_y = self.height() - 275

        dialog = EditDialog(self, target, idx, idx in self.locked_indices)
        dialog.setModal(False) # 改为非模态
        dialog.show() # 显示但不阻塞

    def toggle_target_lock(self, idx):
        # 对话框中的加锁/解锁按钮调用这里，更新锁列表后立即重绘
        if idx in self.locked_indices:
            self.locked_indices.remove(idx)
            is_locked = False
        else:
            self.locked_indices.append(idx)
            is_locked = True
        self.update()
        
    @staticmethod
    def associate_one_to_one(points, texts):
        """
        贪心匹配：将图元与文本一一对应，每个文本只分配给最近的图元。
        输入：
            points: list of [x, y, cls, text]  图元列表，text初始为空
            texts: list of [x, y, cls, text]  文本列表，text为OCR结果
        输出：
            list of [x, y, cls, text]  关联后的图元信息，供策略引擎使用
        """
        # 1. 如果没有图元或文本，直接返回空列表
        if not points or not texts:
            return []
        # 2. 构建所有可能的配对 (距离dist, 图元索引point_idx, 文本索引text_idx)
        pairs = []
        for point_idx, point in enumerate(points):
            point_x, point_y = point[0], point[1] # 图元中心坐标
            for text_idx, text in enumerate(texts):
                text_x, text_y = text[0], text[1] # 文本中心坐标
                dist = ((point_x - text_x) ** 2 + (point_y - text_y) ** 2) ** 0.5 # 欧氏距离
                pairs.append((dist, point_idx, text_idx))
        # 3. 按距离从小到大排序（距离越近越优先匹配）
        pairs.sort(key=lambda x: x[0])
        # 4. 记录哪些图元和文本已经被分配
        assigned_point = set()
        assigned_text = set()
        # 5. 准备结果列表，初始化为 [x, y, cls, ""]（text先占位）
        associated = [[point[0], point[1], point[2], ""] for point in points]
        # 6. 遍历排序后的配对，进行贪心分配
        for dist, point_idx, text_idx in pairs:
            # 如果该图元和该文本都未被占用，则建立关联
            if point_idx not in assigned_point and text_idx not in assigned_text:
                associated[point_idx][3] = texts[text_idx][3] # 将文本字符串填入图元
                assigned_point.add(point_idx) # 标记图元已分配
                assigned_text.add(text_idx) # 标记文本已分配
        # 7. 返回关联后的结果（格式为 [x, y, cls, text]）
        return associated

    def on_detect_clicked(self):
        """
        点击detect按钮后，自动检测所有图元，并保存到策略引擎
        包括：截图 -> 推理 -> 获取中心点 -> 重绘
        """
        print("正在推理...")
        # 加载视觉服务
        if self.model_svc is None:
            self.model_svc = ModelService()
        self.btn_detect.hide()# 截图前隐藏按钮
        self.btn_unlock.hide()
        self.btn_lock_by_number.hide()
        self.btn_fill_by_number.hide()
        QApplication.processEvents()# 刷新页面
        # 截图当前窗口（得到QPixmap）
        pixmap = self.grab()  # 获取整个窗口的图像
        self.btn_detect.show()# 截图后恢复按钮
        self.btn_unlock.show()
        self.btn_lock_by_number.show()
        self.btn_fill_by_number.show()
        # 转换为Qimage并统一为RGBA非预乘格式
        qimage = pixmap.toImage().convertToFormat(QImage.Format_RGBA8888)
        # 将QImage转换为PIL Image（使用内存缓冲区）
        buffer = qimage.bits().asstring(qimage.byteCount())
        pil_image = Image.frombuffer("RGBA", (qimage.width(), qimage.height()), buffer, "raw", "RGBA", 0, 1)
        pil_image = pil_image.convert("RGB")
        # 使用视觉服务推理出结果
        targets = self.model_svc.predict(pil_image)# 每个目标的信息，即包含[x, y, cls_idx, text]的列表
        # 分离图元和文本
        points=[] # 图元信息
        texts=[] # 文本信息
        for target in targets:
            if target[2] == 8: # 若cls为Text
                texts.append(target)
            else:
                points.append(target)

        # 图元关联文本
        associated = self.associate_one_to_one(points, texts)
        self.strategy_engine.update_points(associated)# 把associated作为points保存到策略引擎

        print(f"检测到 {len(associated)} 个图元")# 在控制台打印检测结果

        # # （debug）绘制检测点和类别文字
        # self.detected_points = points
        # self.update()# 触发paintEvent

    def on_unlock_clicked(self):
        """点击unlock按钮后，自动解锁所有已加锁图元"""
        coords=self.strategy_engine.get_all_locked_point()
        if coords is None or len(coords)==0:
            print("未检测到locked目标")
            return
        print(f"检测到{len(coords)}个locked目标")
        # 递归处理队列
        def process_next(idx):
            if idx >= len(coords):
                print("全部解锁完成")
                return
            x, y = coords[idx]
            print(f"正在解锁第 {idx+1} 个目标，坐标 ({x}, {y})")
            # 第一步：点击目标，弹出编辑对话框
            self.action_executor.click_at_window_point(self, x, y)
            # 第二步：等待对话框出现（约800ms），点击“锁”按钮切换状态
            QTimer.singleShot(800, lambda: (
                self.action_executor.click_lock_toggle_button(),
                # 第三步：等待锁状态生效（约500ms），点击“确定”关闭对话框
                QTimer.singleShot(500, lambda: (
                    self.action_executor.click_ok_button(),
                    # 第四步：等待对话框关闭（约500ms），处理下一个
                    QTimer.singleShot(500, lambda: process_next(idx + 1))
                ))
            ))
        # 从第一个开始
        process_next(0)

    def on_lock_by_number_clicked(self):
        """点击lock by number按钮后，自动加锁所有指定number且未加锁的图元"""
        assign = "000001" # 指定number
        # 从策略引擎得到所有number为assign且未加锁的图元坐标
        coords=self.strategy_engine.get_unlocked_point_by_number(assign)
        if coords is None or len(coords)==0:
            print(f"未检测到number为{assign}且未加锁的目标")
            return
        print(f"检测到{len(coords)}个number为{assign}且未加锁的目标")
        # 递归处理队列
        def process_next(idx):
            if idx >= len(coords):
                print("全部加锁完成")
                return
            x, y = coords[idx]
            print(f"正在加锁第 {idx+1} 个目标，坐标 ({x}, {y})")
            # 第一步：点击目标，弹出编辑对话框
            self.action_executor.click_at_window_point(self, x, y)
            # 第二步：等待对话框出现（约800ms），点击“锁”按钮切换状态
            QTimer.singleShot(800, lambda: (
                self.action_executor.click_lock_toggle_button(),
                # 第三步：等待锁状态生效（约500ms），点击“确定”关闭对话框
                QTimer.singleShot(500, lambda: (
                    self.action_executor.click_ok_button(),
                    # 第四步：等待对话框关闭（约500ms），处理下一个
                    QTimer.singleShot(500, lambda: process_next(idx + 1))
                ))
            ))
        # 从第一个开始
        process_next(0)

    def on_fill_by_number_clicked(self):
        """点击fill by number按钮后，自动为指定number的图元填充新属性"""
        assign = "000001" # 指定number
        # 从策略引擎得到所有number为assign的图元坐标
        coords=self.strategy_engine.get_point_by_number(assign)
        if coords is None or len(coords)==0:
            print(f"未检测到number为{assign}的图元")
            return
        print(f"检测到{len(coords)}个number为{assign}的图元")
         # 递归处理队列
        def process_next(idx):
            if idx >= len(coords):
                print("全部填充完成")
                return
            x, y = coords[idx]
            print(f"正在填充第 {idx+1} 个图元，坐标 ({x}, {y})")
            # 找到当前填充的图元
            target = min(
                self.targets,
                key=lambda t: (t[1] - x) ** 2 + (t[2] - y) ** 2
            )
            # 获取当前图元的旧属性
            old_values = {
                "id_input": target[4],
                "status_spin": target[5],
                "note_input": target[6],
            }
            # 从策略引擎根据旧属性得到每个图元属性框的新值
            new_values = self.strategy_engine.generate_new_values(old_values)
            # 第一步：点击目标，弹出编辑对话框，自动填充
            self.action_executor.click_at_window_point(self, x, y)
            # 第二步：等待对话框出现后，填充
            QTimer.singleShot(800, lambda: (
                self.action_executor.fill_dialog_input(new_values),
                # 第三步：点击“确定”关闭对话框
                QTimer.singleShot(500, lambda: (
                    self.action_executor.click_ok_button(),
                    # 第四步：等待对话框关闭，处理下一个
                    QTimer.singleShot(500, lambda: process_next(idx + 1))
                ))
            ))
        # 从第一个开始
        process_next(0)


def main():
    app = QApplication(sys.argv)
    font = QFont("Microsoft YaHei", 10)
    app.setFont(font)
    window = DemoWithLines()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()