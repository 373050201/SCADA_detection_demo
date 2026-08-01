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
from PyQt5.QtGui import QPainter, QColor, QFont, QPainterPath, QPen, QImage
from model_svc import ModelService
from strategy_engine import StrategyEngine
from action_executor import ActionExecutor
from locators import QtLocator, OcrLocator



class EditDialog(QDialog):
    """属性编辑对话框"""
    def __init__(self, parent, target_data):
        """
        target_data: [class_id, cx, cy, size, id_str, status, note]
        """
        super().__init__(parent)
        self.setWindowTitle("编辑属性")
        self.setFixedSize(320, 220)
        self.target_data = target_data  # 引用原始数据，修改后直接更新

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
        self.COLOR_BG = QColor(30, 35, 45)
        self.COLOR_LINE = QColor(255, 255, 255)
        self.COLOR_YELLOW_DOT = QColor(255, 255, 0)  # 标记黄点

        # 生成12个target，每个target为 [class_id, cx, cy, size, id, status, note]
        self.targets = self._generate_targets()
        self.lines = self._generate_lines()

        #按钮和其他相关成员
        # 按钮
        self.btn_detect = QPushButton("detect", self)
        self.btn_detect.clicked.connect(self.on_detect_clicked)
        # 按钮位置（右上角）
        self.btn_detect.move(self.width() - 100, 10)
        self.btn_detect.resize(100, 32)
        # 类别列表
        with open("models/config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        self.cls_list = config["cls_list"]
        # 存储检测到的中心点（用于绘制黄点）
        self.detected_points = []  # 每个元素为 (x_pixel, y_pixel)
        # 属性自动填充开关
        self.auto_fill_enabled = False

    def _generate_targets(self):
        targets = []
        # 强制四个类别各一个
        forced = [[0, 150, 200, 28, "RD_id", 0, "old note"],
                  [1, 380, 320, 26, "RS_id", 0, "old note"],
                  [2, 650, 480, 27, "GD_id", 0, "old note"],
                  [3, 900, 260, 29, "GS_id", 0, "old note"]]
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
                id_list=["RD_id", "RS_id", "GD_id", "GS_id"]
                targets.append([class_id, cx, cy, size, id_list[class_id], 0, "old note"])
        return targets

    def _generate_lines(self):
        lines = []
        num_lines = random.randint(8, 14)
        for _ in range(num_lines):
            line_type = random.choice(['h', 'v', 'd'])
            w, h = self.width(), self.height()
            if line_type == 'h':
                y = random.randint(80, h - 40)
                x1 = random.randint(40, w//3)
                x2 = random.randint(w//2, w-40)
                lines.append((x1, y, x2, y))
            elif line_type == 'v':
                x = random.randint(40, w-40)
                y1 = random.randint(80, h//2)
                y2 = random.randint(h//2, h-40)
                lines.append((x, y1, x, y2))
            else:
                x1 = random.randint(40, w//2)
                y1 = random.randint(80, h//2)
                x2 = random.randint(w//2, w-40)
                y2 = random.randint(h//2, h-40)
                lines.append((x1, y1, x2, y2))
        return lines

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), self.COLOR_BG)

        # 绘制白色直线干扰
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

        #绘制检测黄点及类别文字
        if self.detected_points:
            painter.setPen(Qt.NoPen)  # 无边框
            painter.setBrush(self.COLOR_YELLOW_DOT)
            dot_radius = 5
            for (px, py, cls) in self.detected_points:
                # 绘制黄点
                painter.drawEllipse(px - dot_radius, py - dot_radius, dot_radius * 2, dot_radius * 2)
                # 绘制类别文字（黄色，小号字体）
                painter.setPen(QColor(255, 255, 0))          # 黄色
                font = QFont("Microsoft YaHei", 8)           # 小号字体
                painter.setFont(font)
                cls_name=self.cls_list[cls]
                # 文字放在黄点右下方偏移 (6, 6) 像素，避免遮盖
                painter.drawText(px + 6, py + 6, cls_name)

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
        if dialog_y + 230 > self.height():
            dialog_y = self.height() - 235

        dialog = EditDialog(self, target)
        dialog.setModal(False) # 改为非模态
        dialog.show() # 显示但不阻塞

        # 自动填充从策略引擎获得的新值
        if self.auto_fill_enabled:# 如果自动填充标志为True
            cls_id=target[0]
            current_id=target[4]
            current_status=target[5]
            current_note=target[6]
            new_values=self.strategy_engine.generate_new_values([cls_id, current_id, current_status, current_note])
            # 延迟 300ms 后自动点击输入框并填入内容
            QTimer.singleShot(300, lambda: self.action_executor.fill_dialog_input(field_values=new_values, dialog=dialog))
            # 如果需要自动点击确定，再加一个延迟
            QTimer.singleShot(800, lambda: self.action_executor.click_ok_button(element_id="ok_button", dialog=dialog))
            # 自动填充完成后，通过对话框关闭信号来复位标志
            dialog.finished.connect(lambda: setattr(self, 'auto_fill_enabled', False))

    def on_detect_clicked(self):
        """点击按钮后执行：截图 -> 推理 -> 获取中心点 -> 重绘"""
        print("正在推理...")
        # 加载视觉服务
        if self.model_svc is None:
            self.model_svc = ModelService()
        self.btn_detect.hide()# 截图前隐藏按钮
        QApplication.processEvents()# 刷新页面
        # 截图当前窗口（得到QPixmap）
        pixmap = self.grab()  # 获取整个窗口的图像
        self.btn_detect.show()# 截图后恢复按钮
        # 转换为Qimage并统一为RGBA非预乘格式
        qimage = pixmap.toImage().convertToFormat(QImage.Format_RGBA8888)
        # 将QImage转换为PIL Image（使用内存缓冲区）
        buffer = qimage.bits().asstring(qimage.byteCount())
        pil_image = Image.frombuffer("RGBA", (qimage.width(), qimage.height()), buffer, "raw", "RGBA", 0, 1)
        pil_image = pil_image.convert("RGB")
        # 使用视觉服务推理出结果
        points = self.model_svc.predict(pil_image)# 每个目标的属性，即包含[x, y, cls]的列表，xy为窗口内的位置坐标
        self.strategy_engine.update_targets(points)# 把points作为targets传给策略引擎
        self.detected_points = points
        self.update()  # 触发paintEvent，绘制黄点和类别文字
        # 在控制台打印检测结果
        print(f"检测到 {len(points)} 个目标")
        # 自动填充置True
        self.auto_fill_enabled = True
        # 鼠标移动并点击0号目标处（根据策略决定）
        x, y=self.strategy_engine.get_action_target()# 通过策略引擎，获取目标位置
        self.action_executor.click_at_window_point(self, x, y)# 通过动作执行，移动鼠标并点击目标

def main():
    app = QApplication(sys.argv)
    font = QFont("Microsoft YaHei", 10)
    app.setFont(font)
    window = DemoWithLines()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()