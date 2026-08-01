"""
执行器模块：模拟鼠标操作目标
"""
import pyautogui
from PyQt5.QtCore import QPoint
from typing import Optional
from locators import BaseLocator, OcrLocator



class ActionExecutor:
    """负责模拟鼠标操作"""
    def __init__(self, locator: Optional[BaseLocator] = None):
        self.locator = locator  # 注入定位器

    def click_at_window_point(self, window, local_x, local_y, duration=1.0):
        """
        视觉服务专用
        在给定主界面窗口的客户区坐标 (local_x, local_y) 处模拟鼠标左键单击。

        :param window: PyQt5 QWidget 子类实例（如 DemoWithLines）
        :param local_x: 目标在窗口客户区的 X 坐标
        :param local_y: 目标在窗口客户区的 Y 坐标
        :param duration: 鼠标移动持续时间（秒）
        """
        # 将客户区坐标转换为屏幕绝对坐标
        global_point = window.mapToGlobal(QPoint(int(local_x), int(local_y)))
        # 窗口置顶
        window.activateWindow()
        window.raise_()
        pyautogui.sleep(0.1)
        # 移动鼠标并点击
        pyautogui.moveTo(global_point.x(), global_point.y(), duration=duration)
        pyautogui.click()

    def fill_dialog_input(self, field_values: dict, delay_before=0.3, dialog=None):
        """
        自动在对话框中填入指定字段的值（模拟鼠标点击输入框 + 键盘输入）。
        :param field_values: 字典，键为element_id，值为要填入的内容
                             例如QtLocator传 {"id_input": "DEV001", "status_spin": 123, "note_input": "OK"}
        :param delay_before: 等待对话框渲染完成的延迟（秒）
        :param dialog: 可选，PyQt5 QDialog 实例（或其他 QDialog）
        """
        if isinstance(self.locator, OcrLocator):# OCR定位时，先截图并缓存OCR结果
            self.locator.start_cache()
        else:
            # Qt定位时，通知定位器当前对话框
            if hasattr(self.locator, 'set_current_dialog'):
                self.locator.set_current_dialog(dialog)
                # 确保对话框处于激活状态
                dialog.activateWindow()
                dialog.raise_()
                pyautogui.sleep(delay_before)

        # 调用定位器填充输入框的方法，element_id传入定位器
        for element_id, value in field_values.items():
            self.locator.type_text_in_element(element_id, str(value))

        if isinstance(self.locator, OcrLocator):# 填充结束后清除缓存
            self.locator.clear_cache()

    def click_ok_button(self, element_id: str, delay_before=0.3, dialog=None):
        """点击对话框的“确定”按钮"""
        # Qt定位时，通知定位器当前对话框
        if hasattr(self.locator, 'set_current_dialog'):
            self.locator.set_current_dialog(dialog)
            # 确保对话框处于激活状态
            dialog.activateWindow()
            dialog.raise_()
            pyautogui.sleep(delay_before)

        # 调用定位器单击元素的方法，element_id传入定位器
        self.locator.click_at_element(element_id)