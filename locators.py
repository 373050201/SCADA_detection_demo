"""
定位器模块：抽象基类 + 各种子类的实现（如基于 Qt 句柄）
"""
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import pyautogui
from PyQt5.QtWidgets import QLineEdit, QSpinBox, QPushButton



class BaseLocator(ABC):
    """定位器抽象基类，所有定位器必须实现get_element_screen_point"""

    @abstractmethod
    def get_element_screen_point(self, element_id: str) -> Optional[Tuple[int, int]]:
        """
        根据 element_id 返回目标元素在屏幕上的中心点坐标 (x, y)
        element_id：元素的对象名称，UI中初始化的ObjectName，不同定位器应将对象名称映射为不同语义
        """
        ...

    def click_at_element(self, element_id: str, duration=0.15):
        """左键点击指定元素"""
        x, y=self.get_element_screen_point(element_id)
        pyautogui.moveTo(x, y, duration=duration)
        pyautogui.click()

    def type_text_in_element(self, element_id: str, text: str, duration=0.15):
        """点击元素并输入文本text"""
        point = self.get_element_screen_point(element_id)
        pyautogui.moveTo(point[0], point[1], duration=duration)
        pyautogui.click()
        pyautogui.hotkey('ctrl', 'a')
        pyautogui.press('delete')  
        pyautogui.typewrite(str(text), interval=0.02)



class QtLocator(BaseLocator):
    """Qt定位器，通过 PyQt5 控件句柄定位元素"""

    def __init__(self):
        self.current_dialog = None

    def set_current_dialog(self, dialog):# 设置当前对话框
        self.current_dialog = dialog

    def get_element_screen_point(self, element_id: str):
        """element_id映射：就是ObjectName本身"""
        if self.current_dialog:
            widget = self.current_dialog.findChild(QLineEdit, element_id)
            if widget is None:
                widget = self.current_dialog.findChild(QSpinBox, element_id)
            if widget is None:
                widget = self.current_dialog.findChild(QPushButton, element_id)
            if widget:
                center = widget.mapToGlobal(widget.rect().center())
                return (center.x(), center.y())
        return None



class OcrLocator(BaseLocator):
    """Ocr定位器，通过在图片上查找文字定位元素"""
    def __init__(self, ocr_engine=None, offset_map=None):
        """
        :param ocr_engine: 可选，已初始化的 OCR 引擎实例，具有 ocr(img) 方法，默认用RapidOCR
        :param offset_map: 可选，字典 {element_id: (dx, dy)}，用于从文字中心偏移到目标点击点
        """
        self.screenshot_func = pyautogui.screenshot # 截图函数：全屏截图，可自定义，返回待识别图像
        self.ocr_engine = ocr_engine
        if ocr_engine is None: # 默认用RapidOCR
            from rapidocr import RapidOCR
            self.ocr_engine = RapidOCR(params={
        "EngineConfig.onnxruntime.providers": ["CUDAExecutionProvider",],
        "EngineConfig.onnxruntime.cuda_ep_cfg": {"device_id": 0}
        })
        self.offset_map = offset_map or {}
        # ObjectName与具体文本的映射字典
        self.element_to_text = {
            "id_input": "ID:",
            "status_spin": "Status:",
            "note_input": "Note:",
            "ok_button": "确定",
            "cancel_button": "取消"
        }
        self._cached_results = None # 缓存OCR结果，实现一次截图多次使用

    def start_cache(self):
        """
        预截图并执行一次OCR，将结果缓存。
        调用后，后续所有get_element_screen_point将直接使用缓存，不再重新截图/推理。
        """
        img = self.screenshot_func()
        self._cached_results = self.ocr_engine(img)

    def clear_cache(self):
        """清除缓存"""
        self._cached_results = None
    
    def get_element_screen_point(self, element_id):
        """element_id映射：具体文本，如'确定' """
        # element_id映射为text
        text = self.element_to_text.get(element_id, element_id)  # 找不到则直接用原字符串

        # 优先使用缓存结果
        if self._cached_results is not None:
            results=self._cached_results
        else:
            img = self.screenshot_func() # 截图
            results = self.ocr_engine(img)  # OCR识别

        # 匹配文字，假设是RapidOCR格式
        for bbox, detected_text, _ in zip(results.boxes, results.txts, results.scores):
            if text in detected_text.strip():
                # 计算中心点
                x1, y1 = bbox[0]  # 左上
                x2, y2 = bbox[2]  # 右下
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                # 应用偏移（如有）
                dx, dy = self.offset_map.get(element_id, (0, 0))
                return (cx + dx, cy + dy) # 找到就直接返回，场景中若有多个目标，应修改此逻辑
        return None