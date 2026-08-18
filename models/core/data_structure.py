"""
目标检测结果数据结构模块：统一不同检测模型的输出格式
"""
from dataclasses import dataclass



@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int

    @property
    def center(self):
        return int((self.x1+self.x2)/2),int((self.y1+self.y2)/2)
