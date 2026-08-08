"""
策略引擎模块：后台大脑，根据情况制定策略给出指令
"""
from typing import Tuple, Optional



class StrategyEngine:
    """
    points: 视觉服务检测到的图元信息列表
    每个元素为[x, y, cls, text]，分别为图元中心xy坐标，图元类别idx，图元关联的文本
    """
    def __init__(self):
        self.points=[]

    def update_points(self, points: Tuple[Tuple[int, int], ...]):
        """
        更新图元信息列表
        """
        self.points=points

    def get_action_point_demo(self) -> Optional[Tuple[int, int]]:
        """
        根据预设策略，返回要操作的图元像素坐标 (x, y)
        demo：如果有图元，返回第一个图元的坐标；否则返回 None
        """
        if not self.targets:
            return None
        # 这里可以扩展更复杂的规则，比如只选绿色菱形等
        point = self.points[0]
        return (point[0], point[1])

    def get_all_locked_point(self) -> Optional[list[Tuple[int, int]]]:
        """
        返回所有的上锁的图元像素坐标(x, y)的list
        """
        if not self.points:
            return None
        results=[]
        for point in self.points:
            if point[2] in [4, 5, 6, 7]:
                results.append((point[0], point[1]))
        return results

    def get_point_by_number(self, number: str) -> Optional[list[Tuple[int, int]]]:
        """
        返回所有编号为number的图元像素坐标(x, y)组成的list
        """
        if not self.points:
            return None
        results=[]
        for point in self.points:
            if point[3] == number:
                results.append((point[0], point[1]))
        return results

    def get_unlocked_point_by_number(self, number: str) -> Optional[list[Tuple[int, int]]]:
        """
        返回所有编号为number且未上锁的图元像素坐标(x, y)组成的list
        """
        if not self.points:
            return None
        results=[]
        for point in self.points:
            if point[3] == number and point[2] in [0, 1, 2, 3]:
                results.append((point[0], point[1]))
        return results

    def generate_new_values(self, old_values: dict = None)-> dict:
        """
        根据预设规则生成新的属性值。
        old_values: 旧属性，例如 {"id_input": "000001", "status_spin": 1, "note_input": "old note"}
        返回新属性字典，例如 {"id_input": "000001", "status_spin": 100, "note_input": "Updated"}
        """
        # 此处可以根据旧属性制定新属性
        ...
        
        new_values={}#新值
        new_values["id_input"]=old_values["id_input"]
        new_values["status_spin"]=100
        new_values["note_input"]="Updated"

        return new_values
    
    # 可选：增加更多策略方法，例如按 class_id 过滤、按时间条件等