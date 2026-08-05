"""
策略引擎模块：后台大脑，根据情况制定策略给出指令
"""
from typing import Tuple, Optional



class StrategyEngine:
    """
    targets: 视觉服务检测到的目标列表
    每个元素为[x, y, cls]，xy为目标在窗口内的位置坐标，cls为目标类别idx
    """
    def __init__(self):
        self.targets=[]

    def update_targets(self, targets: Tuple[Tuple[int, int], ...]):
        """
        更新目标列表
        """
        self.targets=targets

    def get_action_target_demo(self) -> Optional[Tuple[int, int]]:
        """
        根据预设策略，返回要操作的目标像素坐标 (x, y)
        demo：如果有目标，返回第一个目标的坐标；否则返回 None
        """
        if not self.targets:
            return None
        # 这里可以扩展更复杂的规则，比如只选绿色菱形等
        target = self.targets[0]
        return (target[0], target[1])

    def get_all_locked_target(self) -> Optional[list[Tuple[int, int]]]:
        """
        返回所有的上锁的目标像素坐标(x, y)的list
        """
        if not self.targets:
            return None
        results=[]
        for target in self.targets:
            if target[2] in [4, 5, 6, 7]:
                results.append((target[0], target[1]))
        return results

    def generate_new_values(self, target_data:list)-> dict:
        """
        根据目标数据和预设规则生成新的属性值。
        target_data: 如[cls_id, id, status, note]，cls_id同cls_idx，后三项为旧属性
        返回字典，例如 {"id_input": "DEV001", "status_spin": 100, "note_input": "Updated"}
        """
        cls_id=target_data[0]
        id=target_data[1]
        current_status=target_data[2]
        current_note=target_data[3]

        ...#此处可以根据cls_id和旧属性制定新属性

        new_values={}#新值
        new_values["id_input"]=id
        new_values["status_spin"]=current_status+1
        new_values["note_input"]="updated"

        return new_values
    
    # 可选：增加更多策略方法，例如按 class_id 过滤、按时间条件等