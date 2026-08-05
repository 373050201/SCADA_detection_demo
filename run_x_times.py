"""
自动运行x次脚本
"""
import subprocess
import sys
import os



x=500
script_to_run = "mock_UI_generator.py"
counter_file="counter.txt"

# 循环结束前删除 counter.txt
if os.path.exists(counter_file):
    os.remove(counter_file)
    print("已删除 counter.txt")
else:
    print("开始循环，counter.txt 不存在，无需删除")

for _ in range(x):
    result = subprocess.run([sys.executable, script_to_run])
    if result.returncode != 0:
        print(f"第 {_+1} 次运行失败，退出码: {result.returncode}")
        break

# 循环结束后删除 counter.txt
if os.path.exists(counter_file):
    os.remove(counter_file)
    print("循环结束，已删除 counter.txt")
else:
    print("counter.txt 不存在，无需删除")