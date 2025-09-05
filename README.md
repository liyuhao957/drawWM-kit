# 两步法图像增强（批量）

用 OpenCV + NumPy 批量增强当前目录的 JPG/PNG 图片，为每张图生成“最终增强图”和“边缘图”。

- 脚本：`final_two_step.py`
- 依赖：Python 3.8+，`opencv-python`、`numpy`、`scipy`

## 安装
```bash
pip install opencv-python numpy scipy
# 无界面环境可用： pip install opencv-python-headless numpy scipy
```

## 使用
1. 将待处理图片与脚本放在同一目录
2. 运行：
```bash
python final_two_step.py
```
3. 输出：为每张图片生成 `<原名>_enhanced/`，包含：
   - `<原名>_ultra_clean.png`（最终增强）
   - `<原名>_edges.png`（边缘）

示例：
- `a.jpg` → 生成 `a_enhanced/a_ultra_clean.png`、`a_enhanced/a_edges.png`

