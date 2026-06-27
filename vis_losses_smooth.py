import numpy as np
import matplotlib.pyplot as plt
import os

# 1. 读取原始 loss
loss_path ="/home/lxp/Ground_reasoning/Models/scene_slot/3b_baseline_ft_lr2e-5/losses.npy"
losses = np.load(loss_path)

steps = np.arange(1, len(losses) + 1)

# 2. 做滑动平均
window = 50  # 滑动窗口大小
kernel = np.ones(window) / window
# 为了对齐长度，用 'valid'；如果想保持同长度也可以再 pad 一下
smooth_losses = np.convolve(losses, kernel, mode="valid")
smooth_steps = steps[window - 1:]  # 对齐横坐标

# 3. 画图（原始 + 平滑）
plt.figure(figsize=(9, 5))

# 原始曲线：浅色、细一点
plt.plot(steps, losses, alpha=0.2, linewidth=0.5, label="raw loss")

# 平滑曲线：深色、粗一点
plt.plot(smooth_steps, smooth_losses, linewidth=2.0, label=f"smoothed (window={window})")

plt.xlabel("Training step")
plt.ylabel("Loss")
plt.title("Training Loss Curve (Smoothed)")
plt.grid(True)
plt.legend()

out_path = os.path.join(os.path.dirname(loss_path), "loss_curve_smooth.png")
plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close()

print("保存完成：", out_path)
