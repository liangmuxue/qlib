import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_filter1d

# ----------------------
# 模拟你图中的数据分布（与原图特征对齐）
# ----------------------
np.random.seed(42)
# 训练集（蓝色点）
x_train = np.array([-2.5, -2, -1.8, -1.6, -1.5, -1.5, -1.5, -1.4, -1.4, -1.4, -1.3, -1.3, -1.3, -1.3, -1.2, -1.2, -1.2, -1.2, -1.1, -1, 0, 0.6, 0.8, 0.9, 0.9, 0.9, 1, 1, 1, 1.1, 1.1, 1.6, 2, 2.3, 3.1, 3.2, 3.2, 3.5])
y_train = np.array([-0.8, -0.5, 4.5, 0.7, 0.1, -0.1, -2.5, 1.2, -0.7, -1.7, -3.5, -2.1, -1.5, 0.6, -4.5, -2.2, -1.6, 0.1, -1.3, -0.7, 0.6, 2.4, 1.9, 2.5, -0.1, 0.9, 0.1, 1.2, -0.5, 0.6, -1.4, -0.6, 0.2, -2.2, -1.2, -0.4, -0.3, -0.9])

# 测试集（橙色点）
x_test = np.array([-2.5, -2, -1.6, -1.5, -1.5, -1.5, -1.4, -1.4, -1.4, -1.3, -1.3, -1.2, -1.2, -1.1, 0, 0.6, 0.9, 0.9, 1, 1, 1, 1.1, 1.1, 1.2, 1.6, 2, 2.2, 3.1, 3.2, 3.2, 3.5])
y_test = np.array([0.5, 0.1, 1.2, 0.2, 0.0, -1.6, 0.6, -1.7, -2.5, -2.6, -3.6, 0.8, 0.6, -2.1, 0.6, 0.6, 0.1, 1.3, 1.2, -0.1, -0.5, -0.9, -1.0, 0.0, 0.0, 0.3, 0.5, 0.3, 0.0, 0.0, 1.2])

# ----------------------
# 绘制带平滑线的对比图
# ----------------------
plt.figure(figsize=(12, 8), dpi=120)

# 绘制散点
plt.scatter(x_train, y_train, c="#1f77b4", alpha=0.6, s=40, label="Train set")
plt.scatter(x_test, y_test, c="#ff7f0e", alpha=0.6, s=40, label="Test set")

# 训练集平滑拟合线
sort_idx_train = np.argsort(x_train)
x_train_sorted = x_train[sort_idx_train]
y_train_smooth = gaussian_filter1d(y_train[sort_idx_train], sigma=3)
plt.plot(x_train_sorted, y_train_smooth, c="darkblue", lw=2.5, label="Train trend (smoothed)")

# 测试集平滑拟合线
sort_idx_test = np.argsort(x_test)
x_test_sorted = x_test[sort_idx_test]
y_test_smooth = gaussian_filter1d(y_test[sort_idx_test], sigma=3)
plt.plot(x_test_sorted, y_test_smooth, c="darkorange", lw=2.5, label="Test trend (smoothed)")

# 辅助线与美化
plt.axhline(y=0, ls="--", c="gray", alpha=0.7)
plt.axvline(x=0, ls="--", c="gray", alpha=0.7)
plt.xlabel("Feature Value", fontsize=12)
plt.ylabel("SHAP Value", fontsize=12)
plt.title("Train vs Test SHAP Dependence Plot (with Smoothed Trend)", fontsize=14)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()