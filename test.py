import torch
import matplotlib.pyplot as plt
import numpy as np

class CubicSplineTorch:
    def __init__(self):
        return None
    
    def h_poly(self, t):
        alpha = torch.arange(4, device=t.device, dtype=t.dtype)
        tt = t[:, None, :]**alpha[None, :, None]
        A = torch.tensor([
            [1, 0, -3, 2],
            [0, 1, -2, 1],
            [0, 0, 3, -2],
            [0, 0, -1, 1]
            ], dtype=t.dtype, device=t.device)
        return A @ tt
    
    def interp(self, x, y, xs):
        """
        执行三次 Hermite 样条插值
        
        :param self: 说明
        :param x: 关键点的时间索引 (Knots indices/time), 例如 [0, 1, 2, 3] (Batch, N_knots)
        :param y: 关键点的值 (Knots values), 即网络预测的轨迹控制点。形状 (Batch, N_knots, Dims)
        :param xs: 想要查询插值结果的时间点 (Query times) (Batch, N_query_points)
        """

        # 1. 计算切线/斜率 (m)
        # unsqueeze(2) 是为了让时间差能广播到 (Batch, N_knots-1, Dims)
        m = (y[:, 1:, :] - y[:, :-1, :]) / torch.unsqueeze(x[:, 1:] - x[:, :-1], 2)

        # 2. 拐点出平滑过渡
        # k = torch.zeros_like(y, dtype=y.dtype, device=y.device)
        # k[:, 0] = m[:, 0]
        # k[:, -1] = m[:, -1]
        # k[:, 1:-1] = (m[:, 1:] + m[:, :-1]) / 2
        m = torch.cat([m[:, None, 0], (m[:, 1:] + m[:, :-1]) / 2, m[:, None, -1]], 1)

        # 3. 确定查询点 xs 落在哪个区间内
        # idx [Batch, N_query_points]
        # 假设 x 所有 batch 都一样，所以取 x[0]
        idxs = torch.searchsorted(x[0, 1:], xs[0, :])

        # 4. 归一化时间变量 t，相当于在某段路程内的进度
        # x[0, idx] 利用了广播机制，变成和 xs 一样的形状，[Batch, N_query_points]
        dx = x[:, idxs + 1] - x[:, idxs]
        t = (xs - x[:, idxs]) / dx

        # 5. 计算 Hermite 基函数值
        # self.h_poly 返回形状: [Batch, 4, N_query_points]
        hh = self.h_poly(t)
        hh = torch.transpose(hh, 1, 2)  # 变成 [Batch, N_query_points, 4]

        # 6. $$Result = (h_0 \cdot y_1) + (h_1 \cdot k_1 \cdot dx) + (h_2 \cdot y_2) + (h_3 \cdot k_2 \cdot dx)$$
        # y_1, y_2 是位置，k_1, k_2 是切线斜率，dx 是区间长度
        inter_points = hh[:, :, 0:1] * y[:, idxs, :]                  # h00 * p0
        inter_points = inter_points + hh[:, :, 1:2] * m[:, idxs] * dx[:,:,None] # h10 * m0 * dx
        inter_points = inter_points + hh[:, :, 2:3] * y[:, idxs + 1, :]        # h01 * p1
        inter_points = inter_points + hh[:, :, 3:4] * m[:, idxs + 1] * dx[:,:,None] # h11 * m1 * dx

        return inter_points
    
# ==========================================
# 测试逻辑
# ==========================================
def test_gradient():
    print("\n--- Starting Gradient Check ---")
    spline = CubicSplineTorch()
    
    # 1. 准备数据
    # 注意：我们要检查的是对 y 的梯度，所以 y 需要 requires_grad=True
    x = torch.arange(5, dtype=torch.float32).view(1, -1) 
    y = torch.tensor([[
        [0.0, 0.0], 
        [1.0, 1.0], 
        [2.0, 0.0], 
        [3.0, -1.0], 
        [4.0, 0.0]
    ]], dtype=torch.float32, requires_grad=True) # <--- 关键点
    
    xs = torch.linspace(0, 4, 100).view(1, -1)
    
    # 2. 前向传播 (Forward Pass)
    try:
        out = spline.interp(x, y, xs)
        print("Forward pass successful.")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        return

    # 3. 定义一个简单的 Loss
    # 假设我们的目标是让所有插值点的坐标之和尽可能小（只是为了产生梯度）
    loss = out.sum()
    
    # 4. 反向传播 (Backward Pass)
    try:
        # 清空之前的梯度（如果有）
        if y.grad is not None:
            y.grad.zero_()
            
        loss.backward()
        print("Backward pass successful.")
    except Exception as e:
        print(f"Backward pass failed: {e}")
        return

    # 5. 检查梯度是否存在
    if y.grad is None:
        print("❌ Error: y.grad is None! The graph is broken.")
    else:
        print("✅ Success: Gradient calculated!")
        print(f"Gradient shape: {y.grad.shape}")
        # 打印部分梯度值看看是否非零
        print("Gradient sample (first point):", y.grad[0, 0])
        
        # 检查是否有梯度是 NaN (数值不稳定)
        if torch.isnan(y.grad).any():
            print("⚠️ Warning: NaN values found in gradients!")
        else:
            print("Gradient values look numerically stable.")

if __name__ == "__main__":
    test_gradient()