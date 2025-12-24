import torch

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

        :return inter_points: 插值结果 [Batch_size, 生成点数, 3] 其中生成点数由 step 决定
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
    

class TrajOpt:
    def __init__(self):
        self.cs_interp = CubicSplineTorch()
        return None
    
    def TrajGeneratorFromPFreeRot(self, preds, step): 
        """
        从网络预测的关键点生成完整轨迹。
        
        :param self: 说明
        :param preds: 网络预测的相对控制点，形状 (Batch, Num_P, Dims) 注意：通常网络预测的是不包含原点的后续点。
        :param step: 插值步长，决定轨迹的平滑度/密度。
        """
        # Points is in se3
        batch_size, num_p, dims = preds.shape
        points_preds = torch.cat((torch.zeros(batch_size, 1, dims, device=preds.device, requires_grad=preds.requires_grad), preds), axis=1)
        num_p = num_p + 1
        xs = torch.arange(0, num_p-1+step, step, device=preds.device)
        xs = xs.repeat(batch_size, 1)
        x  = torch.arange(num_p, device=preds.device, dtype=preds.dtype)
        x  = x.repeat(batch_size, 1)
        waypoints = self.cs_interp.interp(x, points_preds, xs)
        return waypoints  # R3