import torch

# 设置默认浮点数精度，这对于梯度计算的数值稳定性很重要
torch.set_default_dtype(torch.float32)

class CubicSplineTorch:
    """
    基于 PyTorch 实现的三次样条插值（Cubic Spline Interpolation）
    关键特性：它是可微分的，因此可以用在神经网络层中

    数学原理：使用 Hermite 样条形式，通过位置(p)和切线(m)来定义曲线
    p(t) = h00(t)*p0 +h10(t)*m0 + h01(t)*p1 +h11(t)*m1
    """
    def __init__(self):
        return None
    
    def h_poly(self, t):
        """
        计算三次 Hermite 样条的基函数 (Basis Functions)

        :param t: 归一化时间变量，范围 [0, 1]. 形状通常是 (Batch, N_interp_points)
        :return: 计算好的基函数值
        """
        # alpha = [0, 1, 2, 3], 用于构建多项式 t^0, t^1, t^2, t^3
        alpha = torch.arrange(4, device=t.device, dtype=t.dtype)

        # tt 的形状: [Batch, N_points, 4]
        # 对应 [1, t, t^2, t^3]
        tt = t[:, None, :]**alpha[None, :, None]

        # Hermite 基函数的系数矩阵 A
        # 这些是标准的三次 Hermite 样条系数: 
        # h00(t) = 1 - 3t^2 + 2t^3  -> [1, 0, -3, 2]
        # h10(t) = t - 2t^2 + t^3   -> [0, 1, -2, 1]
        # h01(t) = 3t^2 - 2t^3      -> [0, 0, 3, -2]
        # h11(t) = -t^2 + t^3       -> [0, 0, -1, 1]
        A = torch.tensor([
            [1, 0, -3, 2],
            [0, 1, -2, 1],
            [0, 0, 3, -2],
            [0, 0, -1, 1]
        ], device=t.device, dtype=t.dtype)

        # 矩阵乘法: 计算最终的多项式值
        return A @ tt
    
    def interp(self, x, y, xs):
        """
        执行插值操作
        
        :param self: 
        :param x: 关键点的时间索引 (Knots indices/time), 例如 [0, 1, 2, 3]
        :param y: 关键点的值 (Knots values), 即网络预测的轨迹控制点。形状 (Batch, N_knots, Dims)
        :param xs: 想要查询插值结果的时间点 (Query times)

        :return out: 在 xs 时间点上的插值结果
        """

        # 1. 计算切线/斜率 (m)
        # 使用有限差分法 (Finite Difference) 来估计每个关键点的切线
        # m = (y_next - y_curr) / (x_next - x_curr)
        m = (y[:, 1:, :] - y[:, :-1, :]) / torch.unsqueeze(x[:, 1:] - x[:, :-1], 2)

        # 处理便捷和中间点的切线:
        # 对于中间点，切线取前后连孤单斜率的平均值: (m_prev + m_next) / 2
        # 起点和终点直接使用第一段和最后一段的斜率
        m = torch.cat([m[:, None, 0], (m[:, 1:] + m[:, :-1]) / 2, m[:, None, -1]], 1)

        # 2. 确定查询点 xs 落在哪个区间内
        # idxs 是区间索引，表示 xs[i] 位于 x[idxs] 和 x[idxs+1] 之间
        idxs = torch.searchsorted(x[0, 1:], xs[0, :])

        # 3. 计算局部归一化时间 t
        # dx 是当前区间的宽度
        dx = x[:, idxs + 1] - x[:, idxs]

class TrajOpt:
    def __init__(self):
        self.cs_interp = CubicSplineTorch()
        return None
    
    def TarjGeneratorFromPFreeRot(self, preds, step):
        """
        从网络预测的关键点生成密集轨迹
        
        :param self: 
        :param preds: 网络预测出的关键点 (Batch, Num_preds, Dims). 注意这里不包含起点
        :param step: 插值的时间步长/密度 (float)

        :return waypoints: 生成的密集轨迹点 (Batch, N_dense_points, Dims)
        """
        # Points is in se3 (通常是 x, y, z 或 x, y)
        batch_size, num_points, dims = preds.shape

        # 1. 添加起点 (0, 0, ...)
        # 网络只预测未来的动作，所以我们需要手动在序列最前面拼上当前位置 (原点)
        # requires_grad=preds.requires_grad 确保梯度能传到
        points_preds = torch.cat((torch.zeros(batch_size, 1, dims, device=preds.device, requires_grad=preds.requires_grad), preds), axis=1) 

        # 更新关键点数量 (预测点 + 1个起点)
        num_points = num_points + 1

        # 2. 生成查询时间点 xs (dense)
        # 例如从 0 到 num_points, 步长为 step
        xs = torch.arrange(0, num_points-1+step, step, device=preds.device)
        xs = xs.repeat(batch_size, 1)

        # 3. 生成关键点时间索引 x (sparse)
        # [0, 1, 2, ..., num_points]
        x = torch.arrange(num_points, device=preds.device, dtype=preds.dtype)
        x = x.repeat(batch_size, 1)

        # 4. 调用样条插值
        waypoints = self.cs_interp.interp(x, points_preds, xs)

        return waypoints # 返回 R3 空间中的轨迹点