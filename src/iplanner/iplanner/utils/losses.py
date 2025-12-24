import math
import torch
import torch.nn as nn
import collections
from itertools import repeat

# --- 辅助函数 ---

def _ntuple(n):
    """
    一个高阶函数，它的作用不是直接处理数据，而是生成一个新的函数
    
    :param n: 一个参数n, 表示你希望生成的函数能将输入扩展成n个元素的元组
    
    :return parse: parse 函数回记住你当时传入的 n 值
    """
    def parse(x):
        """
        解决：用户有时候给一个数字，有时候给一个列表，但我们希望最终都能得到一个长度为 n 的元组
        
        :param x: 用户输入的值，可能是单个值，也可能是一个可迭代对象（如列表或元组）

        :return x: 如果 x 是可迭代对象，直接返回它；否则返回一个包含 n 个 x 的元组
        :return tuple(repeat(x, n)): 如果 x 不是可迭代对象，就创建一个包含 n 个 x 的元组；否则直接返回 x
        """
        if isinstance(x, collections.abc.Iterable):  # 检查 x 是否是可迭代对象
            return x  # 如果是，直接返回 x
        return tuple(repeat(x, n))
    return parse

# 本质就是那个记住 n = 2 的 parse 函数
_pair = _ntuple(2)

# --- 损失函数类 ---

class ConvLoss(nn.Module):
    """
    卷积损失函数类
    """
    def __init__(self, input_size, kernel_size, stride, in_channels=3, color=1):
        super(ConvLoss, self).__init__()
        self.color = color
        # 如果你传入的 input_size, kernel_size, stride 是单个数字，就把它们扩展成 (数字, 数字) 的形式
        # 如果是 3 ，就变成 (3, 3)
        # 如果是 (3, 5) 就保持不变
        input_size = _pair(input_size)
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)

        # 创建卷积层
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=kernel_size, stride=stride, bias=False)
        
        # 初始化卷积核为均值滤波器
        # 这里手动把卷积核里的所有数值都设置为了相同的值 1 / 元素总数
        # 这样做卷积运算相当于像素与权重相乘再相加，相当于做一个平均值滤波
        self.conv.weight.data = torch.ones(self.conv.weight.size()).cuda() / self.conv.weight.numel()

        # 计算输出特征图的宽和高
        self.width = (input_size[0] - kernel_size[0]) // stride[0] + 1
        self.height = (input_size[0] - kernel_size[1]) // stride[1] + 1

        # 创建全局最大池化层
        # 将池化核的大小直接设置成了特征图的完整宽和高
        # 其实就是无论特征图数据是什么样的，直接拿一个和图一样大的框盖上去，选出整张图最大的那个数值
        self.pool = nn.MaxPool2d((self.width, self.height))

    def forward(self, x, y):
        # 求每个像素的绝对误差
        loss = self.conv((x - y).abs())
        # 找出误差最大的区域
        value, index = loss.view(-1).max(dim=0)
        # 计算该区域在原图中的左上角坐标
        w = (index // self.width) * self.conv.stride[0]
        h = (index % self.width) * self.conv.stride[1]
        # 在该区域画一个边框, 减去 color 值,通常是为了可视化调试
        x[:, :, w:w + self.conv.kernel_size[0]  , h                               ] -= self.color
        x[:, :, w:w + self.conv.kernel_size[0]  , h + self.conv.kernel_size[1] - 1] -= self.color
        x[:, :, w                               , h:h + self.conv.kernel_size[1]  ] -= self.color
        x[:, :, w + self.conv.kernel_size[0] - 1, h:h + self.conv.kernel_size[1]  ] -= self.color
        return value
    
class CosineSimilarity(nn.Module):
    '''
    **余弦相似度损失函数类**

    它不关心两个特征的“强弱”（数值大小），只关心它们是否指向同一个“方向”。

    - 想象两个箭头。如果它们重叠，相似度是 1；如果垂直，是 0；如果方向相反，是 -1。
    - 在图像特征中，这意味着：即使光照变暗了（像素值整体变小），只要纹理结构（方向）没变，余弦相似度依然很高。
    '''
    def __init__(self, eps=1e-7):
        super(CosineSimilarity, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        N, C, H, W = x.size()
        B, c, h, w = y.size()
        assert(C==c and H==h and W==w)
        x, y = x.view(N,1,C,H*W), y.view(B,C,H*W)
        xx, yy = x.norm(dim=-1), y.norm(dim=-1)
        xx[xx<self.eps], yy[yy<self.eps] = self.eps, self.eps
        return ((x*y).sum(dim=-1)/(xx*yy)).mean(dim=-1)


class CosineLoss(nn.CosineEmbeddingLoss):
    """
    它是 CosineSimilarity 的训练版本。
    """
    def __init__(self, dim=1):
        super(CosineLoss, self).__init__()
        # 预设目标为 1：意思是告诉网络，“x 和 y 应该是一样的”
        self.target = torch.ones(dim).cuda()

    def forward(self, x, y):
        return super(CosineLoss, self).forward(x, y, self.target)/2


class PearsonLoss(nn.CosineEmbeddingLoss):
    """
    **皮尔逊相关损失函数类**

    它是**“去中心化”**版的余弦相似度。比前者更抗干扰。

    - 假设图片 A 的像素值是 [10, 20, 30]，图片 B 是 [110, 120, 130]。
    - 虽然 B 比 A 亮很多（整体偏移了 100），但它们的变化趋势是一模一样的（都涨了 10 和 20）。
    - PearsonLoss 先减去均值（去中心化），把它们变成 [-10, 0, 10]，这样两者就完全一样了。
    - 应用场景：当环境光照剧烈变化，或者传感器存在基线漂移时，用这个 Loss 训练模型会更稳定。
    """
    def __init__(self, dim=1):
        super(PearsonLoss, self).__init__()
        self.target = torch.ones(dim).cuda()

    def forward(self, x, y):
        x = x - x.mean()
        y = y - y.mean()
        return super(PearsonLoss, self).forward(x, y, self.target)


class CorrelationSimilarity(nn.Module):
    '''
    利用了 FFT（快速傅里叶变换） 来高效计算多通道 2D 图像块之间的相关相似度

    - values: 最大的相似度是多少？（比如 0.9，说明很像）。
    - indices: 这个最大值在哪里？（这就代表了位移量）。
    - 作用：这实际上是一个全局定位或配准模块。如果机器人不知道自己在哪里，它可以把自己看到的局部图和全局地图做这个运算，indices 就会告诉它：“嘿，你其实向右偏移了 5 米”。
    '''
    def __init__(self, input_size):
        super(CorrelationSimilarity, self).__init__()
        self.input_size = input_size = _pair(input_size)
        assert(input_size[-1]!=1) # FFT2 is wrong if last dimension is 1
        self.N = math.sqrt(input_size[0]*input_size[1])
        self.fft_args = {'s': input_size, 'dim':[-2,-1], 'norm': 'ortho'}
        self.max = nn.MaxPool2d(kernel_size=input_size)

    def forward(self, x, y):
        X = torch.fft.rfftn(x, **self.fft_args).unsqueeze(1)
        Y = torch.fft.rfftn(y, **self.fft_args)
        g = torch.fft.irfftn((X.conj()*Y).sum(2), **self.fft_args)*self.N
        xx = x.view(x.size(0),-1).norm(dim=-1).view(x.size(0), 1, 1)
        yy = y.view(y.size(0),-1).norm(dim=-1).view(1, y.size(0), 1)
        g = g.view(x.size(0), y.size(0),-1)/xx/yy
        values, indices = torch.max(g, dim=-1)
        indices = torch.stack((indices // self.input_size[1], indices % self.input_size[1]), dim=-1)
        values[values>+1] = +1 
        values[values<-1] = -1 
        assert((values>+1).sum()==0 and (values<-1).sum()==0)
        return values, indices


class Correlation(nn.Module):
    '''
    这是 CorrelationSimilarity 的简化版，主要用于计算 Loss。

    - accept_translation 开关的含义：
        - True：“虽然你现在的预测有点歪，但只要你能平移一下对上，我就算你对。”（容忍位置误差，关注特征模式）。
        - False：“必须在原位严丝合缝地对上，偏一点都不行。”（强行约束位置精度）。
    '''
    def __init__(self, input_size, accept_translation=True):
        super(Correlation, self).__init__()
        self.accept_translation = accept_translation
        input_size = _pair(input_size)
        assert(input_size[-1]!=1) 
        self.N = math.sqrt(input_size[0]*input_size[1])
        self.fft_args = {'s': input_size, 'dim':[-2,-1], 'norm': 'ortho'}
        self.max = nn.MaxPool2d(kernel_size=input_size)

    def forward(self, x, y):
        X = torch.fft.rfftn(x, **self.fft_args)
        Y = torch.fft.rfftn(y, **self.fft_args)
        g = torch.fft.irfftn((X.conj()*Y).sum(2), **self.fft_args)*self.N
        xx = x.view(x.size(0),-1).norm(dim=-1)
        yy = y.view(y.size(0),-1).norm(dim=-1)
        if self.accept_translation is True:
            return self.max(g).view(-1)/xx/yy
        else:
            return g[:,0,0].view(-1)/xx/yy


class CorrelationLoss(Correlation):
    '''
    它是那个最复杂的 FFT Correlation 类的训练版本。
    '''
    def __init__(self, input_size, reduce = True, accept_translation=True):
        super(CorrelationLoss, self).__init__(input_size, accept_translation)
        self.reduce = reduce

    def forward(self, x, y):
        loss = (1 - super(CorrelationLoss, self).forward(x, y))/2
        if self.reduce is True:
            return loss.mean()
        else:
            return loss