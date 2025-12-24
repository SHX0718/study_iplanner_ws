import PIL
import math
import torch
import torchvision.transforms as transforms

from . import traj_opt

class IPlannerAlgo:
    """
    拿到图片和目标 -> 问神经网络怎么走 -> 拿到几个关键路点 -> 把路点连成平滑的线 -> 交给机器人执行
    """

    def __init__(self, args):
        super(IPlannerAlgo, self).__init__()

        # 1. 读取配置 (模型路径、裁剪大小等)
        self.config(args)

        # 2. 定义图像预处理流程
        # 神经网络通常需要特定大小的输入，这里负责把原始图片缩放并转为 Tensor
        self.depth_transform = transforms.Compose([transforms.Resize(tuple(self.crop_size)), transforms.ToTensor()])

        # 3. 加载神经网络模型
        # 这里的 net 就是之前写的 PlannerNet
        net, _ = torch.load(self.model_save, map_location='cpu')
        self.net = net.cuda() if torch.cuda.is_available() else net

        # 4. 初始化轨迹优化器
        # 负责把神经网络输出的离散路点连成平滑曲线
        self.traj_generate = traj_opt.TrajOpt()
        return None
    
    def config(self, args):
        self.model_save = args.model_save
        self.crop_size = args.crop_size
        self.sensor_offset_x = args.sensor_offset_x
        self.sensor_offset_y = args.sensor_offset_y
        self.is_traj_shift = False
        if math.hypot(self.sensor_offset_x, self.sensor_offset_y) > 1e-5:
            self.is_tarj_shift = True
        return None

    def plan(self, image, goal_robot_frame):
        """
        plan 的 Docstring
        
        :param self: 说明
        :param image: 说明
        :param goal_robot_frame: 说明
        """
        # --- 1. 图像预处理 ---
        img = PIL.Image.fromarray(image)
        # 增加一个维度 (C, H, W) -> (1, C, H, W)
        img = self.depth_transform(img).expand(1, 3, -1, -1)

        # 搬运数据到 GPU 上
        if torch.cuda.is_available():
            img = img.cuda()
            goal_robot_frame = goal_robot_frame.cuda()

        # --- 2. 神经网络推理 ---
        # 推理模式，不需要计算梯度
        with torch.no_grad():
            # 核心步骤
            # 输入：深度图 + 目标位置
            # 输出：keypoints fear
            keypoints, fear = self.net(img, goal_robot_frame)

        # --- 3. 坐标系修正 ---
        # 如果相机安装位置和机器人底盘中心有偏移 (sensor_offset) , 需要在这里把点平移回底盘中心
        if self.is_tarj_shift:
            batch_size, _, dims = keypoints.shape
            # 这里有一个细节：手动在关键点最前面加了一个原点 (0, 0, 0)
            # 意味着轨迹必须从机器人当前位置 (0, 0, 0) 开始
            keypoints = torch.cat((torch.zeros(batch_size, 1, dims, device=keypoints.device, requires_grad=False), keypoints), axis=1)
            keypoints[..., 0] += self.sensor_offset_x
            keypoints[..., 1] += self.sensor_offset_y
        
        # --- 4. 轨迹生成 ---
        # 神经网络输出的点是稀疏的，机器人走的会很顿挫
        # 这里调用 traj_opt 把这几个点用连成一条密集的、平滑的轨迹
        traj = self.traj_generate.TrajGeneratorFromPFreeRot(keypoints, step=0.1)

        return keypoints, traj, fear, img