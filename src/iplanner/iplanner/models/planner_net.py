import torch
import torch.nn as nn
from .percept_net import PerceptNet

class Decoder(nn.Module):
    def __init__(self, in_channels, goal_channels, k=5):
        super().__init__()
        self.k = k
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        # 目标编码层 (fg) : 把 3 维的目标坐标 (x, y, z) 映射到 goal_channels 维
        self.fg = nn.Linear(3, goal_channels)

        # 融合后的卷积层
        self.conv1 = nn.Conv2d(in_channels=in_channels+goal_channels, out_channels=512, kernel_size=5, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=0)

        # 全连接层
        # 路径点
        self.fc1 = nn.Linear(256*128, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, k*3)
        # 碰撞预测
        self.frc1 = nn.Linear(1024, 128)
        self.frc2 = nn.Linear(128, 1)

    def forward(self,x, goal):
        goal = self.fg(goal[:, 0:3]) # 确保只取前 3 维 (x, y, z), 目标点数据可能包含额外的数据，比如偏航角
        goal = goal[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])

        x = torch.cat((x, goal), dim=1)
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)

        x = torch.flatten(x, start_dim=1)

        f = self.fc1(x)
        f = self.relu(f)

        waypoints = self.fc2(f)
        waypoints = self.relu(waypoints)
        waypoints = self.fc3(waypoints)
        waypoints = waypoints.reshape(-1, self.k, 3)

        collision = self.frc1(f)
        collision = self.relu(collision)
        collision = self.frc2(collision)
        collision = self.sigmoid(collision)

        return waypoints, collision
    

class PlannerNet(nn.Module):
    def __init__(self, encoder_channel=64, k=5):
        super().__init__()
        self.encoder = PerceptNet(layers=[2, 2, 2, 2])  # ResNet-18
        self.decoder = Decoder(in_channels=512, goal_channels=encoder_channel, k=k)

    def forward(self, x, goal):
        features = self.encoder(x)
        waypoints, collision = self.decoder(features, goal)
        return waypoints, collision

# if __name__ == "__main__":
#     # 1. 实例化模型
#     model = PlannerNet(encoder_channel=64, k=5)
    
#     # 2. 伪造输入数据
#     # 图像: Batch=2, Channel=3, Height=224, Width=224
#     # 修改前: dummy_image = torch.randn(2, 3, 224, 224)
#     # 修改后: 使用配置文件中的 crop_size [360, 640]
#     dummy_image = torch.randn(2, 3, 360, 640)
#     # 目标: Batch=2, 坐标=(x,y,z)
#     dummy_goal = torch.randn(2, 3)
    
#     # 3. 前向传播
#     waypoints, collision = model(dummy_image, dummy_goal)
    
#     # 4. 打印结果
#     print(f"Waypoints shape: {waypoints.shape}") # 预期: [2, 5, 3] (2个样本, 5条路径, 每条3个参数)
#     print(f"Collision shape: {collision.shape}") # 预期: [2, 1] (2个样本, 各1个概率值)
        