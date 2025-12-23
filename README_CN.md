这是 `README.md` 文件的中文翻译版：

# 命令式路径规划器 (iPlanner)

## 概述

欢迎来到 **[iPlanner: Imperative Path Planning](https://arxiv.org/abs/2302.11434)** 代码仓库。iPlanner 通过一种创新的命令式学习方法（Imperative Learning Approach）进行训练，并且仅使用前视深度图像进行局部路径规划。

这里有一个展示 iPlanner 功能的视频：**[视频](https://youtu.be/-IfjSW0wPBI)**

**关键词：** 导航，局部规划，命令式学习

### 许可证

本代码在 MIT 许可证下发布。

**作者: Fan Yang<br />
维护者: Fan Yang, fanyang1@ethz.ch**

iPlanner 软件包已在 Ubuntu 20.04 上的 ROS Noetic 下进行了测试。这是研究代码，不保证适用于特定目的。

<p align="center">
<img src="img/example.jpg" alt="Method" width="70%"/>
</p>

## 安装

#### 依赖项

要运行 iPlanner，你需要安装 [PyTorch](https://pytorch.org/)。我们建议使用 [Anaconda](https://docs.anaconda.com/anaconda/install/index.html) 进行安装。请查看官方网站以获取 Anaconda 和 PyTorch 的相应安装说明。

请按照项目根目录下的 `INSTALL.md` 文件中提供的说明设置环境并安装必要的软件包。

#### 仿真环境设置

请参考 Autonomous Exploration Development Environment 仓库来设置 Gazebo 仿真环境：[网站](https://www.cmu-exploration.com/)，请切换到 **noetic_rgbd_camera** 分支。

#### 构建

要构建仓库并设置正确的 Python 版本以运行，请使用以下命令：

```
catkin build iplanner_node -DPYTHON_EXECUTABLE=$(which python)

```

Python3 应该是你之前设置好的、已安装 Torch 和 PyPose 的 Python 版本。如果使用 Anaconda 环境，请激活 conda 环境并检查：

```
which python

```

## 训练

进入 **iplanner** 文件夹

```
cd <your_imperative_planenr_path>/iplanner

```

#### 预训练网络和训练数据

[在此处](https://drive.google.com/file/d/1UD11sSlOZlZhzij2gG_OmxbBN4WxVsO_/view?usp=share_link)下载预训练的网络权重 `plannernet.pt` 并将其放入 **models** 文件夹中。注意，此预训练网络尚未适应真实世界的数据。

你也可以在仿真环境或真实世界中自行收集数据。启动 **data_collect_node**：

```
roslaunch iplanner_node data_collector.launch

```

提供 `config/data_params.yaml` 中列出的必要话题信息。收集的数据将放入 `data/CollectedData` 文件夹中，并为你可以在 `config/data_params.yaml` 的 **env_name** 下指定的不同环境生成文件夹。

对于每个环境，数据包含以下结构：

```
Environment Data
├── camera
|   ├── camera.png
│   └── split.pt
├── camera_extrinsic.txt
├── cloud.ply
├── color_intrinsic.txt
├── depth
|    ├── depth.png
│   └── split.pt
├── depth_intrinsic.txt
├── maps
│   ├── cloud
│   │   └── tsdf1_cloud.txt
│   ├── data
│   │   ├── tsdf1
├── data
│   │   └── tsdf1_map.txt
│   └── params
│       └── tsdf1_param.txt
└── odom_ground_truth.txt

```

你可以使用[此处的 Google Drive 链接](https://drive.google.com/file/d/1bUN7NV7arMM8ASA2pTJ8hvdkc5N3qoJw/view?usp=sharing)下载我们提供的示例数据。

#### 生成训练数据

使用以下命令导航到项目中的 iplanner 文件夹：

```
cd <<YORU WORKSPACE>>/src/iPlanner/iplanner

```

运行 Python 脚本生成训练数据。需要生成数据的环境在文件 `collect_list.txt` 中指定。你可以在 `config/data_generation.json` 文件中修改数据生成参数。

```
python data_generation.py

```

准备好训练数据后，使用以下命令开始训练过程。你可以在 `config/training_config.json` 文件中指定不同的训练参数。

```
python training_run.py

```

## 运行 iPlanner ROS 节点

启动不带默认局部规划器的仿真环境：

```
roslaunch vehicle_simulator simulation_env.launch

```

运行不带可视化的 iPlanner ROS 节点：

```
roslaunch iplanner_node iplanner.launch

```

或者运行带可视化的 iPlanner ROS 节点：

```
roslaunch iplanner_node iplanner_viz.launch

```

### 路径跟随 (Path Following)

为了确保规划器正确执行规划的路径，你需要运行一个独立的控制器或路径跟随器。按照以下步骤使用 iplanner 仓库提供的 launch 文件设置路径跟随器：

将默认的 iplanner_path_follower 下载到你的工作区中。使用以下命令导航到你的工作区源目录：

```
cd <<YOUR WORKSPACE>>/src

```

然后克隆仓库：

```
git clone [https://github.com/MichaelFYang/iplanner_path_follow.git](https://github.com/MichaelFYang/iplanner_path_follow.git)

```

使用以下命令编译路径跟随器：

```
catkin build iplanner_path_follow

```

请注意，此仓库是 [CMU-Exploration](https://www.cmu-exploration.com/) 路径跟随组件的一个分支。欢迎探索和尝试适合你特定机器人平台的其他控制器或路径跟随器。

### 航点导航 (Waypoint Navigation)

要通过 Rviz 发送航点，请下载 rviz waypoint plugin。使用以下命令导航到你的工作区源目录：

```
cd <<YOUR WORKSPACE>>/src

```

然后克隆仓库：

```
git clone [https://github.com/MichaelFYang/waypoint_rviz_plugin.git](https://github.com/MichaelFYang/waypoint_rviz_plugin.git)

```

使用以下命令编译 waypoint rviz plugin：

```
catkin build waypoint_rviz_plugin

```

### 智能摇杆 (SmartJoystick)

按下摇杆上的 **LB** 按钮，当屏幕上看到输出：

```
Switch to Smart Joystick mode ...

```

即表示智能摇杆功能已启用。它将摇杆命令作为运动意图，并在后台运行 iPlanner 以进行底层的避障。

## 配置文件

参数文件 **`data_params.yaml`** 用于数据收集

* **vehicle_sim.yaml** 配置文件包含：
* **`main_freq`** ROS 节点运行频率
* **`odom_associate_id`** 根据不同的 SLAM 设置，里程计基座可能未设置在机器人基座坐标系下



参数文件 **`vehicle_sim.yaml`** 用于 iPlanner ROS 节点

* **vehicle_sim.yaml** 配置文件包含：
* **`main_freq`** ROS 节点运行频率
* **`image_flap`** 根据相机设置，可能需要将图像上下翻转或不翻转
* **`crop_size`** 裁剪传入相机图像的尺寸
* **`is_fear_act`** 使用预测的碰撞可能性值来停止
* **`joyGoal_scale`** 智能摇杆模式下摇杆发送的目标点的最大距离



## 引用

如果你在研究中使用了此代码库，我们恳请你引用我们的工作。你可以按如下方式引用：

* Yang, F., Wang, C., Cadena, C., & Hutter, M. (2023). iPlanner: Imperative Path Planning. Robotics: Science and Systems Conference (RSS). Daegu, Republic of Korea, July 2023.

Bibtex:

```
@INPROCEEDINGS{Yang-RSS-23, 
    AUTHOR    = {Fan Yang AND Chen Wang AND Cesar Cadena AND Marco Hutter}, 
    TITLE     = {{iPlanner: Imperative Path Planning}}, 
    BOOKTITLE = {Proceedings of Robotics: Science and Systems}, 
    YEAR      = {2023}, 
    ADDRESS   = {Daegu, Republic of Korea}, 
    MONTH     = {July}, 
    DOI       = {10.15607/RSS.2023.XIX.064} 
}

```

## 作者

此代码库由 [Fan Yang](https://github.com/MichaelFYang) 开发和维护。如有任何疑问或需要进一步帮助，可以通过 fanyang1@ethz.ch 联系他。