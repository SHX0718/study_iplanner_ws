在 iplanner/ 文件夹下，我们可以创建以下子文件夹：

models/ (模型库) 🧠

这里专门放神经网络相关的代码。

搬进去的文件：percept_net.py (视觉), planner_net.py (决策)。

planning/ (规划算法) 🛣️

这里放纯粹的数学计算和轨迹生成算法。

搬进去的文件：traj_opt.py (我们正在写的), 还有未来会用到的 ip_algo.py (核心算法), traj_cost.py。

utils/ (工具箱) 🛠️

这里放各种杂七杂八的辅助工具，比如可视化、坐标变换等。

搬进去的文件：traj_viz.py, rosutil.py, torchutil.py (如果有的话)。

mapping/ (建图模块) 🗺️

如果你以后要复现建图部分，可以放这里。

搬进去的文件：esdf_mapping.py, tsdf_map.py。