# AGENTS.md

## 项目身份

- 项目名：parkour
- 仓库：`git@github.com:Tang29Zx/parkour.git`
- 本地根目录：`/home/tang/parkour`
- 基线提交：`5e1f4136f7fec9dac2741add63ca58d32090666f`
- 创建日期：2026-07-19
- 主要技术栈：Python、Isaac Gym、legged_gym、rsl_rl、Shell、Docker
- 包管理方式：pip editable install（具体版本以仓库文件为准）

## 当前目标

为本项目建立 Ubuntu 20.04、Python 3.8、NVIDIA GPU、X11/OpenGL/Vulkan
可视化和 headless 训练共用的 Docker 开发环境，并提供少量并行环境的
固定箱子可视化调试入口。

## 工作约束

- 修改前先阅读仓库 README、setup.py、requirements、安装脚本和相关源码。
- 依赖版本必须来自仓库证据，不默认追随最新版。
- 新增部署文件统一放在 `docker/`；项目调试代码保持最小并与原任务隔离。
- 不修改宿主机 CUDA、NVIDIA 驱动或 Python 环境。
- 不读取、提交或记录 secret、私钥、token 和 `.env` 内容。
- 不删除检查点、大文件或用户已有改动。
- Python 代码和注释使用英文；README、规格和验证记录使用中文。

## 验证要求

- 修改前后检查 `git status` 和聚焦 diff。
- Shell 脚本至少运行 `bash -n`；Python 文件至少运行语法编译检查。
- Docker/Compose 文件在工具可用时做解析或构建检查。
- GPU、X11、Isaac Gym viewer 和项目 viewer 必须分层验证，不把未执行描述为成功。
- 缺少 Isaac Gym 安装包、GPU、驱动、Toolkit、显示服务器或权限时明确记录阻塞。

## Git

- 不自动提交或推送。
- 不使用 `git reset --hard`、`git clean` 或强制推送。
- 保留与本任务无关的用户改动。

