# fashion-classifier
A beginner PyTorch project for Fashion-MNIST classification.

## 项目简介
- 这是我的第一份自我尝试从零搭建的项目，目的是训练我项目的管理和实际工程处理的能力
- This project is my first end-to-end personal project which is built on my own,for the purpose of training my ability of managing a project and dealing with a actual working.

- This is my first personal project that I am developing independently, from planning through implementation. I aim to use it to practice project planning and management, develop my coding and debugging skills, and gain hands-on software engineering experience.
## 任务定义与范围
- 题目为“通过Fashion-MNIST学习Pytorch中数据加载，神经网络训练，验证与GPU推理，并比较CPU和GPU运行的情况”。题目主要围绕神经网络展开，并自行搭建一个机器学习项目，实现对机器学习的工程入门，并了解开源项目搭建过程
- the theme of the project is "With 'To learn about the data-loading in Pytorch,trains ,evaluaiton and GPU inference of the neural network, along with comparing the performance of CPU's and GPU's running with Fashion-MNIST"

- Using the Fashion-MNIST dataset, I will build a neural network in PyTorch to classify 28 × 28 grayscale images of clothing into 10 categories. Through this project, I will learn the main steps in an image-classification workflow, including data loading, model training, validation, and inference on a GPU. I will also compare model training times on the CPU and GPU and learn how to organize a project for open-source collaboration.

The model takes grayscale images of clothing as input and predicts one of 10 categories.
## 数据集
- 本项目使用 torchvision 提供的 Fashion-MNIST 数据集。官方训练集包含 60,000 张图片，测试集包含 10,000 张图片。
- `build_datasets` 将官方训练集按 5:1 划分为训练集和验证集，并保留官方测试集用于最终评估。划分使用随机种子 `42`，以便重复运行时得到相同的训练集和验证集索引。
## 实验环境
```
Python: 3.14.4
PyTorch: 2.10.0+cu128
torchvision: 0.25.0+cu128
PyTorch CUDA runtime: 12.8

PyTorch can use CUDA: True
GPU: NVIDIA GeForce RTX 3050 Laptop GPU
GPU memory GiB: 4.0
```
## 安装与运行
- 在项目根目录激活虚拟环境，然后运行数据检查脚本：
```bash
source .venv/bin/activate
python inspect_data.py
```
## 实验结果
- 待补充
## 项目结构
```text
fashion-classifier/
├── data.py             # 创建数据集划分和 DataLoader
├── inspect_data.py     # 检查数据并保存样本图片
├── reports/
│   └── samples.png     # 数据检查生成的图片
├── data/               # 下载的数据集
└── README.md
```
## 实验记录
| 检查项 | 结果 |
| --- | --- |
| 训练集 / 验证集 / 测试集 | 50,000 / 10,000 / 10,000 |
| 图片批次形状 | 以本地运行结果为准；`batch_size=64` 时预期为 `[64, 1, 28, 28]` |
| 图片批次形状 | 已检查：`[64, 1, 28, 28]`，类型为 `torch.float32` |
| 图片像素范围 | 已检查：当前批次最小值为 `0.0`，最大值为 `1.0` |
| 数据划分检查 | 已通过：索引无重复、无遗漏；同种子可复现，不同种子生效 |
## 已知限制
- 待补充
## 后续计划
- 待补充
## 许可证
