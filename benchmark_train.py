import time

import torch
from torch import nn

from data import build_datasets, build_loaders
from model import FashionMLP


def main():
    # 1. 实验设置：每次运行测一种设备
    device_name = "cuda"
    seed = 42

    # 每次运行从相同的 PyTorch 随机种子开始
    torch.manual_seed(seed)

    # 选择 CUDA 时，先确认当前环境可以使用它
    if device_name == "cuda":
        assert torch.cuda.is_available(), "当前环境无法使用 CUDA"

    # 2. 准备数据：这次只遍历训练加载器
    train_data, val_data, test_data = build_datasets()
    train_loader, _, _ = build_loaders(
        train_data, val_data, test_data
    )

    # 3. 创建模型，先安排设备，再创建优化器
    model = FashionMLP()
    model.to(device_name)
    # TODO A：将模型移动到 device_name 指定的设备

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=0.001,
    )

    model.train()

    total_loss = 0.0
    total_samples = 0
    total_batches = 0

    # 4. 开始计时之前，等待先前的 GPU 工作完成
    if device_name == "cuda":
        # TODO B：执行 CUDA 同步
        torch.cuda.synchronize()

    # TODO C：读取开始时间，替换 None
    start_time = time.perf_counter()

    # 防止计时空位尚未填写就开始训练
    if start_time is None:
        raise NotImplementedError("请先填写开始时间")

    # 5. 完整训练一轮
    for images, labels in train_loader:
        images = images.to(device_name)
        labels = labels.to(device_name)
        # TODO D：将本批 images 和 labels 移到目标设备
        # 注意使用张量设备迁移方法的返回值

        # 以下是你已经实现过的训练过程
        optimizer.zero_grad()
        logits = model(images)

        loss = loss_fn(logits, labels)
        assert torch.isfinite(loss).item(), "loss is infinite or NaN"

        loss.backward()
        optimizer.step()

        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        total_batches += 1

    # 6. GPU 计算真正完成后，才能读取结束时间
    if device_name == "cuda":
        # TODO E：执行 CUDA 同步
        torch.cuda.synchronize()

    # TODO F：读取结束时间，替换 None
    end_time = time.perf_counter()

    # TODO G：根据开始时间与结束时间计算耗时
    elapsed = end_time - start_time

    if elapsed is None:
        raise NotImplementedError("请先填写耗时计算")

    # 7. 计时结束后，再计算并打印结果
    mean_loss = total_loss / total_samples

    print("实验设备：", device_name)
    print("处理批次数：", total_batches)
    print("处理图片数：", total_samples)
    print("整轮平均损失：", mean_loss)
    print("训练耗时（秒）：", elapsed)


if __name__ == "__main__":
    main()