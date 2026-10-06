import torch
from torch import nn
import math
 
from data import build_datasets, build_loaders
from model import FashionMLP


def main():
    # 1. 取得一批训练图片和标签
    train_data, val_data, test_data = build_datasets()
    train_loader, _, _ = build_loaders(
        train_data, val_data, test_data
    )

    batch_iterator = iter(train_loader)
    images, labels = next(batch_iterator)

    # 2. 创建模型，设置训练模式
    model = FashionMLP()
    model.train()

    # 3. 准备损失对象与优化器
    loss_fn = nn.CrossEntropyLoss()       # TODO：创建交叉熵损失对象
    optimizer = torch.optim.Adam(params=model.parameters(),lr=0.001)     # TODO：创建优化器

    # 4. 保存更新前的 fc1 权重副本
    weight_before = model.fc1.weight.clone()  # TODO：保存独立的数值副本

    # 5. 清除旧梯度，再进行前向计算
    # TODO：清除旧梯度
    optimizer.zero_grad()
    logits = model(images)

    # 6. 根据原始分数和真实标签计算损失
    loss =  loss_fn(logits,labels)         # TODO：计算本批平均损失
    assert torch.isfinite(loss).item(), "loss is infinite of NaN"
    # 7. 计算参数梯度，并读取需要检查的梯度
    # TODO：执行梯度计算
    loss.backward()
    gradient = model.fc1.weight.grad      # TODO：读取 fc1 权重的梯度
    assert gradient is not None, "no gradients occur"
    assert torch.isfinite(gradient).all().item(), "grad is infinite or NaN"
    # 8. 更新模型参数
    # TODO：执行一次参数更新
    optimizer.step()
    # 9. 比较更新前后的权重
    weight_change = abs(model.fc1.weight - weight_before).max().item()  # TODO：计算权重最大绝对变化量
    assert math.isfinite(weight_change), "weights are infinite or NaN"
    assert weight_change > 0, "weights have no change"
    # 10. 展示与检查结果
    print("图片形状：", images.shape)
    print("输出形状：", logits.shape)
    print("本批损失：", loss)
    print("权重最大变化量：", weight_change)

    # TODO：检查损失和梯度是否为有限数值
    # TODO：确认选定的权重确实发生变化
  
    

if __name__ == "__main__":
    main()