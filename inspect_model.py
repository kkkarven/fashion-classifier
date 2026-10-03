import torch
from data import build_datasets, build_loaders
from model import FashionMLP


def main():
    # 1. 创建数据集和加载器。
    train_data, val_data, test_data = build_datasets()
    train_loader, _, _ = build_loaders(
        train_data, val_data, test_data
    )

    # 2. 取得一批真实图片和对应标签。
    batch_iterator = iter(train_loader)
    images, labels = next(batch_iterator)

    # 3. 创建模型，并设置为评估模式。
    model = FashionMLP()
    model.eval()

    # 4. 执行前向计算，这次不记录梯度。
    with torch.no_grad():
        logits = model(images)
        single_logits = model(images[:1])
        predictions = logits.argmax(dim=1)

    # 5. 显示输入、输出和预测。
    print("模型结构：")
    print(model)
    print("图片形状：", images.shape)
    print("标签形状：", labels.shape)
    print("输出形状：", logits.shape)
    print("单张输出形状：", single_logits.shape)
    print("前10个预测：", predictions[:10].tolist())
    print("前10个标签：", labels[:10].tolist())

    # 6. 检查输出是否符合约定。
    assert logits.shape == (images.shape[0], 10)
    assert single_logits.shape == (1, 10)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(single_logits).all()

    print("前向检查通过")


if __name__ == "__main__":
    main()