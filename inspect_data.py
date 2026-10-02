from data import build_datasets, build_loaders
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

def inspect_data(train_loader, val_loader, test_loader):
    # 查看三份数据集的样本数
    print(
        "样本数:",
        len(train_loader.dataset),
        len(val_loader.dataset),
        len(test_loader.dataset),
    )

    # 取训练集的第一批图片和标签
    images, labels = next(iter(train_loader))
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].squeeze(0), cmap="gray")
        class_name = test_loader.dataset.classes[labels[i].item()]
        ax.set_title(class_name)
        ax.axis("off")
    fig.tight_layout()
    Path("reports").mkdir(parents=True, exist_ok=True)
    fig.savefig("reports/samples.png")
    plt.close(fig)

    print("图片形状和类型:", images.shape, images.dtype)
    print("标签形状和类型:", labels.shape, labels.dtype)
    print("图片像素范围:", images.min().item(), images.max().item())
    print("标签范围:", labels.min().item(), labels.max().item())
    print("前10个标签:", labels[:10].tolist())
    assert set(train_loader.dataset.indices).isdisjoint(val_loader.dataset.indices)

if __name__ == "__main__":
    train_data, val_data, test_data = build_datasets()
    train_loader, val_loader, test_loader = build_loaders(
           train_data, val_data, test_data
    )
    inspect_data(train_loader, val_loader, test_loader)
    train_ids = set(train_data.indices)
    val_ids = set(val_data.indices)

    assert len(train_ids) == len(train_data)
    assert len(val_ids) == len(val_data)
    assert (train_ids | val_ids) == set(range(60000))
    repeat_train, repeat_val, _ = build_datasets(seed=42)
    assert train_data.indices == repeat_train.indices
    assert val_data.indices == repeat_val.indices
    changed_train, _, _ = build_datasets(seed=43)
    assert train_data.indices != changed_train.indices
    print("划分检查通过：索引无重复、无遗漏；同种子可复现，不同种子生效")