import torch
from torch import nn
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import build_datasets, build_loaders
from model import FashionMLP

def main():
    train_data, val_data, test_data = build_datasets()
    _ , _ , test_loader = build_loaders(
        train_data, val_data, test_data
    )
    model = FashionMLP()
    loss_fn = nn.CrossEntropyLoss()  
    mlp_record = torch.load(f = "fashion_mlp.pt",map_location = "cpu",weights_only = True)
    model.load_state_dict(mlp_record)

    model.eval()
    test_total_loss = 0.0
    test_total_samples = 0
    total_correct = 0
    total_accuracy = 0.0
    records = []
    with torch.no_grad():
        for images, labels in test_loader:
            # 3.1 TODO：使用已有模型进行前向计算
            logits = model(images)

            # 3.2 TODO：使用已有损失对象计算本批平均损失
            loss = loss_fn(logits,labels)
            assert torch.isfinite(loss).item(), "loss is infinite of NaN"

            # TODO：检查损失是否为有限数值

            # 3.3 取得本批图片数量
            batch_size = images.shape[0]
            test_total_loss += loss.item() * batch_size
            test_total_samples += batch_size
            total_correct += (torch.argmax(logits,dim=1) == labels).sum().item()
            # TODO：累计本批所有样本的损失
            # TODO：累计本批图片数量
            # 取得本批每张图片的预测编号，形状是 [B]
            predictions = torch.argmax(logits, dim=1)

            # 依次访问本批中的每一个图片位置
            for index in range(batch_size):
                # 三项内容都来自同一个位置
                image = images[index]
                true_label = labels[index].item()
                predicted_label = predictions[index].item()

                # TODO 1：比较上面的两个类别编号，得到“是否预测错误”
                is_wrong = true_label != predicted_label

                if is_wrong:
                    # TODO 2：用元组组合图片、真实编号、预测编号
                    record = (image,true_label,predicted_label)

                    records.append(record)
        test_mean_loss = test_total_loss / test_total_samples  # TODO：根据累计结果计算            
        total_accuracy = total_correct/ test_total_samples
            # 1. 创建图片保存目录
        output_dir = Path("reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "errors.png"

        # 2. 选择前九条错误记录，用于展示
        display_records = records[:9]

        # 3. 创建整张图和九个子图
        fig, axes = plt.subplots(
            nrows=3,
            ncols=3,
            figsize=(10, 10),
        )

        # 隐藏坐标轴；没有放图片的子图也不显示刻度
        for ax in axes.flat:
            ax.axis("off")

        # 4. 把每条错误记录放到对应的子图中
        for index, record in enumerate(display_records):
            image, true_label, predicted_label = record

            # 按顺序取得当前子图
            ax = axes.flat[index]

            # 根据类别编号取得类别名称
            true_name = test_data.classes[true_label]
            predicted_name = test_data.classes[predicted_label]

            # TODO A：把 image 从 [1, 28, 28] 转成 [28, 28]
            image_2d = image.squeeze(dim = 0)

            # TODO B：构造包含真实类别和预测类别的标题字符串
            title = f"true_label:{true_name}\npred_label:{predicted_name}"

            # 将二维图片显示在当前子图中
            ax.imshow(image_2d, cmap="gray")

            # 设置当前子图的标题
            ax.set_title(title)

        # 5. 调整子图间距，然后保存整张图
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

        print("错误图片已保存：", output_path)
        print("测试图片数：", test_total_samples)
        print("平均测试损失：", test_mean_loss)
        print("准确率：",total_accuracy)
        assert len(records) == test_total_samples - total_correct,"wrong/right don't match"
        print("预测错误数",len(records))
    


if __name__ == "__main__":
    main()