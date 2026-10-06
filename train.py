import torch
from torch import nn

from data import build_datasets, build_loaders
from model import FashionMLP


def main():
    # 1. 准备数据
    train_data, val_data, test_data = build_datasets()
    train_loader, val_loader, _ = build_loaders(
        train_data, val_data, test_data
    )

    # 2. 创建模型、损失对象和优化器
    model = FashionMLP()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=0.001,
    )
    best_accuracy = 0.0
    epochs = 5
    for epoch in range(1, epochs + 1):
        # 3. 设置训练模式，准备本轮统计
        model.train()

        total_loss = 0.0      # 累计各批中所有样本的损失
        total_samples = 0    # 累计处理的图片数量
        total_batches = 0    # 累计处理的批次数量


        # 4. 遍历一轮训练数据
        for images, labels in train_loader:
            # 4.1 TODO：清除上一批留下的梯度
            optimizer.zero_grad()
            # 4.2 TODO：进行前向计算
            logits = model(images)

            # 4.3 TODO：计算本批平均损失，检查其是否有限
            loss = loss_fn(logits,labels)
            assert torch.isfinite(loss).item(), "loss is infinite of NaN"
            # 4.4 TODO：反向传播，计算梯度
            loss.backward()
            # 4.5 TODO：使用优化器更新参数
            optimizer.step()
            # 4.6 取得本批图片数量
            batch_size = images.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            total_batches += 1
            # TODO：累计本批的样本损失总和
            # TODO：累计本批图片数量
            # TODO：累计批次数量

            

        # 5. 遍历结束后，计算整轮平均损失
        mean_loss = total_loss/total_samples     # TODO：根据累计结果计算

        # 6. 展示本轮结果
        print("处理批次数：", total_batches)
        print("处理图片数：", total_samples)
        print("整轮平均损失：", mean_loss)

            # 1. 切换到评估模式，使用刚才训练过的模型
        model.eval()

        # 2. 准备独立的验证统计
        val_total_loss = 0.0
        val_total_samples = 0
        total_correct = 0
        total_accuracy = 0.0
        # 3. 验证计算不记录梯度关系
        with torch.no_grad():
            for images, labels in val_loader:
                # 3.1 TODO：使用已有模型进行前向计算
                logits = model(images)

                # 3.2 TODO：使用已有损失对象计算本批平均损失
                loss = loss_fn(logits,labels)
                assert torch.isfinite(loss).item(), "loss is infinite of NaN"

                # TODO：检查损失是否为有限数值

                # 3.3 取得本批图片数量
                batch_size = images.shape[0]
                val_total_loss += loss.item() * batch_size
                val_total_samples += batch_size
                total_correct += (torch.argmax(logits,dim=1) == labels).sum().item()
                # TODO：累计本批所有样本的损失
                # TODO：累计本批图片数量



        # 4. 遍历结束后，计算平均验证损失
        val_mean_loss = val_total_loss / val_total_samples  # TODO：根据累计结果计算
        total_accuracy = total_correct/ val_total_samples

        # 5. 打印验证结果
        print("验证图片数：", val_total_samples)
        print("平均验证损失：", val_mean_loss)
        print("准确率：",total_accuracy)
        print(f"第{epoch}/{epochs}轮")
        record = model.state_dict()
        if total_accuracy > best_accuracy:
            torch.save(record,"fashion_mlp.pt")
            print(f"第{epoch}轮内容已保存为最佳")
            best_accuracy = total_accuracy
if __name__ == "__main__":
    main()