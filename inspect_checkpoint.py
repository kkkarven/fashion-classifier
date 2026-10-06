import torch
from torch import nn

from data import build_datasets, build_loaders
from model import FashionMLP

def main():
    train_data, val_data, test_data = build_datasets()
    _ , val_loader, _ = build_loaders(
        train_data, val_data, test_data
    )
    model = FashionMLP()   
    mlp_record = torch.load(f = "/home/hp/fashion-classifier/fashion_mlp.pt",map_location = "cpu",weights_only = True)
    model.load_state_dict(mlp_record)
    batch_iterator = iter(val_loader)
    images, labels = next(batch_iterator)
    
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
    
    print("checkpoint检查通过")
    


if __name__ == "__main__":
    main()