import torch
from torch import nn
import random
from data import build_datasets, build_loaders
from model import FashionMLP

def main():
    _ , _ , test_data = build_datasets()

    model = FashionMLP()   
    mlp_record = torch.load(f = "fashion_mlp.pt",map_location = "cpu",weights_only = True)
    model.load_state_dict(mlp_record)
    index = random.randrange(len(test_data))
    apictrue, label = test_data[index]
    a4tensor_pictrue  = apictrue.unsqueeze(dim = 0)
    model.eval()
    
        # 4. 执行前向计算，这次不记录梯度。
    with torch.no_grad():
        single_logits = model(a4tensor_pictrue)
        predictions = single_logits.argmax(dim=1)
    
        # 5. 显示输入、输出和预测。
    print("模型结构：")
    print(model)
    print("选中图片：")
    print(index)
    print("图片形状：", a4tensor_pictrue.shape)
    print("标签形状：", label)
    print("单张输出形状：", single_logits.shape)
    print("预测：", test_data.classes[predictions.item()])
    print("标签：", test_data.classes[label])
    
        # 6. 检查输出是否符合约定。
    assert single_logits.shape == (1, 10)
    assert torch.isfinite(single_logits).all()
    
    print("单张图片检查通过")
    


if __name__ == "__main__":
    main()