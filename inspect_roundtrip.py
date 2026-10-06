import torch
from torch import nn
import random
from data import build_datasets, build_loaders
from model import FashionMLP

def main():
    _ , _ , test_data = build_datasets()

    modelA = FashionMLP()   
    mlp_record = torch.load(f = "fashion_mlp.pt",map_location = "cpu",weights_only = True)
    modelA.load_state_dict(mlp_record)
    index = random.randrange(len(test_data))
    apictrue, label = test_data[index]
    a4tensor_pictrue  = apictrue.unsqueeze(dim = 0)
    modelA.eval()
    
        # 4. 执行前向计算，这次不记录梯度。
    with torch.no_grad():
        logitsA = modelA(a4tensor_pictrue)
    
        # 5. 显示输入、输出和预测。
    print("模型结构：")
    print(modelA)
    print("选中图片：")
    print(index)
    print("图片形状：", a4tensor_pictrue.shape)
    print("标签形状：", label)
    print("输出形状：", logitsA.shape)

        # 6. 检查输出是否符合约定。
    assert logitsA.shape == (a4tensor_pictrue.shape[0], 10)
    assert torch.isfinite(logitsA).all()

    record = modelA.state_dict()
    torch.save(record,"fashion_mlp_A.pt")

    modelB = FashionMLP()   
    mlp_record = torch.load(f = "fashion_mlp_A.pt",map_location = "cpu",weights_only = True)
    modelB.load_state_dict(mlp_record)
    modelB.eval()
    
        # 4. 执行前向计算，这次不记录梯度。
    with torch.no_grad():
        logitsB = modelB(a4tensor_pictrue)
    
        # 5. 显示输入、输出和预测。
    print("模型结构：")
    print(modelB)
    print("选中图片：")
    print(index)
    print("图片形状：", a4tensor_pictrue.shape)
    print("标签形状：", label)
    print("输出形状：", logitsB.shape)

        # 6. 检查输出是否符合约定。
    assert logitsB.shape == (a4tensor_pictrue.shape[0], 10)
    assert torch.isfinite(logitsB).all()

    assert torch.allclose(logitsA,logitsB,rtol=0,atol=1e-6),"the same model's output is not equal"
    print("通过复用性检查")
if __name__ == "__main__":
    main()