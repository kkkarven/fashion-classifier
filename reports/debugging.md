# 排错记录：预测标题显示类别编号

## 1. 预期与实际现象
我希望标题显示实际的类别名称
实际图片中，真实类别显示类别名称，预测类别显示类别名称的标号

## 2. 初步判断
我当时怀疑可能是变量类型没写对

## 3. 检查依据
我检查了标题语句title = f"true_label:{true_name}\npred_label:{predicted_label}"

其中，`true_name` 是真实标签名称，类型是str，`predicted_label` 是预测标签的标号，类型是int，`predicted_name` 是预测标签名称，类型是str

## 4. 修改与验证
我把标题中的predicted_label改为predicted_name
重新运行后，我观察到输出的预测类别和真实类别都直接显示名称

## 5. 得到的认识
类别编号可以通过test_data.classes取得对应名称。检查后的结论是：这些变量各自的类型没有问题，标题选择了保存编号的变量，而我希望展示名称。