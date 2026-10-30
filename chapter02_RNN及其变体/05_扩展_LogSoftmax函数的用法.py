"""
案例:
    演示 LogSoftmax()函数的用法

用处:
    nn.CrossEntropyLoss() 的计算公式为：\mathcal{L} = - \sum_{i=1}^{n} y_i \log(S(f_\theta(\mathbf{x}_i)))
    当需要手动计算时，就可以使用 nn.LogSoftmax() 函数来计算 \log(S(f_\theta(\mathbf{x}_i))) 这一部分。
说明：
    分类交叉熵损失函数CrossEntropyLoss(新版写法) = NLLLoss损失函数 + LogSoftmax(dim=-1)  旧版写法
"""

# 导包
import torch
import torch.nn as nn


# 1. 创建数据 -> 模拟模型输出的原始分时(logits), 表示3个分类的预测值, 即: 全连接层的处理后的结果.
output = torch.tensor([3.2, 5.1, -1.7])

# 2. ---------------- 思路1: 用LogSoftmax()函数计算. ----------------
# 2.1 创建LogSoftmax()函数对象
log_softmax = nn.LogSoftmax(dim=0)
# 2.2 具体的计算, 先softmax(), 然后log()
log_probs = log_softmax(output)
# 2.3 打印结果
print(f'计算结果(对数概率): {log_probs}')       # tensor([-2.0404, -0.1404, -6.9404])


# 3. ---------------- 思路2: 手动验算, 先softmax(), 然后log() ----------------
# 3.1 创建softmax()函数对象
softmax = torch.softmax(output, dim=0)
print(f'softmax()计算结果: {softmax}')        # tensor([0.1300, 0.8690, 0.0010])

# 3.2 手动计算log() -> 对softmax()结果, 取自然对数, 得到: 对数概率.
log_softmax_probs = torch.log(softmax)
print(f'计算结果(对数概率): {log_softmax_probs}')   # tensor([-2.0404, -0.1404, -6.9404])

print(log_probs == log_softmax_probs)