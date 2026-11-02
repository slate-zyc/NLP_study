"""
案例:
    演示注意力机制的计算流程.

任务描述：
    以 seq2seq 为例，仅演示一下计算过程，都是自定义，都是假设
    QKV，维度均为假设，
任务：输入查询张量q，通过注意力机制来计算如下信息：
    1、查询张量q的注意力权重分布：查询张量q和k的相关性（相识度）
    2、查询张量q的结果表示：有一个普通的q升级成一个更强大q；用q和v做bmm运算
    3 注意：qkv 以实际为主，这里只是演示

MyAttention类实现思路分析
    1 init函数 (self, query_size, key_size, value_size1, value_size2, output_size)
        准备2个线性层 注意力权重分布self.attn；注意力结果表示，按照指定维度进行输出层 self.attn_combine
    2 forward(self, Q, K, V):
        求查询张量q的注意力权重分布, attn_weights[1,32]
        求查询张量q的注意力结果表示 bmm运算, attn_applied[1,1,64]
        q 与 attn_applied 融合（实际这里还需要上一时间步的预测结果经过词嵌入层后的词向量），再按照指定维度输出 output[1,1,32]
    返回注意力结果表示output:[1,1,32], 注意力权重分布attn_weights:[1,32]
"""

# 导包
import torch
import torch.nn as nn
import torch.nn.functional as F


# todo 1.自定义注意力机制模块, 实现: Q(查询张量) 和 键值对(K, V)的注意力计算.

class MyAttn(nn.Module):
    #                   32          32          32              64      32
    def __init__(self, query_size, key_size, value_size1, value_size2, output_size):
        super().__init__()
        self.query_size = query_size        # 查询张量Q的维度, 例如: 32 -> Q [1,1,32]
        self.key_size = key_size            # 键张量K的维度, 例如: 32 -> K [1,1,32]
        self.value_size1 = value_size1      # 值张量V的序列长度, 例如: 32
        self.value_size2 = value_size2      # 值张量V的维度, 例如: 64 -> V [1,32,64]
        self.output_size = output_size      # 输出张量的维度, 例如: 32

        # 线性层1 注意力权重分布
        # 用于计算得分时的维度转换，保证后面能和V正常加权求和
        self.attn = nn.Linear(self.query_size + self.key_size, self.value_size1)

        # 线性层2 注意力结果表示按照指定维度输出层 self.attn_combine
        # 用于预测结果时的维度转换
        self.attn_combine = nn.Linear(self.query_size + self.value_size2, self.output_size)

    def forward(self, Q, K, V):
        # 1 求查询张量q的注意力权重分布, attn_weights[1,32]
        # [1,1,32],[1,1,32]--> [1,32],[1,32]（全连接层只能介绍二维数据）->[1,64]
        # [1,64] --> [1,32]    # 第一个线性层的作用
        # tmp1 = torch.cat((Q[0], K[0]), dim=1)    # 拼接
        # tmp2 = self.attn(tmp1)                   # 计算得分
        # tmp3 = F.softmax(tmp2, dim=1)            # softmax转为概率
        attn_weights = F.softmax(self.attn(torch.cat((Q[0], K[0]), dim=-1)), dim=-1)

        # 2 求查询张量q的结果表示 bmm运算, attn_applied[1,1,64]
        # 先把[1,32]变为[1,1,32]，方便bmm运算
        # [1,1,32] * [1,32,64] ---> [1,1,64]    # 与v加权求和，得到中间语义张量
        attn_applied =  torch.bmm(attn_weights.unsqueeze(0), V)

        # 3 q 与 attn_applied 融合，再按照指定维度输出 output[1,1,64]
        # 实际这里还需要上一时间步的预测结果经过词嵌入层后的词向量
        output = torch.cat((Q[0], attn_applied[0]), dim=-1)
        # 3-2 shape [1,96] --> [1,32] --> [1,1,32]
        output = self.attn_combine(output).unsqueeze(0)

        # 4 返回注意力结果表示output:[1,1,32], 注意力权重分布attn_weights:[1,32]
        return output, attn_weights



# todo 2. 测试代码
if __name__ == '__main__':
    # 1. 实例化参数设置.
    query_size, key_size, value_size1, value_size2, output_size = 32, 32, 32, 64, 32

    # 2. 创建随机输入张量.
    Q = torch.randn(1, 1, query_size)               # 查询张量: [批次, 序列, 特征] -> [1, 1, 32]torch.Size([1, 32])
    K = torch.randn(1, 1, key_size)                 # 键张量: [批次, 序列, 特征] -> [1, 1, 32]
    V = torch.randn(1, value_size1, value_size2)    # 值张量: [批次, 单词数, 词向量维度] -> [1, 32, 64]

    # 3. 实例化自定义的注意力机制模块, 并测试.
    my_attn = MyAttn(query_size, key_size, value_size1, value_size2, output_size)
    output, attn_weights = my_attn(Q, K, V)

    # 4. 输出结果.
    print('-' * 40)
    print(f'查询张量Q的注意力结果表示: {output.shape}, {output}')               # torch.Size([1, 1, 32])
    print(f'查询张量Q的注意力权重分布: {attn_weights.shape}, {attn_weights}')   # torch.Size([1, 32])