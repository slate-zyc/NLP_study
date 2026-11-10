'''
EncoderDecoder(                               # 自定义的Transformer架构, 编码解码模型.
  (source_embed): Sequential(                 # (输入部分)任务处理链 -> 词嵌入层 + 位置编码
    (0): Embedding(
      (embed): Embedding(1000, 512)
    )
    (1): PositionalEncoding(
      (dropout): Dropout(p=0.1, inplace=False)
    )
  )


  (encoder): Encoder(                         # 编码器
    (layers): ModuleList(
      (0-5): 6 x EncoderLayer(                # 编码器 = 6个编码器层

        (self_attn): MultiHeadAttention(      # 每个编码器层 = 2个子层, 这里是第1个子层, 即: 多头注意力子层
          (linears): ModuleList(
            (0-3): 4 x Linear(in_features=512, out_features=512, bias=True)   # 多头注意力子层有 4个 全连接层.
          )
          (dropout): Dropout(p=0.1, inplace=False)
        )

        (feed_forward): FeedForward(          # 第2个子层: 前馈全连接子层(就是两个全连接层, 升维, 降维, 强化特征)
          (linear1): Linear(in_features=512, out_features=2048, bias=True)
          (linear2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )

        (sublayer): ModuleList(               # 每个编码器层 = 2个子层
          (0-1): 2 x SublayerConnection(
            (norm): LayerNorm()
            (dropout): Dropout(p=0.2, inplace=False)
          )
        )
      )
    )
    (norm): LayerNorm()
  )




  (target_embed): Sequential(         # 解码器的输入部分任务链 = 词嵌入层 + 位置编码
    (0): Embedding(
      (embed): Embedding(1000, 512)
    )
    (1): PositionalEncoding(
      (dropout): Dropout(p=0.1, inplace=False)
    )
  )


  (decoder): Decoder(                # 解码器
    (layers): ModuleList(
      (0-5): 6 x DecoderLayer(       # 解码器 = 6个解码器层

        (self_attn): MultiHeadAttention(    # 第1个子层: 掩码多头注意力子层, 防止偷看未来词.
          (linears): ModuleList(
            (0-3): 4 x Linear(in_features=512, out_features=512, bias=True)
          )
          (dropout): Dropout(p=0.1, inplace=False)
        )

        (src_attn): MultiHeadAttention(     # 第2个子层: 多头注意力子层, 算注意力权重的.
          (linears): ModuleList(
            (0-3): 4 x Linear(in_features=512, out_features=512, bias=True)
          )
          (dropout): Dropout(p=0.1, inplace=False)
        )


        (feed_forward): FeedForward(        # 第3个子层: 前馈全连接子层
          (linear1): Linear(in_features=512, out_features=2048, bias=True)
          (linear2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )

        (layers): ModuleList(
          (0-2): 3 x SublayerConnection(    # 每个解码器层 = 3个子层
            (norm): LayerNorm()
            (dropout): Dropout(p=0.2, inplace=False)
          )
        )
      )
    )
    (norm): LayerNorm()
  )


  (generator): Generator(                  # 输出部分: 把词向量维度 转成 词汇表大小的概率分布, 最终就可以预测结果了
    (linear): Linear(in_features=512, out_features=1000, bias=True)
  )
'''