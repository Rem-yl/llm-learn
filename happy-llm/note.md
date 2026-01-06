# Note for happy-LLM
## DataSource
- [happy-llm](https://github.com/datawhalechina/happy-llm/blob/main/docs/chapter2/%E7%AC%AC%E4%BA%8C%E7%AB%A0%20Transformer%E6%9E%B6%E6%9E%84.md)


## Transformer
### Attention
我们知道注意力机制是`transformer`中非常重要的模块，它通过QKV三个矩阵能够很好地实现长序列相关的问题。

`attention`的公式如下:
$$
attention(Q,K,V) = softmax(Q @ K^T / \sqrt{d_k}) @ V
$$

接下来我们深入探讨一下，QKV分别起到什么作用。
- `Q`: 学习将输入x映射到查询空间
- `K`: 学习将输入x映射到可查询的索引空间
- `V`: 学习将输入x映射到可检索的信息空间

$score = Q @ K^T$, 每一行代表当前的token与其他位置token的相关性
例如输入 w = ["the", "cat", "sleep"], 得到的attention_map:
tensor([[0.2645, 0.4711, 0.2645],
        [0.2645, 0.2645, 0.4711],
        [0.2192, 0.3904, 0.3904]])

以第一行为例, [0.2645, 0.4711, 0.2645]表示单词"the"和["the", "cat", "sleep"]的相关性分别是0.2645, 0.4711, 0.2645


```python
attention_weights = [
    [0.2, 0.3, 0.5],  # "The" attends to all words with these weights
    [0.1, 0.8, 0.1],  # "cat" attends mostly to itself
    [0.3, 0.4, 0.3],  # "sleeps" attends to all fairly evenly
]

V = [
      [1.0, 0.0, 0.5],  # value vector for "The"
      [0.0, 1.0, 0.3],  # value vector for "cat"
      [0.5, 0.5, 0.8],  # value vector for "sleeps"
  ]

output[1, :] = 0.1*[1.0, 0.0, 0.5] + 0.8*[0.0, 1.0, 0.3] + 0.1*[0.5,0.5,0.5] 
```

其中V中的每一行代表每个token的value vector，最终attention @ V得到的矩阵output 每一行是所有value vector的权重平均和，output每一行表示每个token的上下文表示向量

### Mask attention
通过屏蔽掉指定位置的token，让模型在训练过程中只能使用历史信息而不能看到未来信息，这就指明了attention中的mask是一个上三角矩阵
[
  [1, 0, 0],
  [1, 1, 0],
  [1, 1, 1]
]

在具体实现中，我们会将mask设置为`-inf`的上三角矩阵，然后将其和attention_map相加后做softmax操作

```python

import torch

attention = torch.randn(2, 3, 3)
mask = torch.full_like(attention, float("-inf"))
mask = torch.triu(mask, diagonal=1)
score = torch.softmax(attention+mask, dim=-1)

print(score)
print(torch.sum(score, dim=-1))
```

## Multi-head attention
简单来说，多头注意力就是将原始的输入序列进行多组的自注意力处理，然后将每组计算得到的注意力结果concat起来

$$
Multi-Attention = Concat(head1, ..., headn) \\
where headi = Attention(qi,ki,vi)
$$

一个一个循环计算headi费时费力而且浪费了GPU的并行能力，在具体的代码实现中，我们使用一个组合的大矩阵来运算，通过先计算内积再拼接的方式代替一个个计算然后拼接

```python
import torch

batch_size, seq_len, d_model = 32, 10, 512
num_head = 8
x = torch.randn(batch_size, seq_len, d_model)


def split_heads(x: torch.Tensor, d_k: int):
    # [b,s,d_model] -> [b,num_heads,s, d_k]
    batch_size, seq_len, d_model = x.shape
    d_k = d_model // num_head
    x = x.view(batch_size, seq_len, num_head, d_k)

    return x.transpose(1, 2)


def combine_heads(x: torch.Tensor):
    # [b,num_heads,s,d_k] -> [b,s,d_model]
    batch, n_heads, seq_len, d_k = x.shape
    x = x.transpose(1, 2)

    return x.contiguous().view(batch_size, seq_len, d_model)
```
