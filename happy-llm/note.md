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

## FFN
$$
FFN(X) = Dropout(linear(relu(linear(x))))
$$

## LayerNorm
LN在特征维度上，保证mean=0, var=1

$$
LN(x) = \gamma * \frac{x-\mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

step by step formula
$$
\begin{align}
\text{Step 1: } & \mu = \frac{1}{d} \sum_{i=1}^{d} x_i \\
\text{Step 2: } & \sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2
 \\
\text{Step 3: } & \hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 +
\epsilon}} \\
\text{Step 4: } & y_i = \gamma_i \hat{x}_i + \beta_i
\end{align}
$$

```python
class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps=1e-5):
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        # step1: compute mean and variance across the last dimension
        mean = x.mean(dim=-1, keep_dim=True)
        var = x.mean(dim=-1, keep_dim=True, unbias=False)

        # step2: use normlize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # step3: apply learnable affine transformation
        output = self.gamma * x_norm + self.beta

        return output
```

**unbias说明**

> Bessel's correction is the technique of using N-1 instead of N when computing the sample variance to get an unbiased estimator of the population variance.

`unbias=True`表示使用`N-1`来做方差估计时的无偏修正。
在LayerNorm的均值和方差计算中，我们默认得到了全部的真实数据，所以不需要进行无偏修正


**When to Apply Bessel's Correction**

Apply (use N-1):
- Statistical inference (estimating population from sample)
- Computing confidence intervals
- Hypothesis testing
- When using np.std() with ddof=1 (degrees of freedom delta)
- Pandas .std() (uses N-1 by default)

Don't apply (use N):
- Layer Normalization (normalizing actual data, not estimating)
- Batch Normalization (during forward pass)
- Descriptive statistics of your complete dataset
- When you have the entire population


## Postion Embedding
从注意力机制的计算过程我们可以发现，对与序列中的每个token，其他各个位置对于其来说都是平等的，即“我喜欢你”和“你喜欢我”在注意力机制看来完全是一样的。因此我们需要使用位置编码来保留输入的相对位置信息。

**两种主要的embedding形式**
1. 可学习的embedding
  - bert, gpt等常用

2. Sinusoidal Position Embeddings(Original Transformer)

Formula:
$$
\begin{align}
PE_{(pos, 2i)} &=
\sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right) \\
PE_{(pos, 2i+1)} &=
\cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
\end{align}
$$

公式解读:
- 低维度快速振荡，让模型捕捉细颗粒度
Position Encoding Matrix:
                  d_model dimensions →
              ┌─────────────────────────────┐
Position 0    │ sin(0/α₀) cos(0/α₁) sin...  │  High freq
changes
Position 1    │ sin(1/α₀) cos(1/α₁) sin...  │
Position 2    │ sin(2/α₀) cos(2/α₁) sin...  │
Position 3    │ sin(3/α₀) cos(3/α₁) sin...  │
...           │    ...      ...      ...    │
              └─────────────────────────────┘
              ↑         ↑           ↑
            Fast      Medium      Slow
           frequency  frequency  frequency

Where α₀ < α₁ < α₂ < ... (increasing wavelengths)
