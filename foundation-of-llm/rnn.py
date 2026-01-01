import torch
import torch.nn as nn
from typing import Dict, List


def sentences2id(vocab: Dict[str, int], sentences: List[str]) -> List[List[int]]:
    return [[vocab[word] for word in sentence.split()] for sentence in sentences]


def padding(word_ids: List[List[int]], default_idx: int = 0) -> torch.Tensor:
    max_len = -1
    result = []
    max_len = max([len(word_id) for word_id in word_ids])

    for word_id in word_ids:
        if len(word_id) < max_len:
            word_id.extend([default_idx] * (max_len - len(word_id)))

        result.append(word_id)

    return torch.tensor(result, dtype=torch.long)


class CustomRNN(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int):
        """
        CustomRNN Model
            - formula: h_t = g(W_H * h_{t-1} + W_I * x_t); o_t = f(W_O * h_t)
        """
        super(CustomRNN, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.embedding = torch.randn(vocab_size, embed_size) * 0.01  # embedding layer
        self.W_H = nn.Linear(self.embed_size, self.hidden_size)
        self.W_I = nn.Linear(self.hidden_size, self.hidden_size)

        self.W_O = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, x: torch.Tensor):
        """
        前向传播

        Args:
            x: 词ID序列, shape=(batch_size, seq_len)

        Returns:
            output: 每个时间步的词表得分, shape=(batch_size, seq_len, vocab_size)
            h_t: 最终隐藏状态, shape=(batch_size, hidden_size)
        """
        batch_size, seq_len = x.shape

        # 1. 词嵌入：将词ID转换为词向量
        # x: (batch_size, seq_len) -> embedded: (batch_size, seq_len, embed_size)
        embedded = self.embedding[x]

        # 2. 初始化隐藏状态
        h_t = torch.zeros(batch_size, self.hidden_size)

        # 3. 逐时间步处理
        outputs = []
        for t in range(seq_len):
            # 获取当前时间步的输入: (batch_size, embed_size)
            x_t = embedded[:, t, :]

            # RNN 核心公式: h_t = tanh(W_H(x_t) + W_I(h_{t-1}))
            h_t = torch.tanh(self.W_H(x_t) + self.W_I(h_t))

            # 计算输出: (batch_size, vocab_size)
            o_t = self.W_O(h_t)
            outputs.append(o_t)

        # 堆叠所有时间步的输出: (batch_size, seq_len, vocab_size)
        output = torch.stack(outputs, dim=1)

        return output, h_t

    def predict_next_word(self, x: torch.Tensor, h_0=None, temperature=1.0):
        """
        预测下一个词

        Args:
            x: 输入词ID序列, shape=(batch_size, seq_len)
            h_0: 初始隐藏状态, shape=(batch_size, hidden_size)，可选
            temperature: 温度参数，控制随机性 (越小越确定，越大越随机)

        Returns:
            next_word_id: 预测的下一个词ID, shape=(batch_size,)
            probs: 词表上的概率分布, shape=(batch_size, vocab_size)
            h_t: 更新后的隐藏状态, shape=(batch_size, hidden_size)
        """
        # 前向传播
        logits, h_t = self.forward(x)

        # 取最后一个时间步的输出
        last_logits = logits[:, -1, :]  # (batch_size, vocab_size)

        # 应用温度
        last_logits = last_logits / temperature

        # 转换为概率分布
        probs = torch.softmax(last_logits, dim=-1)  # (batch_size, vocab_size)

        # 采样下一个词
        next_word_id = torch.multinomial(probs, num_samples=1).squeeze(
            -1
        )  # (batch_size,)

        return next_word_id, probs, h_t

    def generate_text(
        self, start_word_id: int, max_len=20, temperature=1.0, end_word_id=None
    ):
        """
        自回归生成文本序列

        Args:
            start_word_id: 起始词ID
            max_len: 最大生成长度
            temperature: 温度参数
            end_word_id: 结束词ID（可选）

        Returns:
            generated_ids: 生成的词ID列表
        """
        # 初始化
        generated_ids = [start_word_id]

        for _ in range(max_len):
            # 自回归：将已生成的所有词作为输入
            current_sequence = torch.tensor([generated_ids])  # (1, current_len)

            # 预测下一个词
            next_id, probs, h_t = self.predict_next_word(
                current_sequence, temperature=temperature
            )
            next_id = next_id.item()

            # 检查是否结束
            if end_word_id is not None and next_id == end_word_id:
                break

            # 将预测的词添加到序列中
            generated_ids.append(next_id)

        return generated_ids


def predict_text(
    model: CustomRNN, vocab: Dict[str, int], text: str, max_len=20, temperature=1.0
):
    """
    根据输入文本预测并生成后续文本

    Args:
        model: 训练好的 CustomRNN 模型
        vocab: 词表字典
        text: 输入文本（用空格分隔的词）
        max_len: 最大生成长度
        temperature: 温度参数

    Returns:
        generated_text: 生成的文本字符串
    """
    # 创建反向词表（ID -> 词）
    id_to_word = {v: k for k, v in vocab.items()}

    # 将文本转换为词ID
    words = text.split()
    word_ids = [vocab.get(word, vocab.get("<UNK>", 0)) for word in words]

    # 转换为张量
    input_tensor = torch.tensor([word_ids])

    # 预测下一个词
    next_id, probs, _ = model.predict_next_word(input_tensor, temperature=temperature)

    # 从预测的词开始生成
    generated_ids = model.generate_text(
        start_word_id=next_id.item(),
        max_len=max_len,
        temperature=temperature,
        end_word_id=vocab.get("</s>"),
    )

    # 转换回文本
    generated_words = [id_to_word.get(id, "<UNK>") for id in generated_ids]
    generated_text = " ".join(generated_words)

    return generated_text


if __name__ == "__main__":
    # 示例用法
    print("=== RNN 语言模型示例 ===\n")

    # 1. 创建词表和模型
    vocab = {
        "<PAD>": 0,
        "<START>": 1,
        "</s>": 2,
        "I": 3,
        "love": 4,
        "AI": 5,
        "is": 6,
        "great": 7,
        "so": 8,
    }
    id_to_word = {v: k for k, v in vocab.items()}

    vocab_size = len(vocab)
    embed_size = 10
    hidden_size = 20

    model = CustomRNN(vocab_size, embed_size, hidden_size)
    print(
        f"模型创建成功: vocab_size={vocab_size}, embed_size={embed_size}, hidden_size={hidden_size}\n"
    )

    # 2. 测试前向传播
    sentences = ["I love AI", "AI is so great"]
    word_ids = sentences2id(vocab, sentences)
    input_tensor = padding(word_ids, vocab["<PAD>"])

    print(f"输入句子: {sentences}")
    print(f"词ID张量: {input_tensor}")
    print(f"输入形状: {input_tensor.shape}\n")

    output, hidden = model.forward(input_tensor)
    print(f"输出形状: {output.shape}")
    print(f"隐藏状态形状: {hidden.shape}\n")

    # 3. 测试预测下一个词
    test_input = torch.tensor([[vocab["I"], vocab["love"]]])
    next_id, probs, _ = model.predict_next_word(test_input)

    print(f"输入序列: 'I love'")
    print(f"预测的下一个词ID: {next_id.item()}")
    print(f"预测的下一个词: {id_to_word.get(next_id.item(), '<UNK>')}")
    print(f"概率最高的前3个词:")
    top_probs, top_indices = torch.topk(probs[0], k=3)
    for prob, idx in zip(top_probs, top_indices):
        print(f"  {id_to_word.get(idx.item(), '<UNK>')}: {prob.item():.4f}\n")

    # 4. 测试文本生成
    print("=== 文本生成 ===")
    generated_ids = model.generate_text(
        start_word_id=vocab["<START>"],
        max_len=10,
        temperature=1.0,
        end_word_id=vocab["</s>"],
    )
    generated_words = [id_to_word.get(id, "<UNK>") for id in generated_ids]
    print(f"生成的序列: {' '.join(generated_words)}\n")

    # 5. 测试不同温度参数
    print("=== 不同温度参数的生成结果 ===")
    for temp in [0.5, 1.0, 2.0]:
        generated_ids = model.generate_text(
            start_word_id=vocab["I"], max_len=5, temperature=temp
        )
        generated_words = [id_to_word.get(id, "<UNK>") for id in generated_ids]
        print(f"temperature={temp}: {' '.join(generated_words)}")
