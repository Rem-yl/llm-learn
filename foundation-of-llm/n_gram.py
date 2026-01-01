import random
import math
import re
from collections import defaultdict, Counter


class NGramModel:
    def __init__(self, n=2):
        """
        Initialize the NGramModel with a given order n.

        Args:
            n (int, optional): The order of the n-gram model. Defaults to 2.

        Property:
            self.ngram_counts (defaultdict): 存储上下文对应的下个单词出现的次数。
                例如: defaultdict(<class 'collections.Counter'>, {('<s>',): Counter({'chapter': 1}), ('chapter',): Counter({'1': 1})})
                表示 '<s>'对应下个单词为 'chapter' 的出现次数为 1, 'chapter' 对应下个单词为 '1' 的出现次数为 1, 对应公式中的 C(w_{i-n+1:i})

            self.context_counts (Counter): 存储上下文出现的次数.
                例如: Counter({('<s>',): 1}) 表示 '<s>' 出现的次数为 1, 对应公式中的 C(w_{i-n+1:i-1})

            self.vocab (set): A set to store the vocabulary of words.
        """
        self.n = n
        self.ngram_counts = defaultdict(Counter)
        self.context_counts = Counter()
        self.vocab = set()

    def train(self, sentences):
        """训练 N-gram 模型"""
        for sentence in sentences:
            tokens = ["<s>"] * (self.n - 1) + sentence + ["</s>"]
            self.vocab.update(tokens)
            for i in range(len(tokens) - self.n + 1):
                context = tuple(tokens[i : i + self.n - 1])
                word = tokens[i + self.n - 1]
                self.ngram_counts[context][word] += 1
                self.context_counts[context] += 1

    def next_word_probs(self, context):
        """
        Get the probability of each word following the given context.

        Args:
            context (list of str): The context words.

        Returns:
            list of tuple: A list of (word, probability) tuples, sorted in descending order of probabilities.
                Like: [('word1', 0.5), ('word2', 0.3), ('word3', 0.2)]
        """
        context = tuple(context[-(self.n - 1) :])
        counts = self.ngram_counts.get(context, {})
        V = len(self.vocab)
        total = self.context_counts.get(context, 0) + V  # Laplace smoothing

        # 遍历整个词表self.vocab, 计算每个单词在上下文出现过的概率, 并进行排序
        return sorted(
            ((w, (counts.get(w, 0) + 1) / total) for w in self.vocab),
            key=lambda x: x[1],
            reverse=True,
        )

    def generate(self, max_len=20):
        context = ["<s>"] * (self.n - 1)
        result = []
        for _ in range(max_len):
            probs = self.next_word_probs(context)
            words, p = zip(*probs)
            word = random.choices(words, weights=p)[0]
            if word == "</s>":
                break
            result.append(word)
            context.append(word)
        return " ".join(result)

    def perplexity(self, sentence):
        tokens = ["<s>"] * (self.n - 1) + sentence + ["</s>"]
        V = len(self.vocab)
        log_prob = 0
        count = 0
        for i in range(len(tokens) - self.n + 1):
            context = tuple(tokens[i : i + self.n - 1])
            word = tokens[i + self.n - 1]
            count_word = self.ngram_counts.get(context, {}).get(word, 0) + 1
            total = self.context_counts.get(context, 0) + V
            prob = count_word / total
            log_prob += math.log(prob)
            count += 1
        return math.exp(-log_prob / count)


def load_text_as_sentences(file_path):
    """
    读取文本文件，并将其拆分成句子，最后将每个句子拆分成单词。

    Example:
    >>> sentences = load_text_as_sentences("data/test.txt")
    >>> sentences[0][:5]
    ['chapter', '1', 'phineas', 'finn', 'proposes']

    Args:
        file_path (str): 文件路径

    Returns:
        list of list: 每个句子是一个 list, 包含单词
    """

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().lower()
    sentences = re.split(r"[。！？.!?]", text)
    tokenized = [re.findall(r"\w+", s) for s in sentences if s.strip()]
    return tokenized


def main():
    file_path = "data/test.txt"
    sentences = load_text_as_sentences(file_path)

    bigram = NGramModel(n=3)
    bigram.train(sentences)

    # 预测示例
    print("预测：", bigram.next_word_probs(("i love",))[:5])

    # 生成文本
    print("生成：", bigram.generate(30))

    # 测试困惑度
    print("困惑度：", bigram.perplexity(["this", "is", "a", "test"]))


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    main()
