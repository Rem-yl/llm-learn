import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import wandb

# 自定义单步 LSTM Cell


class CustomLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # LSTM 四个门的权重和偏置 (输入门, 遗忘门, 输出门, 候选记忆)
        self.W_ih = nn.Linear(input_size, 4 * hidden_size)
        self.W_hh = nn.Linear(hidden_size, 4 * hidden_size)

    def forward(self, x, hx):
        # hx = (h, c)
        h, c = hx
        gates = self.W_ih(x) + self.W_hh(h)  # shape: [batch, 4*hidden_size]

        i_gate, f_gate, o_gate, g_gate = gates.chunk(4, dim=1)

        i = torch.sigmoid(i_gate)
        f = torch.sigmoid(f_gate)
        o = torch.sigmoid(o_gate)
        g = torch.tanh(g_gate)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


# 多步多层 LSTM
class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        layers = []
        for layer in range(num_layers):
            in_size = input_size if layer == 0 else hidden_size
            layers.append(CustomLSTMCell(in_size, hidden_size))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, hx=None):
        # x: [seq_len, batch_size, input_size]
        seq_len, batch_size, _ = x.size()
        if hx is None:
            h0 = [
                x.new_zeros(batch_size, self.hidden_size)
                for _ in range(self.num_layers)
            ]
            c0 = [
                x.new_zeros(batch_size, self.hidden_size)
                for _ in range(self.num_layers)
            ]
        else:
            h0, c0 = hx

        outputs = []
        h_n, c_n = [], []

        # 初始化当前隐状态
        hs = list(h0)
        cs = list(c0)

        for t in range(seq_len):
            inp = x[t]
            for layer in range(self.num_layers):
                h, c = self.layers[layer](inp, (hs[layer], cs[layer]))
                hs[layer], cs[layer] = h, c
                inp = h  # 下一层输入是当前层隐藏状态
            outputs.append(h)
        output = torch.stack(outputs, dim=0)  # [seq_len, batch, hidden_size]
        return output, (torch.stack(hs), torch.stack(cs))


# 简单字符级数据集
class CharDataset(Dataset):
    def __init__(self, text, seq_len):
        chars = sorted(list(set(text)))
        self.char2idx = {c: i for i, c in enumerate(chars)}
        self.idx2char = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)
        self.text = text
        self.seq_len = seq_len

    def __len__(self):
        return len(self.text) - self.seq_len

    def __getitem__(self, idx):
        chunk = self.text[idx : idx + self.seq_len + 1]
        input_seq = torch.tensor(
            [self.char2idx[c] for c in chunk[:-1]], dtype=torch.long
        )
        target_seq = torch.tensor(
            [self.char2idx[c] for c in chunk[1:]], dtype=torch.long
        )
        return input_seq, target_seq


# 整体模型：Embedding + CustomLSTM + Linear
class CharRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = CustomLSTM(embedding_dim, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hx=None):
        # x: [seq_len, batch_size]
        emb = self.embedding(x)  # [seq_len, batch, embed_dim]
        output, hx = self.lstm(emb, hx)
        logits = self.fc(output)  # [seq_len, batch, vocab_size]
        return logits, hx


def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    # 简单预处理：转小写，去除多余空白和换行
    text = text.lower().replace("\n", " ").strip()
    return text


def train():
    file_path = "data/test.txt"
    text = load_text(file_path)

    seq_len = 30
    dataset = CharDataset(text, seq_len)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CharRNN(
        dataset.vocab_size, embedding_dim=32, hidden_size=64, num_layers=2
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    criterion = nn.CrossEntropyLoss()
    wandb.init(project="char-rnn", name="char-rnn")

    model.train()
    for epoch in range(10):
        total_loss = 0.0
        for inputs, targets in dataloader:
            inputs = inputs.transpose(0, 1).to(device)  # 转为 [seq_len, batch]
            targets = targets.transpose(0, 1).to(device)

            optimizer.zero_grad()
            logits, _ = model(inputs)
            loss = criterion(logits.view(-1, dataset.vocab_size), targets.reshape(-1))
            loss.backward()
            optimizer.step()

            wandb.log({"batch_loss": loss.item()})
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")
        torch.save(model.state_dict(), f"checkpoints/char-rnn-epoch-{epoch + 1}.pth")
        wandb.log({"epoch_loss": total_loss / len(dataloader)})

    wandb.finish()


if __name__ == "__main__":
    train()
