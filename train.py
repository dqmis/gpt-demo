import torch

from src.models.mini_gpt import MiniGPT
from src.utils.data import get_batch
from src.utils.evaluation import estimate_loss

from matplotlib import pyplot as plt


# hyperparameters
batch_size = 128  # how many independent sequences will we process in parallel?
block_size = 128  # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 288
n_layer = 6
n_head = 6
dropout = 0.2
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open("./data/tinyshakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [
    stoi[c] for c in s
]  # encoder: take a string, output a list of integers
decode = lambda l: "".join(
    [itos[i] for i in l]
)  # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


model = MiniGPT(vocab_size, block_size, n_embd, n_head, n_layer)
m = model.to(device)

train_losses = []
val_losses = []

print(sum(p.numel() for p in m.parameters()), "parameters")

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(
            m, eval_iters, train_data, val_data, block_size, device, batch_size
        )
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )
        train_losses.append(losses["train"])
        val_losses.append(losses["val"])

    # sample a batch of data
    xb, yb = get_batch("train", train_data, val_data, block_size, device, batch_size)

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=100)[0].tolist()))

plt.figure()
plt.plot(train_losses, label="train")
plt.plot(val_losses, label="val")
plt.legend()
plt.savefig("out.png")
