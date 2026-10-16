"""PyTorch's copy of experiments/mnist/train.bend.

prepare writes the files both sides share: the initial parameters of a
784-128-10 MLP from PyTorch's default nn.Linear init under a seed, as
little-endian F32 words (weight1 [128, 784], bias1 [128], weight2 [10, 128],
bias2 [10]), and the training order, a permutation of the 60000 training
examples per epoch drawn from a generator with the same seed, its first
`steps per epoch * batch` indices kept (DataLoader's drop_last), as
little-endian U32 words.

train reads those files and the IDX files and trains as the Bend program does:
step k takes the batch indices from word k * batch on, zero_grad, forward,
cross_entropy with mean reduction, backward, SGD step. It prints the same
lines: `loss <step> <word>`, `correct <n> 10000`, then with --words every word
of the trained parameters.

Run from oracle/:
  uv run --frozen python -I mnist/train.py prepare SEED OUT_DIR [--epochs N] [--batch B]
  uv run --frozen python -I mnist/train.py train MNIST_DIR INIT ORDER BATCH STEPS LR [--words]
"""

import argparse
import struct
import sys
import time
from pathlib import Path

import torch

TRAIN = 60000
FEATURES = 784
HIDDEN = 128
CLASSES = 10


def words_of(tensors):
    flat = [v for t in tensors for v in t.detach().reshape(-1).tolist()]
    return struct.pack(f"<{len(flat)}f", *flat)


def model_of(seed):
    torch.manual_seed(seed)
    return torch.nn.Linear(FEATURES, HIDDEN), torch.nn.Linear(HIDDEN, CLASSES)


def prepare(seed, out, epochs, batch):
    out.mkdir(parents=True, exist_ok=True)
    first, second = model_of(seed)
    (out / f"init-{seed}.bin").write_bytes(
        words_of([first.weight, first.bias, second.weight, second.bias])
    )
    generator = torch.Generator().manual_seed(seed)
    kept = (TRAIN // batch) * batch
    order = []
    for _ in range(epochs):
        order += torch.randperm(TRAIN, generator=generator)[:kept].tolist()
    (out / f"order-{seed}-b{batch}.bin").write_bytes(
        struct.pack(f"<{len(order)}I", *order)
    )


def idx(path, offset, count, width):
    data = Path(path).read_bytes()[offset:]
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).reshape(count, width)


def bits(x):
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def tensor_of(words, start, shape):
    n = 1
    for d in shape:
        n *= d
    values = struct.unpack_from(f"<{n}f", words, 4 * start)
    return torch.tensor(values, dtype=torch.float32).reshape(shape), start + n


def train(mnist, init, order, batch, steps, lr, show_words):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    images = idx(Path(mnist) / "train-images-idx3-ubyte", 16, TRAIN, FEATURES)
    labels = (
        idx(Path(mnist) / "train-labels-idx1-ubyte", 8, TRAIN, 1).reshape(-1).long()
    )
    words = Path(init).read_bytes()
    w1, at = tensor_of(words, 0, [HIDDEN, FEATURES])
    b1, at = tensor_of(words, at, [HIDDEN])
    w2, at = tensor_of(words, at, [CLASSES, HIDDEN])
    b2, at = tensor_of(words, at, [CLASSES])
    params = [p.requires_grad_() for p in (w1, b1, w2, b2)]
    indices = Path(order).read_bytes()
    optimizer = torch.optim.SGD(params, lr=lr)
    start = time.perf_counter()
    for k in range(steps):
        rows = torch.tensor(
            struct.unpack_from(f"<{batch}I", indices, 4 * k * batch), dtype=torch.long
        )
        x = images[rows].float() / 255
        optimizer.zero_grad()
        hidden = torch.nn.functional.relu(torch.nn.functional.linear(x, w1, b1))
        loss = torch.nn.functional.cross_entropy(
            torch.nn.functional.linear(hidden, w2, b2), labels[rows]
        )
        loss.backward()
        optimizer.step()
        print(f"loss {k} {bits(loss.item())}")
    print(f"train_ms {round(1000 * (time.perf_counter() - start))}", file=sys.stderr)
    test = idx(Path(mnist) / "t10k-images-idx3-ubyte", 16, 10000, FEATURES)
    truth = idx(Path(mnist) / "t10k-labels-idx1-ubyte", 8, 10000, 1).reshape(-1).long()
    correct = 0
    with torch.no_grad():
        for b in range(10):
            x = test[1000 * b : 1000 * (b + 1)].float() / 255
            hidden = torch.nn.functional.relu(torch.nn.functional.linear(x, w1, b1))
            logits = torch.nn.functional.linear(hidden, w2, b2)
            correct += int((logits.argmax(1) == truth[1000 * b : 1000 * (b + 1)]).sum())
    print(f"correct {correct} 10000")
    if show_words:
        for p in params:
            for v in p.detach().reshape(-1).tolist():
                print(bits(v))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("seed", type=int)
    p.add_argument("out", type=Path)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch", type=int, default=64)
    t = sub.add_parser("train")
    t.add_argument("mnist")
    t.add_argument("init")
    t.add_argument("order")
    t.add_argument("batch", type=int)
    t.add_argument("steps", type=int)
    t.add_argument("lr", type=float)
    t.add_argument("--words", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.seed, args.out, args.epochs, args.batch)
    else:
        train(
            args.mnist,
            args.init,
            args.order,
            args.batch,
            args.steps,
            args.lr,
            args.words,
        )


if __name__ == "__main__":
    main()
