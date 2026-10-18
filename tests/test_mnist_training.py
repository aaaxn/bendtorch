"""A smoke run of examples/mnist/train.bend.

It writes a tiny MNIST in IDX format (6 training and 12 test images), initial
parameters and an order, builds the program natively, trains two steps of
batch 3 and checks every printed word against the tensor backward reference
in tools/autograd_tensor_backward.py, which runs the same tape operations:
the loss of each step, the test count and every trained parameter word. The
test labels are the reference's own predictions, every third one moved to
another class, so the count is 8 of 12 only when every row's argmax is right.
The output must be the same at one thread and at several. A data file
that is not MNIST's, or that is shorter or longer than its header says, and
a short parameter file must stop the program with exit code 4 before it
prints anything.
"""

import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.autograd_broadcast import word
from tools.autograd_tensor_backward import (
    div_word,
    entropy,
    leaf,
    linear,
    op,
    reference,
    run,
    sgd,
)
from tools.common import ROOT, THREADS, bend, environment, native

TOOLS = ROOT / ".tools"
SOURCE = ROOT / "examples/mnist/train.bend"
SHAPES = [[128, 784], [128], [10, 128], [10]]
TRAIN = 6
TEST = 12
BATCH = 3
STEPS = 2
LR = "0.1"


def pixel(image: int, index: int) -> int:
    return (image * 7919 + index * 104729) % 256 if index % 3 == 0 else 0


def idx(path: Path, magic: int, dimensions: list[int], data: bytes) -> None:
    path.write_bytes(struct.pack(f">{1 + len(dimensions)}I", magic, *dimensions) + data)


def images(count: int, first: int) -> bytes:
    return bytes(
        pixel(first + image, index) for image in range(count) for index in range(784)
    )


def labels(count: int) -> list[int]:
    return [image * 7 % 10 for image in range(count)]


def initial() -> list[list[int]]:
    words = []
    start = 0
    for shape in SHAPES:
        count = 1
        for dimension in shape:
            count *= dimension
        words.append(
            [
                word(((k * 7919 + 13) % 2001 - 1000) / 28000)
                for k in range(start, start + count)
            ]
        )
        start += count
    return words


def pixels(data: bytes) -> list[int]:
    return [div_word(word(float(byte)), word(255.0)) for byte in data]


def values(lines: list[str]) -> list[list[int]]:
    found = []
    for at, line in enumerate(lines):
        if line == "autograd:value":
            count = int(lines[at + 2].removeprefix("tensor:count:"))
            found.append(
                [
                    int(text.removeprefix("tensor:word:"))
                    for text in lines[at + 3 : at + 3 + count]
                ]
            )
    return found


def network(params: list[list[int]], x: list[int], rows: int) -> list[dict]:
    return [
        *[leaf(shape, words) for shape, words in zip(SHAPES, params, strict=True)],
        leaf([rows, 784], x),
        linear(4, 0, 1),
        op("relu", 6),
        linear(7, 2, 3),
    ]


def value_of(number: int) -> float:
    return struct.unpack("<f", struct.pack("<I", number))[0]


# The reference's trained parameter words, the loss lines and the test
# labels: one tape per step, as the program builds.
def trained(order: list[int]) -> tuple[list[list[int]], list[str], list[int]]:
    params = initial()
    train = images(TRAIN, 0)
    targets = labels(TRAIN)
    lines = []
    for k in range(STEPS):
        rows = order[k * BATCH : (k + 1) * BATCH]
        x = pixels(b"".join(train[784 * row : 784 * (row + 1)] for row in rows))
        script = [
            *network(params, x, BATCH),
            entropy(9, [targets[row] for row in rows]),
            run(10),
            sgd(word(float(LR)), [0, 1, 2, 3]),
            *[{"op": "read", "id": identifier} for identifier in (10, 0, 1, 2, 3)],
        ]
        loss, *params = values(reference({"id": "mnist", "steps": script}))
        lines.append(f"loss {k} {loss[0]}")
    script = [
        *network(params, pixels(images(TEST, TRAIN)), TEST),
        {"op": "read", "id": 9},
    ]
    (logits,) = values(reference({"id": "mnist-test", "steps": script}))
    truth = []
    for row in range(TEST):
        scores = [value_of(number) for number in logits[10 * row : 10 * (row + 1)]]
        guess = scores.index(max(scores))
        truth.append(guess if row % 3 else (guess + 1) % 10)
    return params, lines, truth


class MnistTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="mnist-training-")
        root = Path(cls.directory.name)
        cls.data = root / "mnist"
        cls.data.mkdir()
        idx(
            cls.data / "train-images-idx3-ubyte",
            2051,
            [TRAIN, 28, 28],
            images(TRAIN, 0),
        )
        idx(cls.data / "train-labels-idx1-ubyte", 2049, [TRAIN], bytes(labels(TRAIN)))
        cls.order_list = [3, 0, 5, 1, 4, 2]
        params, lines, cls.truth = trained(cls.order_list)
        cls.expected = [
            *lines,
            f"correct {TEST - TEST // 3} {TEST}",
            *[str(number) for words in params for number in words],
        ]
        idx(
            cls.data / "t10k-images-idx3-ubyte",
            2051,
            [TEST, 28, 28],
            images(TEST, TRAIN),
        )
        idx(cls.data / "t10k-labels-idx1-ubyte", 2049, [TEST], bytes(cls.truth))
        cls.init = root / "init.bin"
        flat = [number for words in initial() for number in words]
        cls.init.write_bytes(struct.pack(f"<{len(flat)}I", *flat))
        cls.order = root / "order.bin"
        cls.order.write_bytes(struct.pack(f"<{len(cls.order_list)}I", *cls.order_list))
        cls.binary = root / "train"
        subprocess.run(
            bend(TOOLS, SOURCE, "-o", root / "train.c"),
            env=environment(TOOLS),
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [
                str(TOOLS / "llvm/usr/bin/clang-19"),
                "-std=c11",
                "-O3",
                str(root / "train.c"),
                "-lpthread",
                "-lm",
            ]
            + ["-o", str(cls.binary)],
            env=environment(TOOLS),
            check=True,
            capture_output=True,
        )

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def train(self, data: Path, threads: int) -> subprocess.CompletedProcess[str]:
        arguments = [
            str(data),
            str(self.init),
            str(self.order),
            str(BATCH),
            str(STEPS),
            LR,
            "words",
        ]
        return subprocess.run(
            native(self.binary, *arguments, threads=threads),
            capture_output=True,
            text=True,
        )

    def test_two_steps_match_the_reference_word_for_word_at_any_thread_count(self):
        one = self.train(self.data, 1)
        many = self.train(self.data, max(2, THREADS))
        self.assertEqual(one.returncode, 0, one.stderr)
        self.assertEqual(one.stdout, many.stdout)
        self.assertEqual(one.stdout.splitlines(), self.expected)

    def test_a_file_that_is_not_mnist_is_refused(self):
        with tempfile.TemporaryDirectory(prefix="mnist-wrong-") as directory:
            wrong = Path(directory)
            for path in self.data.iterdir():
                (wrong / path.name).write_bytes(path.read_bytes())
            idx(
                wrong / "t10k-labels-idx1-ubyte",
                2049,
                [TEST + 1],
                bytes([*self.truth, 0]),
            )
            refused = self.train(wrong, 1)
        self.assertEqual(refused.returncode, 4)
        self.assertEqual(refused.stdout, "")

    def test_a_file_shorter_or_longer_than_its_header_is_refused(self):
        # The header keeps its count; the payload loses or gains bytes.
        for name, size in (
            ("t10k-labels-idx1-ubyte", 8),
            ("t10k-labels-idx1-ubyte", 8 + TEST - 1),
            ("t10k-images-idx3-ubyte", 16 + 784 * TEST - 1),
            ("train-images-idx3-ubyte", 16 + 784 * (TRAIN - 1)),
            ("train-labels-idx1-ubyte", 8 + TRAIN + 1),
        ):
            with (
                self.subTest(name=name, size=size),
                tempfile.TemporaryDirectory(prefix="mnist-short-") as directory,
            ):
                cut = Path(directory)
                for path in self.data.iterdir():
                    (cut / path.name).write_bytes(path.read_bytes())
                data = (cut / name).read_bytes()
                (cut / name).write_bytes((data + bytes(1))[:size])
                refused = self.train(cut, 1)
                self.assertEqual(refused.returncode, 4)
                self.assertEqual(refused.stdout, "")

    def test_a_short_parameter_file_is_refused(self):
        with tempfile.TemporaryDirectory(prefix="mnist-short-") as directory:
            short = Path(directory) / "init.bin"
            short.write_bytes(self.init.read_bytes()[:-4])
            arguments = [str(self.data), str(short), str(self.order)]
            arguments += [str(BATCH), str(STEPS), LR]
            refused = subprocess.run(
                native(self.binary, *arguments, threads=1),
                capture_output=True,
                text=True,
            )
        self.assertEqual(refused.returncode, 4)
        self.assertEqual(refused.stdout, "")


if __name__ == "__main__":
    unittest.main()
