import Base
import ../../src/tensor/core.bend as Core
import ../../src/autograd/tensor_backward.bend as Backward
import ../../src/autograd/sgd.bend as Sgd

# Trains a 784-128-10 MLP on MNIST with the tape: Linear, relu, Linear,
# cross_entropy with mean reduction, backward, Sgd.step, one mini-batch per
# step. PyTorch's copy is oracle/mnist/train.py; docs/mnist.md says how to
# run both.
#
#   train <mnist dir> <init file> <order file> <batch> <steps> <lr> [words]
#
# <mnist dir> holds the four uncompressed IDX files. <init file> holds the
# initial parameters as little-endian F32 words: weight1 [128, 784], bias1
# [128], weight2 [10, 128], bias2 [10], each row-major. <order file> holds
# training example indices as little-endian U32 words; step k takes the
# `batch` indices from word k * batch on. A pixel is its byte divided by
# 255. Every step builds a new tape from the current parameters, so every
# gradient slot starts empty, as PyTorch's optimizer.zero_grad() leaves it.
#
# Prints `loss <step> <word>` for every step, the loss's F32 bits as a
# decimal U32, then `correct <n> <of>` for the test set, then, with
# `words`, every word of the trained parameters, one per line. The wall
# times of the training and of the test pass go to stderr, as `train_ms` and
# `test_ms`.

type Params is Type:
  Params{w1: Core.Dynamic, b1: Core.Dynamic, w2: Core.Dynamic, b2: Core.Dynamic}

# What one training step hands back: the updated parameters and the loss
# before the update, or stage * 1000 + the error code of the first failure.
# The error code of a test batch, 0 when it ran, and its correct rows.
type Score is Data:
  Score{code: U32, correct: Nat}

type Trained is Type:
  Trained{params: Params, loss: F32}
  Broken{code: U32}

# Bytes to words.

def le_u32(a: U32, b: U32, c: U32, d: U32) -> U32:
  (a + (b * 256) + (c * 65536) + (d * 16777216) : U32)

def from_bits(word: U32) -> F32:
  U32{bits} = word
  F32{bits}

def le_f32(a: U32, b: U32, c: U32, d: U32) -> F32:
  from_bits(le_u32(a, b, c, d))

def floats(bytes: List<&2, U32>) -> List<&2, F32>:
  match bytes:
    case Con{a, Con{b, Con{c, Con{d, rest}}}}:
      le_f32(a, b, c, d) <> floats(rest)
    case _:
      Nil{}

def u32s(bytes: List<&2, U32>) -> List<&2, U32>:
  match bytes:
    case Con{a, Con{b, Con{c, Con{d, rest}}}}:
      le_u32(a, b, c, d) <> u32s(rest)
    case _:
      Nil{}

# A pixel byte to its value: the byte divided by 255, one F32 rounding.
def pixels(bytes: List<&2, U32>) -> List<&2, F32>:
  match bytes:
    case Nil{}:
      Nil{}
    case Con{b, rest}:
      F32.div(U32.to_f32(b), 255.0) <> pixels(rest)

# Files.

def got(m: File & Result<&1, &1, U32 & String, List<&2, U32>>) -> IO(File & List<&2, U32>):
  (f, r) = m
  do IO<File & List<&2, U32>>:
    bytes : List<&2, U32> <- IO.pass(List<&2, U32>, r)
    IO.pure(File & List<&2, U32>, (f, bytes))

def read_at(f: File, offset: U32, size: U32) -> IO(File & List<&2, U32>):
  do IO<File & List<&2, U32>>:
    m : File & Result<&1, &1, U32 & String, List<&2, U32>> <- File.read_at(f, offset, size)
    got(m)

def closed(pair: File & List<&2, U32>) -> IO(List<&2, U32>):
  (f, bytes) = pair
  do IO<List<&2, U32>>:
    File.close(f)
    return bytes

def exact_if(ok: Bool, +path: String, bytes: List<&2, U32>) -> IO(List<&2, U32>):
  match ok:
    case True{}:
      IO.pure(List<&2, U32>, bytes)
    case False{}:
      IO.die(List<&2, U32>, 4, "short read: " ++ path)

# The bytes, if there are `size` of them; else the program stops with exit
# code 4, since File.read_at hands back the bytes the file has.
def exact(+path: String, +size: U32, +bytes: List<&2, U32>) -> IO(List<&2, U32>):
  exact_if(Nat.is_eq(List.length(&2, U32, bytes), U32.to_nat(size)), path, bytes)

# Exactly `size` bytes at offset in the file at path.
def bytes_at(+path: String, offset: U32, +size: U32) -> IO(List<&2, U32>):
  do IO<List<&2, U32>>:
    f : File <- IO.try(File, File.open(path, "r"))
    pair : File & List<&2, U32> <- read_at(f, offset, size)
    bytes : List<&2, U32> <- closed(pair)
    exact(path, size, bytes)

def sized(m: File & Result<&1, &1, U32 & String, U32>) -> IO(U32):
  (f, r) = m
  do IO<U32>:
    n : U32 <- IO.pass(U32, r)
    File.close(f)
    return n

# The number of bytes in the file at path.
def file_size(+path: String) -> IO(U32):
  do IO<U32>:
    f : File <- IO.try(File, File.open(path, "r"))
    m : File & Result<&1, &1, U32 & String, U32> <- File.size(f)
    sized(m)

# The file and the records read so far, the last one first.
def kept_row(rows: List<&2, List<&2, U32>>, pair: File & List<&2, U32>) -> File & List<&2, List<&2, U32>>:
  (f, row) = pair
  (f, Con{row, rows})

def exact_row(+path: String, +size: U32, pair: File & List<&2, U32>) -> IO(File & List<&2, U32>):
  (f, +bytes) = pair
  do IO<File & List<&2, U32>>:
    row : List<&2, U32> <- exact(path, size, bytes)
    return (f, row)

# Reads the record of `size` bytes at base + index * size for each index on
# one open file, each checked to be whole.
def rows_on(indices: List<&2, U32>, +path: String, +base: U32, +size: U32, pair: File & List<&2, List<&2, U32>>) ->
  IO(File & List<&2, List<&2, U32>>):
  match indices:
    case Nil{}:
      IO.pure(File & List<&2, List<&2, U32>>, pair)
    case Con{+index, rest}:
      (f, rows) = pair
      do IO<File & List<&2, List<&2, U32>>>:
        read : File & List<&2, U32> <- read_at(f, (base + (index * size) : U32), size)
        whole : File & List<&2, U32> <- exact_row(path, size, read)
        rows_on(rest, path, base, size, kept_row(rows, whole))

# The rows, last first, joined in order.
def joined(rows: List<&2, List<&2, U32>>, tail: List<&2, U32>) -> List<&2, U32>:
  match rows:
    case Nil{}:
      tail
    case Con{row, earlier}:
      joined(earlier, List.append(&2, U32, row, tail))

def rows_closed(pair: File & List<&2, List<&2, U32>>) -> IO(List<&2, U32>):
  (f, rows) = pair
  do IO<List<&2, U32>>:
    File.close(f)
    return joined(rows, Nil{})

# The records of `size` bytes at base + index * size for each index, in order,
# read on one open file.
def rows_at(+path: String, +base: U32, +size: U32, indices: List<&2, U32>) -> IO(List<&2, U32>):
  do IO<List<&2, U32>>:
    f : File <- IO.try(File, File.open(path, "r"))
    pair : File & List<&2, List<&2, U32>> <- rows_on(indices, path, base, size, (f, Nil{}))
    rows_closed(pair)

# Tensors.

def built(result: Result<&2, &1, Core.Error, Core.Dynamic>) -> IO(Core.Dynamic):
  match result:
    case Fail{error}:
      IO.die(Core.Dynamic, 2, "tensor:error:" ++ U32.show(Core.error_code(error)))
    case Done{tensor}:
      IO.pure(Core.Dynamic, tensor)

def tensor(+dimensions: List<&2, U32>, values: List<&2, F32>) -> IO(Core.Dynamic):
  built(Core.construct(dimensions, values))

def params_of(+words: List<&2, F32>) -> IO(Params):
  do IO<Params>:
    w1 : Core.Dynamic <- tensor([128, 784], List.take(&2, F32, words, 100352n))
    b1 : Core.Dynamic <- tensor([128], List.take(&2, F32, List.drop(&2, F32, words, 100352n), 128n))
    w2 : Core.Dynamic <- tensor([10, 128], List.take(&2, F32, List.drop(&2, F32, words, 100480n), 1280n))
    b2 : Core.Dynamic <- tensor([10], List.drop(&2, F32, words, 101760n))
    return Params{w1, b1, w2, b2}

# One training step.

# 0 for a result that is done, else stage * 1000 + its error code.
def status(-A: Data, +stage: U32, result: Result<&2, &2, Backward.Error, A>) -> U32:
  match result:
    case Fail{error}:
      ((stage * 1000) + Backward.error_code(error) : U32)
    case Done{value}:
      0

def descent_status(result: Result<&2, &2, Sgd.Error, Unit>) -> U32:
  match result:
    case Fail{error}:
      (6000 + Sgd.error_code(error) : U32)
    case Done{value}:
      0

def first(+a: U32, +b: U32) -> U32:
  Bool.pick(U32, U32.is_eq(a, 0), b, a)

def leaf_tape(pair: Backward.Tape & Nat) -> Backward.Tape:
  (tape, id) = pair
  tape

# The tape of the parameters, ids 0 to 3, and the batch, id 4. The batch is a
# constant, as PyTorch's input tensor has requires_grad=False, so backward
# never computes its gradient.
def leaves(params: Params, x: Core.Dynamic) -> Backward.Tape:
  Params{w1, b1, w2, b2} = params
  leaf_tape(Backward.constant(leaf_tape(Backward.leaf(leaf_tape(Backward.leaf(leaf_tape(Backward.leaf(
    leaf_tape(Backward.leaf(Backward.empty(), w1)), b1)), w2)), b2)), x))

# A tape, the code of the first failure so far (0 for none) and the id the
# last recording gave (0 if it failed).
type Noted is Type:
  Noted{tape: Backward.Tape, code: U32, id: Nat}

def noted_id(result: Result<&2, &2, Backward.Error, Nat>) -> Nat:
  match result:
    case Fail{error}:
      0n
    case Done{id}:
      id

def noted(+stage: U32, +code: U32, pair: Backward.Tape & Result<&2, &2, Backward.Error, Nat>) -> Noted:
  (tape, +result) = pair
  Noted{tape, first(code, status(Nat, stage, result)), noted_id(result)}

def head(noted_hidden: Noted) -> Noted:
  Noted{tape, +code, +id} = noted_hidden
  noted(3, code, Backward.linear(tape, id, 2n, 3n))

def hidden(noted_first: Noted) -> Noted:
  Noted{tape, +code, +id} = noted_first
  head(noted(2, code, Backward.relu(tape, id)))

# logits = linear(relu(linear(x, w1, b1)), w2, b2) and the id of the logits.
def forward(tape: Backward.Tape) -> Noted:
  hidden(noted(1, 0, Backward.linear(tape, 4n, 0n, 1n)))

def scalar(pair: Core.Dynamic & Result<&2, &2, Core.Error, F32>) -> F32:
  (tensor, result) = pair
  match result:
    case Fail{error}:
      F32.div(0.0, 0.0)
    case Done{value}:
      value

def assembled(r8: Result<&2, &1, Backward.Error, Core.Dynamic>, r0: Result<&2, &1, Backward.Error, Core.Dynamic>,
  r1: Result<&2, &1, Backward.Error, Core.Dynamic>, r2: Result<&2, &1, Backward.Error, Core.Dynamic>,
  r3: Result<&2, &1, Backward.Error, Core.Dynamic>) -> Trained:
  match r8 r0 r1 r2 r3:
    case Done{loss} Done{w1} Done{b1} Done{w2} Done{b2}:
      Trained{Params{w1, b1, w2, b2}, scalar(Core.read_dynamic(loss, 0))}
    case r8 r0 r1 r2 r3:
      Broken{7000}

def Read() -> Type:
  Backward.Tape & Result<&2, &1, Backward.Error, Core.Dynamic>

def read_b2(r8: Result<&2, &1, Backward.Error, Core.Dynamic>, r0: Result<&2, &1, Backward.Error, Core.Dynamic>,
  r1: Result<&2, &1, Backward.Error, Core.Dynamic>, r2: Result<&2, &1, Backward.Error, Core.Dynamic>, pair: Read()) -> Trained:
  (tape, r3) = pair
  assembled(r8, r0, r1, r2, r3)

def read_w2(r8: Result<&2, &1, Backward.Error, Core.Dynamic>, r0: Result<&2, &1, Backward.Error, Core.Dynamic>,
  r1: Result<&2, &1, Backward.Error, Core.Dynamic>, pair: Read()) -> Trained:
  (tape, r2) = pair
  read_b2(r8, r0, r1, r2, Backward.value(tape, 3n))

def read_b1(r8: Result<&2, &1, Backward.Error, Core.Dynamic>, r0: Result<&2, &1, Backward.Error, Core.Dynamic>,
  pair: Read()) -> Trained:
  (tape, r1) = pair
  read_w2(r8, r0, r1, Backward.value(tape, 2n))

def read_w1(r8: Result<&2, &1, Backward.Error, Core.Dynamic>, pair: Read()) -> Trained:
  (tape, r0) = pair
  read_b1(r8, r0, Backward.value(tape, 1n))

def read_loss(pair: Read()) -> Trained:
  (tape, r8) = pair
  read_w1(r8, Backward.value(tape, 0n))

def checked_if(ok: Bool, +code: U32, trained: Trained) -> Trained:
  match ok:
    case True{}:
      trained
    case False{}:
      Broken{code}

def checked(+code: U32, trained: Trained) -> Trained:
  checked_if(U32.is_eq(code, 0), code, trained)

# The loss and the parameters after the step.
def descended(+loss: Nat, +code: U32, pair: Backward.Tape & Result<&2, &2, Sgd.Error, Unit>) -> Trained:
  (tape, result) = pair
  checked(first(code, descent_status(result)), read_loss(Backward.value(tape, loss)))

def ran(+lr: F32, +loss: Nat, +code: U32, pair: Backward.Tape & Result<&2, &2, Backward.Error, Unit>) -> Trained:
  (tape, result) = pair
  descended(loss, first(code, status(Unit, 5, result)), Sgd.step(tape, lr, [0n, 1n, 2n, 3n]))

def lossed(+lr: F32, noted_loss: Noted) -> Trained:
  Noted{tape, +code, +id} = noted_loss
  ran(lr, id, code, Backward.backward(tape, id))

def scored_logits(+targets: List<&2, U32>, +lr: F32, noted_logits: Noted) -> Trained:
  Noted{tape, +code, +id} = noted_logits
  lossed(lr, noted(4, code, Backward.cross_entropy(tape, id, targets)))

# Forward, cross_entropy, backward of the loss, then Sgd.step.
def step(params: Params, x: Core.Dynamic, +targets: List<&2, U32>, +lr: F32) -> Trained:
  scored_logits(targets, lr, forward(leaves(params, x)))

# Evaluation.

def paired(w1: Core.Dynamic & Core.Dynamic, b1: Core.Dynamic & Core.Dynamic, w2: Core.Dynamic & Core.Dynamic,
  b2: Core.Dynamic & Core.Dynamic) -> Params & Params:
  (w1a, w1b) = w1
  (b1a, b1b) = b1
  (w2a, w2b) = w2
  (b2a, b2b) = b2
  (Params{w1a, b1a, w2a, b2a}, Params{w1b, b1b, w2b, b2b})

def cloned(params: Params) -> Params & Params:
  Params{w1, b1, w2, b2} = params
  paired(Core.clone_dynamic(w1), Core.clone_dynamic(b1), Core.clone_dynamic(w2), Core.clone_dynamic(b2))

def word(result: Result<&2, &2, Core.Error, F32>) -> F32:
  match result:
    case Fail{error}:
      F32.div(0.0, 0.0)
    case Done{value}:
      value

def minus_infinity() -> F32:
  F32.neg(F32.div(1.0, 0.0))

# How many rows of the row-major [rows, 10] logits have their argmax at the
# row's label, the first column on a tie, as PyTorch's argmax. pair is the
# read of word `index`, at `column` of its row; `best` and `top` are the
# argmax and the largest word of the row's earlier columns; `last` says the
# column is 9.
def hits(left: Nat, labels: List<&2, U32>, +column: U32, +best: U32, +top: F32, +total: Nat, +index: U32, last: Bool,
  pair: Core.Dynamic & Result<&2, &2, Core.Error, F32>) -> Nat:
  match left labels last:
    case 1n+more Con{+label, rest} True{}:
      (t, result) = pair
      +argmax = Bool.pick(U32, F32.is_gt(word(result), top), column, best)
      hits(more, rest, 0, 0, minus_infinity(), Nat.add(total, Bool.pick(Nat, U32.is_eq(argmax, label), 1n, 0n)),
        (index + 1 : U32), False{}, Core.read_dynamic(t, (index + 1 : U32)))
    case 1n+more Con{+label, rest} False{}:
      (t, result) = pair
      +value = word(result)
      +greater = F32.is_gt(value, top)
      hits(more, Con{label, rest}, (column + 1 : U32), Bool.pick(U32, greater, column, best),
        Bool.pick(F32, greater, value, top), total, (index + 1 : U32), U32.is_eq(column, 8),
        Core.read_dynamic(t, (index + 1 : U32)))
    case left labels last:
      total

def logits_hits(+code: U32, +labels: List<&2, U32>, result: Result<&2, &1, Backward.Error, Core.Dynamic>) -> Score:
  match result:
    case Fail{error}:
      Score{(8000 + Backward.error_code(error) : U32), 0n}
    case Done{logits}:
      Score{code, hits(Nat.mul(10n, List.length(&2, U32, labels)), labels, 0, 0, minus_infinity(), 0n, 0, False{},
        Core.read_dynamic(logits, 0))}

def logits_read(+code: U32, +labels: List<&2, U32>, pair: Read()) -> Score:
  (tape, result) = pair
  logits_hits(code, labels, result)

def scored_tape(+labels: List<&2, U32>, noted_logits: Noted) -> Score:
  Noted{tape, +code, +id} = noted_logits
  logits_read(code, labels, Backward.value(tape, id))

def scored_pair(x: Core.Dynamic, +labels: List<&2, U32>, pair: Params & Params) -> Params & Score:
  (kept, used) = pair
  (kept, scored_tape(labels, forward(leaves(used, x))))

# The parameters, kept, and the error code and number of correct rows of one
# test batch.
def scored(params: Params, x: Core.Dynamic, +labels: List<&2, U32>) -> Params & Score:
  scored_pair(x, labels, cloned(params))

# The program.

def die_code(-A: Type, +code: U32) -> IO(A):
  IO.die(A, 3, "autograd:error:" ++ U32.show(code))

def show_words(left: Nat, +index: U32, pair: Core.Dynamic & Result<&2, &2, Core.Error, F32>) -> IO(Unit):
  match left:
    case 0n:
      IO.pure(Unit, Unit{})
    case 1n+rest:
      (t, result) = pair
      do IO<Unit>:
        IO.print(U32.show(F32.bits(word(result))))
        show_words(rest, (index + 1 : U32), Core.read_dynamic(t, (index + 1 : U32)))

def show_counted(pair: Core.Dynamic & U32) -> IO(Unit):
  (t, +count) = pair
  show_words(U32.to_nat(count), 0, Core.read_dynamic(t, 0))

def show_tensor(t: Core.Dynamic) -> IO(Unit):
  show_counted(Core.count(t))

def show_params(params: Params) -> IO(Unit):
  Params{w1, b1, w2, b2} = params
  do IO<Unit>:
    show_tensor(w1)
    show_tensor(b1)
    show_tensor(w2)
    show_tensor(b2)

# The batch of step k: the pixels [batch, 784] and the labels.
def loaded(+dir: String, +batch: U32, +indices: List<&2, U32>) -> IO(Core.Dynamic & List<&2, U32>):
  do IO<Core.Dynamic & List<&2, U32>>:
    px : List<&2, U32> <- rows_at(dir ++ "/train-images-idx3-ubyte", 16, 784, indices)
    labels : List<&2, U32> <- rows_at(dir ++ "/train-labels-idx1-ubyte", 8, 1, indices)
    x : Core.Dynamic <- tensor([batch, 784], pixels(px))
    IO.pure(Core.Dynamic & List<&2, U32>, (x, labels))

def batch_at(+dir: String, +order: String, +batch: U32, +k: U32) -> IO(Core.Dynamic & List<&2, U32>):
  do IO<Core.Dynamic & List<&2, U32>>:
    words : List<&2, U32> <- bytes_at(order, (k * batch * 4 : U32), (batch * 4 : U32))
    loaded(dir, batch, u32s(words))

def stepped(params: Params, pair: Core.Dynamic & List<&2, U32>, +lr: F32) -> Trained:
  (x, +targets) = pair
  step(params, x, targets, lr)

def reported(+k: U32, trained: Trained) -> IO(Params):
  match trained:
    case Broken{+code}:
      die_code(Params, code)
    case Trained{params, +loss}:
      do IO<Params>:
        IO.print("loss " ++ U32.show(k) ++ " " ++ U32.show(F32.bits(loss)))
        return params

def train(left: Nat, +dir: String, +order: String, +batch: U32, +lr: F32, +k: U32, params: Params) -> IO(Params):
  match left:
    case 0n:
      IO.pure(Params, params)
    case 1n+rest:
      do IO<Params>:
        pair : Core.Dynamic & List<&2, U32> <- batch_at(dir, order, batch, k)
        next : Params <- reported(k, stepped(params, pair, lr))
        train(rest, dir, order, batch, lr, (k + 1 : U32), next)

def guard(ok: Bool, +code: U32) -> IO(Unit):
  match ok:
    case True{}:
      IO.pure(Unit, Unit{})
    case False{}:
      die_code(Unit, code)

# The number of test rows the parameters classify right, 1000 rows a batch,
# the last batch the rest. pair is the previous batch's: the parameters, its
# error code and its count.
def evaluate(left: Nat, +dir: String, +b: U32, +rows: U32, +total: Nat, pair: Params & Score) -> IO(Params & Nat):
  match left:
    case 0n:
      (params, result) = pair
      Score{+code, +n} = result
      do IO<Params & Nat>:
        guard(U32.is_eq(code, 0), code)
        IO.pure(Params & Nat, (params, Nat.add(total, n)))
    case 1n+rest:
      (params, result) = pair
      Score{+code, +n} = result
      +size = Bool.pick(U32, U32.is_lt((rows - (b * 1000) : U32), 1000), (rows - (b * 1000) : U32), 1000)
      do IO<Params & Nat>:
        guard(U32.is_eq(code, 0), code)
        px : List<&2, U32> <- bytes_at(dir ++ "/t10k-images-idx3-ubyte", (16 + (b * 784000) : U32), (size * 784 : U32))
        labels : List<&2, U32> <- bytes_at(dir ++ "/t10k-labels-idx1-ubyte", (8 + (b * 1000) : U32), size)
        x : Core.Dynamic <- tensor([size, 784], pixels(px))
        evaluate(rest, dir, (b + 1 : U32), rows, Nat.add(total, n), scored(params, x, labels))

# IDX headers: big-endian U32 words, the magic number first, then the count,
# then for images the rows and the columns.

def be_u32s(bytes: List<&2, U32>) -> List<&2, U32>:
  match bytes:
    case Con{a, Con{b, Con{c, Con{d, rest}}}}:
      ((a * 16777216) + (b * 65536) + (c * 256) + d : U32) <> be_u32s(rest)
    case _:
      Nil{}

# The count of an IDX file of 28 x 28 images, or None if it is not one.
def images_count(words: List<&2, U32>) -> Maybe<&2, U32>:
  match words:
    case Con{+magic, Con{+count, Con{+rows, Con{+columns, rest}}}}:
      Bool.pick(Maybe<&2, U32>, Bool.and(U32.is_eq(magic, 2051), Bool.and(U32.is_eq(rows, 28), U32.is_eq(columns, 28))),
        Some{count}, None{})
    case _:
      None{}

# The count of an IDX file of labels, or None if it is not one.
def labels_count(words: List<&2, U32>) -> Maybe<&2, U32>:
  match words:
    case Con{+magic, Con{+count, rest}}:
      Bool.pick(Maybe<&2, U32>, U32.is_eq(magic, 2049), Some{count}, None{})
    case _:
      None{}

# The count both files hold, or None if either is not an MNIST file or the
# counts differ.
def counted_pair(images: Maybe<&2, U32>, labels: Maybe<&2, U32>) -> Maybe<&2, U32>:
  match images labels:
    case Some{+n} Some{+m}:
      Bool.pick(Maybe<&2, U32>, U32.is_eq(n, m), Some{n}, None{})
    case images labels:
      None{}

# The count, if the files hold the header and `count` records and nothing
# else: 784 bytes an image, one byte a label. The count is below 5000000, so
# the sizes fit a U32.
def fitted(+image_size: U32, +label_size: U32, count: Maybe<&2, U32>) -> Maybe<&2, U32>:
  match count:
    case None{}:
      None{}
    case Some{+n}:
      Bool.pick(Maybe<&2, U32>, Bool.and(U32.is_lt(n, 5000000), Bool.and(U32.is_eq(image_size, (16 + (n * 784) : U32)),
        U32.is_eq(label_size, (8 + n : U32)))), Some{n}, None{})

def needed(+path: String, count: Maybe<&2, U32>) -> IO(U32):
  match count:
    case None{}:
      IO.die(U32, 4, "not an MNIST image and label pair: " ++ path)
    case Some{n}:
      IO.pure(U32, n)

# The number of examples of an MNIST image file and its label file. A
# header that is not MNIST's, counts that differ, or a file shorter or longer
# than its header says stops the program with exit code 4 before it trains.
def examples(+dir: String, +images: String, +labels: String) -> IO(U32):
  do IO<U32>:
    image_size : U32 <- file_size(dir ++ "/" ++ images)
    label_size : U32 <- file_size(dir ++ "/" ++ labels)
    head : List<&2, U32> <- bytes_at(dir ++ "/" ++ images, 0, 16)
    tail : List<&2, U32> <- bytes_at(dir ++ "/" ++ labels, 0, 8)
    needed(dir ++ "/" ++ images,
      fitted(image_size, label_size, counted_pair(images_count(be_u32s(head)), labels_count(be_u32s(tail)))))

def dumped(words: Bool, params: Params) -> IO(Unit):
  match words:
    case True{}:
      show_params(params)
    case False{}:
      IO.pure(Unit, Unit{})

def finish(+rows: U32, words: Bool, pair: Params & Nat) -> IO(Unit):
  (params, +n) = pair
  do IO<Unit>:
    IO.print("correct " ++ Nat.show(n) ++ " " ++ U32.show(rows))
    dumped(words, params)

def trained_on(+dir: String, +init: String, +order: String, +batch: U32, +steps: Nat, +lr: F32, words: Bool,
  +test_rows: U32) -> IO(Unit):
  do IO<Unit>:
    bytes : List<&2, U32> <- bytes_at(init, 0, 407080)
    params : Params <- params_of(floats(bytes))
    t0 : Nat <- IO.now()
    trained : Params <- train(steps, dir, order, batch, lr, 0, params)
    t1 : Nat <- IO.now()
    IO.print_err("train_ms " ++ Nat.show(Nat.sub(t1, t0)))
    t2 : Nat <- IO.now()
    result : Params & Nat <- evaluate(Nat.div(Nat.add(U32.to_nat(test_rows), 999n), 1000n), dir, 0, test_rows, 0n,
      (trained, Score{0, 0n}))
    t3 : Nat <- IO.now()
    IO.print_err("test_ms " ++ Nat.show(Nat.sub(t3, t2)))
    finish(test_rows, words, result)

# Checks both IDX pairs first, then trains and tests.
def run(+dir: String, +init: String, +order: String, +batch: U32, +steps: Nat, +lr: F32, words: Bool) -> IO(Unit):
  do IO<Unit>:
    train_rows : U32 <- examples(dir, "train-images-idx3-ubyte", "train-labels-idx1-ubyte")
    test_rows : U32 <- examples(dir, "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")
    trained_on(dir, init, order, batch, steps, lr, words, test_rows)

def usage() -> IO(Unit):
  IO.die(Unit, 1, "usage: train <mnist dir> <init file> <order file> <batch> <steps> <lr> [words]")

def parsed(+dir: String, +init: String, +order: String, batch: Maybe<&2, U32>, steps: Maybe<&2, Nat>,
  lr: Maybe<&2, F32>, words: Bool) -> IO(Unit):
  match batch steps lr:
    case Some{+b} Some{+n} Some{+r}:
      run(dir, init, order, b, n, r, words)
    case batch steps lr:
      usage()

def has_more(rest: List<String>) -> Bool:
  match rest:
    case Nil{}:
      False{}
    case Con{head, tail}:
      True{}

def dispatch(args: List<String>) -> IO(Unit):
  match args:
    case Con{program, Con{+dir, Con{+init, Con{+order, Con{batch, Con{steps, Con{lr, rest}}}}}}}:
      parsed(dir, init, order, U32.read(batch), Nat.read(steps), F32.read(lr), has_more(rest))
    case _:
      usage()

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    dispatch(args)
