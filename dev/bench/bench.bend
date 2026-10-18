import Base
import ../../src/tensor/core.bend as Core
import ../../src/tensor/elementwise.bend as Elementwise
import ../../src/tensor/reduction.bend as Reduction
import ../../src/tensor/matmul.bend as Matrix
import ../../src/tensor/transforms.bend as Transforms
import ../../src/autograd/broadcast.bend as Broadcast
import ../../src/autograd/reduce.bend as Reduce
import ../../src/autograd/sum_to_size.bend as SumToSize
import ../../src/autograd/tensor_backward.bend as Backward
import ../../src/autograd/sgd.bend as Sgd
import ../../examples/mnist/train.bend as Mnist

# Times one workload on fixed inputs. tools/bench.py drives it.
#
#   bench <workload> <n> <reps>
#
# The program builds its inputs from a fixed formula, runs the workload once
# untimed, then `reps` times between two clock reads, and prints
#
#   ms <milliseconds of the reps>
#   digest <digest of the last output>
#
# The digest folds the F32 bits of every output word, FNV-1a over 32-bit
# words, then the word count. A failed operation prints `failed <code>` and
# exits 2.

# Inputs.

# Word i of input `seed`: small values of both signs, no zero run.
def value_at(+seed: U32, +i: U32) -> F32:
  +k = ((((i * 7919) + seed) % 2001) : U32)
  F32.div(F32.sub(U32.to_f32(k), 1000.0), 28000.0)

def values(left: Nat, +seed: U32, +i: U32) -> List<&2, F32>:
  match left:
    case 0n:
      Nil{}
    case 1n+rest:
      value_at(seed, i) <> values(rest, seed, (i + 1 : U32))

def numel(dims: List<&2, U32>) -> Nat:
  match dims:
    case Nil{}:
      1n
    case Con{d, rest}:
      Nat.mul(U32.to_nat(d), numel(rest))

def built(result: Result<&2, &1, Core.Error, Core.Dynamic>) -> IO(Core.Dynamic):
  match result:
    case Fail{error}:
      IO.die(Core.Dynamic, 2, "failed input " ++ U32.show(Core.error_code(error)))
    case Done{tensor}:
      IO.pure(Core.Dynamic, tensor)

def tensor(+dims: List<&2, U32>, +seed: U32) -> IO(Core.Dynamic):
  built(Core.construct(dims, values(numel(dims), seed, 0)))

# Class i * 7 % classes for row i.
def targets(left: Nat, +classes: U32, +i: U32) -> List<&2, U32>:
  match left:
    case 0n:
      Nil{}
    case 1n+rest:
      (((i * 7) % classes) : U32) <> targets(rest, classes, (i + 1 : U32))

# Digests.

def mix(+h: U32, +w: U32) -> U32:
  ((h .^. w) * 16777619 : U32)

def word_of(result: Result<&2, &2, Core.Error, F32>) -> U32:
  match result:
    case Fail{error}:
      4294967295
    case Done{value}:
      F32.bits(value)

def fold(left: Nat, +i: U32, +h: U32, pair: Core.Dynamic & Result<&2, &2, Core.Error, F32>) -> U32:
  match left:
    case 0n:
      h
    case 1n+rest:
      (t, r) = pair
      fold(rest, (i + 1 : U32), mix(h, word_of(r)), Core.read_dynamic(t, (i + 1 : U32)))

def counted(+h: U32, pair: Core.Dynamic & U32) -> U32:
  (t, +count) = pair
  mix(fold(U32.to_nat(count), 0, h, Core.read_dynamic(t, 0)), count)

def digest(+h: U32, t: Core.Dynamic) -> U32:
  counted(h, Core.count(t))

# Operations on one or two tensors that hand their inputs back.

type Form is Data:
  Plain{}
  NT{}
  TN{}

type Op is Data:
  Bin{op: Elementwise.Operation}
  BroadcastTo{dims: List<&2, U32>}
  Reshape{there: List<&2, U32>, back: List<&2, U32>}
  Transpose{}
  Permute{axes: List<&2, U32>}
  Sum{}
  ReduceTo{dims: List<&2, U32>}
  SumToSize{dims: List<&2, U32>}
  Product{form: Form}
  Clone{}

type Step is Type:
  Ok{a: Core.Dynamic, b: Core.Dynamic, out: Maybe<&1, Core.Dynamic>}
  Bad{code: U32}

def from_binary(pair: (Core.Dynamic & Core.Dynamic) & Result<&2, &1, Core.Error, Core.Dynamic>) -> Step:
  (ins, r) = pair
  (a, b) = ins
  match r:
    case Fail{error}:
      Bad{Core.error_code(error)}
    case Done{t}:
      Ok{a, b, Some{t}}

def from_broadcast(b: Core.Dynamic, pair: Core.Dynamic & Result<&2, &1, Broadcast.Error, Core.Dynamic>) -> Step:
  (a, r) = pair
  match r:
    case Fail{error}:
      Bad{Broadcast.error_code(error)}
    case Done{t}:
      Ok{a, b, Some{t}}

def from_reduce(b: Core.Dynamic, pair: Core.Dynamic & Result<&2, &1, Reduce.Error, Core.Dynamic>) -> Step:
  (a, r) = pair
  match r:
    case Fail{error}:
      Bad{Reduce.error_code(error)}
    case Done{t}:
      Ok{a, b, Some{t}}

def from_transform(b: Core.Dynamic, pair: Core.Dynamic & Result<&2, &1, Transforms.Error, Core.Dynamic>) -> Step:
  (a, r) = pair
  match r:
    case Fail{error}:
      Bad{Transforms.error_code(error)}
    case Done{t}:
      Ok{a, b, Some{t}}

def from_sum(b: Core.Dynamic, r: Result<&2, &1, Core.Error, Core.Dynamic & Core.Dynamic>) -> Step:
  match r:
    case Fail{error}:
      Bad{Core.error_code(error)}
    case Done{pair}:
      (a, t) = pair
      Ok{a, b, Some{t}}

def from_product(r: Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>) -> Step:
  match r:
    case Fail{error}:
      Bad{1}
    case Done{triple}:
      (a, rest) = triple
      (b, t) = rest
      Ok{a, b, Some{t}}

def reshaped_back(b: Core.Dynamic, r: Result<&2, &1, Transforms.Error, Core.Dynamic>) -> Step:
  match r:
    case Fail{error}:
      Bad{Transforms.error_code(error)}
    case Done{t}:
      Ok{t, b, None{}}

def reshaped(+back: List<&2, U32>, b: Core.Dynamic, r: Result<&2, &1, Transforms.Error, Core.Dynamic>) -> Step:
  match r:
    case Fail{error}:
      Bad{Transforms.error_code(error)}
    case Done{t}:
      reshaped_back(b, Transforms.reshape_dynamic(t, back))

def product(+form: Form, a: Core.Dynamic, b: Core.Dynamic) -> Result<&2, &1, Matrix.Error, Core.Dynamic & Core.Dynamic & Core.Dynamic>:
  match form:
    case Plain{}:
      Matrix.matmul(a, b)
    case NT{}:
      Matrix.matmul_nt(a, b)
    case TN{}:
      Matrix.matmul_tn(a, b)

def from_clone(b: Core.Dynamic, pair: Core.Dynamic & Core.Dynamic) -> Step:
  (a, t) = pair
  Ok{a, b, Some{t}}

def apply(+op: Op, a: Core.Dynamic, b: Core.Dynamic) -> Step:
  match op:
    case Bin{+o}:
      from_binary(Elementwise.binary(o, a, b))
    case BroadcastTo{+dims}:
      from_broadcast(b, Broadcast.broadcast_to_dimensions(a, dims))
    case Reshape{+there, +back}:
      reshaped(back, b, Transforms.reshape_dynamic(a, there))
    case Transpose{}:
      from_transform(b, Transforms.transpose_dynamic(a, 0, 1))
    case Permute{+axes}:
      from_transform(b, Transforms.permute_dynamic(a, axes))
    case Sum{}:
      from_sum(b, Reduction.sum_dynamic(a))
    case ReduceTo{+dims}:
      from_reduce(b, Reduce.reduce_to_dimensions(a, dims))
    case SumToSize{+dims}:
      from_reduce(b, SumToSize.sum_to_size_dimensions(a, dims))
    case Product{+form}:
      from_product(product(form, a, b))
    case Clone{}:
      from_clone(b, Core.clone_dynamic(a))

def again(left: Nat, +op: Op, step: Step) -> Step:
  match left:
    case 0n:
      step
    case 1n+rest:
      match step:
        case Ok{a, b, out}:
          again(rest, op, apply(op, a, b))
        case Bad{+code}:
          Bad{code}

def step_digest(step: Step) -> Result<&2, &2, U32, U32>:
  match step:
    case Bad{+code}:
      Fail{code}
    case Ok{a, b, out}:
      match out:
        case None{}:
          Done{digest(2166136261, a)}
        case Some{t}:
          Done{digest(2166136261, t)}

# Tape workloads: a list of prepared tapes, one recording or sweep each.

# A recording names the id it must return, the id whose value the workload
# digests; a sweep starts at the tape's newest entry, the output the
# prepared recordings returned last.
type Act is Data:
  Linear{output: Nat}
  Sweep{}
  Mse{output: Nat}
  Entropy{targets: List<&2, U32>, output: Nat}
  Descend{}

def status(-A: Data, result: Result<&2, &2, Backward.Error, A>) -> U32:
  match result:
    case Fail{error}:
      (1000 + Backward.error_code(error) : U32)
    case Done{value}:
      0

def descent_status(result: Result<&2, &2, Sgd.Error, Unit>) -> U32:
  match result:
    case Fail{error}:
      (2000 + Sgd.error_code(error) : U32)
    case Done{value}:
      0

def coded(-A: Data, pair: Backward.Tape & Result<&2, &2, Backward.Error, A>) -> Backward.Tape & U32:
  (tape, r) = pair
  (tape, status(A, r))

# A recording that returns another id than the workload names fails with
# 3000.
def recorded_as(+output: Nat, pair: Backward.Tape & Result<&2, &2, Backward.Error, Nat>) -> Backward.Tape & U32:
  (tape, r) = pair
  match r:
    case Fail{error}:
      (tape, (1000 + Backward.error_code(error) : U32))
    case Done{+id}:
      (tape, Bool.pick(U32, Nat.is_eq(id, output), 0, 3000))

# The tape's newest entry, the last recording's output.
def newest(pair: List<&1, Backward.Entry> & Nat) -> Backward.Tape & Nat:
  (entries, +count) = pair
  (Backward.Tape{entries}, Nat.sub(count, 1n))

def sweep_from(pair: Backward.Tape & Nat) -> Backward.Tape & Result<&2, &2, Backward.Error, Unit>:
  (tape, +output) = pair
  Backward.backward(tape, output)

# The backward of the newest entry. An empty tape has none and fails.
def sweep_newest(tape: Backward.Tape) -> Backward.Tape & Result<&2, &2, Backward.Error, Unit>:
  Backward.Tape{entries} = tape
  sweep_from(newest(Backward.counted(entries)))

def descended(pair: Backward.Tape & Result<&2, &2, Sgd.Error, Unit>) -> Backward.Tape & U32:
  (tape, r) = pair
  (tape, descent_status(r))

def act(+a: Act, tape: Backward.Tape) -> Backward.Tape & U32:
  match a:
    case Linear{+output}:
      recorded_as(output, Backward.linear(tape, 0n, 1n, 2n))
    case Sweep{}:
      coded(Unit, sweep_newest(tape))
    case Mse{+output}:
      recorded_as(output, Backward.mse_loss(tape, 0n, 1n))
    case Entropy{+targets, +output}:
      recorded_as(output, Backward.cross_entropy(tape, 0n, targets))
    case Descend{}:
      descended(Sgd.step(tape, 0.1, [0n]))

type Acted is Type:
  Acted{tapes: List<&1, Backward.Tape>, code: U32}

def first(+a: U32, +b: U32) -> U32:
  Bool.pick(U32, U32.is_eq(a, 0), b, a)

def acted_rest(tape: Backward.Tape, +c: U32, later: Acted) -> Acted:
  Acted{tapes, +more} = later
  Acted{tape <> tapes, first(c, more)}

def acted_one(pair: Backward.Tape & U32, later: Acted) -> Acted:
  (tape, +c) = pair
  acted_rest(tape, c, later)

def run_tapes(+a: Act, tapes: List<&1, Backward.Tape>) -> Acted:
  match tapes:
    case Nil{}:
      Acted{Nil{}, 0}
    case Con{tape, rest}:
      acted_one(act(a, tape), run_tapes(a, rest))

def leaf(pair: Backward.Tape & Nat) -> Backward.Tape:
  (tape, id) = pair
  tape

# A preparing recording or sweep that fails leaves an empty tape, so the
# workload's act fails instead of timing a shorter tape.
def recorded(pair: Backward.Tape & Result<&2, &2, Backward.Error, Nat>) -> Backward.Tape:
  (tape, r) = pair
  match r:
    case Fail{error}:
      Backward.empty()
    case Done{id}:
      tape

def swept(pair: Backward.Tape & Result<&2, &2, Backward.Error, Unit>) -> Backward.Tape:
  (tape, r) = pair
  match r:
    case Fail{error}:
      Backward.empty()
    case Done{unit}:
      tape

# The digest of the value of each listed id, then of the gradient of each
# id in the second list, chained. A listed id without a value fails with
# 9997, and one whose gradient slot is empty with 9998: a workload lists the
# leaves its sweep must reach, the biases among them.
def value_digest(+h: U32, pair: Backward.Tape & Result<&2, &1, Backward.Error, Core.Dynamic>) -> Backward.Tape & Result<&2, &2, U32, U32>:
  (tape, r) = pair
  match r:
    case Fail{error}:
      (tape, Fail{9997})
    case Done{t}:
      (tape, Done{digest(h, t)})

def grad_digest(+h: U32, pair: Backward.Tape & Result<&2, &1, Backward.Error, Maybe<&1, Core.Dynamic>>) ->
  Backward.Tape & Result<&2, &2, U32, U32>:
  (tape, r) = pair
  match r:
    case Fail{error}:
      (tape, Fail{9997})
    case Done{slot}:
      match slot:
        case None{}:
          (tape, Fail{9998})
        case Some{t}:
          (tape, Done{digest(h, t)})

def values_next(rec: U32 -> Backward.Tape -> Backward.Tape & Result<&2, &2, U32, U32>, pair: Backward.Tape & Result<&2, &2, U32, U32>) ->
  Backward.Tape & Result<&2, &2, U32, U32>:
  (tape, r) = pair
  match r:
    case Fail{+code}:
      (tape, Fail{code})
    case Done{+h}:
      rec(h, tape)

def grads_next(rec: U32 -> Backward.Tape -> Result<&2, &2, U32, U32>, pair: Backward.Tape & Result<&2, &2, U32, U32>) ->
  Result<&2, &2, U32, U32>:
  (tape, r) = pair
  match r:
    case Fail{+code}:
      Fail{code}
    case Done{+h}:
      rec(h, tape)

def values_digest(ids: List<&2, Nat>, +h: U32, tape: Backward.Tape) -> Backward.Tape & Result<&2, &2, U32, U32>:
  match ids:
    case Nil{}:
      (tape, Done{h})
    case Con{+id, rest}:
      values_next(g => t => values_digest(rest, g, t), value_digest(h, Backward.value(tape, id)))

def grads_digest(ids: List<&2, Nat>, +h: U32, tape: Backward.Tape) -> Result<&2, &2, U32, U32>:
  match ids:
    case Nil{}:
      Done{h}
    case Con{+id, rest}:
      grads_next(g => t => grads_digest(rest, g, t), grad_digest(h, Backward.grad(tape, id)))

def checked_digest(+code: U32, h: Result<&2, &2, U32, U32>) -> Result<&2, &2, U32, U32>:
  match h:
    case Fail{+other}:
      Bool.pick(Result<&2, &2, U32, U32>, U32.is_eq(code, 0), Fail{other}, Fail{code})
    case Done{+value}:
      Bool.pick(Result<&2, &2, U32, U32>, U32.is_eq(code, 0), Done{value}, Fail{code})

def tape_digest(+values: List<&2, Nat>, +grads: List<&2, Nat>, acted: Acted) -> Result<&2, &2, U32, U32>:
  Acted{tapes, +code} = acted
  match tapes:
    case Nil{}:
      Fail{9999}
    case Con{tape, rest}:
      checked_digest(code, grads_next(g => t => grads_digest(grads, g, t), values_digest(values, 2166136261, tape)))

# MNIST: Mnist.step on a batch, the program's own training step.

type Batch is Type:
  Batch{params: Mnist.Params, x: Core.Dynamic}

def train_all(+targets: List<&2, U32>, batches: List<&1, Batch>) -> List<&1, Mnist.Trained>:
  match batches:
    case Nil{}:
      Nil{}
    case Con{batch, rest}:
      Batch{params, x} = batch
      Mnist.step(params, x, targets, 0.1) <> train_all(targets, rest)

def params_digest(+loss: F32, params: Mnist.Params) -> U32:
  Mnist.Params{w1, b1, w2, b2} = params
  digest(digest(digest(digest(mix(2166136261, F32.bits(loss)), w1), b1), w2), b2)

def trained_digest(list: List<&1, Mnist.Trained>) -> Result<&2, &2, U32, U32>:
  match list:
    case Nil{}:
      Fail{9999}
    case Con{trained, rest}:
      match trained:
        case Mnist.Broken{+code}:
          Fail{code}
        case Mnist.Trained{params, +loss}:
          Done{params_digest(loss, params)}

# Timing.

def stamped(-A: Type, x: A) -> IO(A & Nat):
  do IO<A & Nat>:
    t : Nat <- IO.now()
    return (x, t)

def report(+t0: Nat, pair: Result<&2, &2, U32, U32> & Nat) -> IO(Unit):
  (r, +t1) = pair
  match r:
    case Fail{+code}:
      IO.die(Unit, 2, "failed " ++ U32.show(code))
    case Done{+h}:
      do IO<Unit>:
        IO.print("ms " ++ Nat.show(Nat.sub(t1, t0)))
        IO.print("digest " ++ U32.show(h))

def report_step(+t0: Nat, pair: Step & Nat) -> IO(Unit):
  (last, +t1) = pair
  report(t0, (step_digest(last), t1))

def report_tapes(+values: List<&2, Nat>, +grads: List<&2, Nat>, +t0: Nat, pair: Acted & Nat) -> IO(Unit):
  (acted, +t1) = pair
  report(t0, (tape_digest(values, grads, acted), t1))

def report_mnist(+t0: Nat, pair: List<&1, Mnist.Trained> & Nat) -> IO(Unit):
  (list, +t1) = pair
  report(t0, (trained_digest(list), t1))

def warm_op(+reps: Nat, +op: Op, step: Step) -> IO(Unit):
  do IO<Unit>:
    t0 : Nat <- IO.now()
    pair : Step & Nat <- stamped(Step, again(reps, op, step))
    report_step(t0, pair)

def time_op(+reps: Nat, +op: Op, a: Core.Dynamic, b: Core.Dynamic) -> IO(Unit):
  warm_op(reps, op, apply(op, a, b))

# The prepared tapes: `reps` + 1 copies built by `make`, the first used
# untimed.
type Tapes is Type:
  Tapes{warm: Backward.Tape, timed: List<&1, Backward.Tape>}

def timed_after(+a: Act, +values: List<&2, Nat>, +grads: List<&2, Nat>, warmed: Backward.Tape & U32,
  timed: List<&1, Backward.Tape>) -> IO(Unit):
  do IO<Unit>:
    t0 : Nat <- IO.now()
    pair : Acted & Nat <- stamped(Acted, run_tapes(a, timed))
    report_tapes(values, grads, t0, pair)

def time_tapes(+a: Act, +values: List<&2, Nat>, +grads: List<&2, Nat>, prepared: Tapes) -> IO(Unit):
  Tapes{warm, timed} = prepared
  timed_after(a, values, grads, act(a, warm), timed)

# Building the prepared tapes.

# A tape whose leaves are fresh inputs of these shapes, seeds 1, 2, ...
def leaves(dims: List<&2, List<&2, U32>>, +seed: U32, tape: Backward.Tape) -> IO(Backward.Tape):
  match dims:
    case Nil{}:
      IO.pure(Backward.Tape, tape)
    case Con{+d, rest}:
      do IO<Backward.Tape>:
        t : Core.Dynamic <- tensor(d, seed)
        leaves(rest, (seed + 1 : U32), leaf(Backward.leaf(tape, t)))

type Prep is Data:
  Bare{}
  ThenLinear{}
  ThenRelu{}
  ThenMse{}
  ThenEntropy{targets: List<&2, U32>}
  ThenMulSwept{}
  ThenMul{}
  ThenSum{}
  ThenMatmul{}
  ThenMlp{targets: List<&2, U32>}

# 64-32-10 on the leaves W1, b1, W2, b2, x (ids 0 to 4): each recording
# takes the id the one before it returned, so the loss is the tape's newest
# entry. A refused recording leaves an empty tape.
def mlp_loss(+targets: List<&2, U32>, pair: Backward.Tape & Result<&2, &2, Backward.Error, Nat>) -> Backward.Tape:
  (tape, r) = pair
  match r:
    case Fail{error}:
      Backward.empty()
    case Done{+logits}:
      recorded(Backward.cross_entropy(tape, logits, targets))

def mlp_second(+targets: List<&2, U32>, pair: Backward.Tape & Result<&2, &2, Backward.Error, Nat>) -> Backward.Tape:
  (tape, r) = pair
  match r:
    case Fail{error}:
      Backward.empty()
    case Done{+hidden}:
      mlp_loss(targets, Backward.linear(tape, hidden, 2n, 3n))

def mlp_relu(+targets: List<&2, U32>, pair: Backward.Tape & Result<&2, &2, Backward.Error, Nat>) -> Backward.Tape:
  (tape, r) = pair
  match r:
    case Fail{error}:
      Backward.empty()
    case Done{+first}:
      mlp_second(targets, Backward.relu(tape, first))

def mlp(+targets: List<&2, U32>, tape: Backward.Tape) -> Backward.Tape:
  mlp_relu(targets, Backward.linear(tape, 4n, 0n, 1n))

def prepared(+prep: Prep, tape: Backward.Tape) -> Backward.Tape:
  match prep:
    case Bare{}:
      tape
    case ThenLinear{}:
      recorded(Backward.linear(tape, 0n, 1n, 2n))
    case ThenRelu{}:
      recorded(Backward.relu(tape, 0n))
    case ThenMse{}:
      recorded(Backward.mse_loss(tape, 0n, 1n))
    case ThenEntropy{+targets}:
      recorded(Backward.cross_entropy(tape, 0n, targets))
    case ThenMulSwept{}:
      swept(sweep_newest(recorded(Backward.mul(tape, 0n, 1n))))
    case ThenMul{}:
      recorded(Backward.mul(tape, 0n, 1n))
    case ThenSum{}:
      recorded(Backward.sum(tape, 0n))
    case ThenMatmul{}:
      recorded(Backward.matmul(tape, 0n, 1n))
    case ThenMlp{+targets}:
      mlp(targets, tape)

def make(+dims: List<&2, List<&2, U32>>, +prep: Prep) -> IO(Backward.Tape):
  do IO<Backward.Tape>:
    tape : Backward.Tape <- leaves(dims, 1, Backward.empty())
    return prepared(prep, tape)

def make_list(left: Nat, +dims: List<&2, List<&2, U32>>, +prep: Prep) -> IO(List<&1, Backward.Tape>):
  match left:
    case 0n:
      IO.pure(List<&1, Backward.Tape>, Nil{})
    case 1n+rest:
      do IO<List<&1, Backward.Tape>>:
        tape : Backward.Tape <- make(dims, prep)
        tapes : List<&1, Backward.Tape> <- make_list(rest, dims, prep)
        return tape <> tapes

def tapes(+reps: Nat, +dims: List<&2, List<&2, U32>>, +prep: Prep) -> IO(Tapes):
  do IO<Tapes>:
    warm : Backward.Tape <- make(dims, prep)
    timed : List<&1, Backward.Tape> <- make_list(reps, dims, prep)
    return Tapes{warm, timed}

def tape_job(+reps: Nat, +dims: List<&2, List<&2, U32>>, +prep: Prep, +a: Act, +values: List<&2, Nat>,
  +grads: List<&2, Nat>) -> IO(Unit):
  do IO<Unit>:
    prepared : Tapes <- tapes(reps, dims, prep)
    time_tapes(a, values, grads, prepared)

# MNIST batches.

def mnist_params() -> IO(Mnist.Params):
  do IO<Mnist.Params>:
    w1 : Core.Dynamic <- tensor([128, 784], 1)
    b1 : Core.Dynamic <- tensor([128], 2)
    w2 : Core.Dynamic <- tensor([10, 128], 3)
    b2 : Core.Dynamic <- tensor([10], 4)
    return Mnist.Params{w1, b1, w2, b2}

def batch(+rows: U32) -> IO(Batch):
  do IO<Batch>:
    params : Mnist.Params <- mnist_params()
    x : Core.Dynamic <- tensor([rows, 784], 5)
    return Batch{params, x}

def batches(left: Nat, +rows: U32) -> IO(List<&1, Batch>):
  match left:
    case 0n:
      IO.pure(List<&1, Batch>, Nil{})
    case 1n+rest:
      do IO<List<&1, Batch>>:
        b : Batch <- batch(rows)
        bs : List<&1, Batch> <- batches(rest, rows)
        return b <> bs

def mnist_job(+reps: Nat, +rows: U32) -> IO(Unit):
  +ts = targets(U32.to_nat(rows), 10, 0)
  do IO<Unit>:
    warm : Batch <- batch(rows)
    timed : List<&1, Batch> <- batches(reps, rows)
    w : List<&1, Mnist.Trained> <- IO.pure(List<&1, Mnist.Trained>, train_all(ts, Con{warm, Nil{}}))
    t0 : Nat <- IO.now()
    pair : List<&1, Mnist.Trained> & Nat <- stamped(List<&1, Mnist.Trained>, train_all(ts, timed))
    report_mnist(t0, pair)

# Construction and MNIST evaluation: pure loops timed whole, each rep
# building its own [n, 784] batch from a list, as the MNIST program does.

def constructed(+h: U32, result: Result<&2, &1, Core.Error, Core.Dynamic>) -> U32:
  match result:
    case Fail{error}:
      mix(h, 4294967295)
    case Done{t}:
      digest(h, t)

def construct_loop(left: Nat, +rows: U32, +h: U32) -> U32:
  match left:
    case 0n:
      h
    case 1n+rest:
      construct_loop(rest, rows, constructed(h, Core.construct([rows, 784], values(numel([rows, 784]), 5, 0))))

def scored_digest(+h: U32, pair: Mnist.Params & Mnist.Score) -> Mnist.Params & U32:
  (params, score) = pair
  Mnist.Score{+code, +correct} = score
  (params, mix(mix(h, code), U32.from_nat(correct)))

def evaluated(+rows: U32, +h: U32, params: Mnist.Params, result: Result<&2, &1, Core.Error, Core.Dynamic>) ->
  Mnist.Params & U32:
  match result:
    case Fail{error}:
      (params, mix(h, 4294967295))
    case Done{x}:
      scored_digest(h, Mnist.scored(params, x, targets(U32.to_nat(rows), 10, 0)))

def eval_next(left: Nat, +rows: U32, pair: Mnist.Params & U32) -> Mnist.Params & U32:
  match left:
    case 0n:
      pair
    case 1n+rest:
      (params, +h) = pair
      eval_next(rest, rows, evaluated(rows, h, params, Core.construct([rows, 784], values(numel([rows, 784]), 5, 0))))

def report_pure(+t0: Nat, pair: U32 & Nat) -> IO(Unit):
  (+h, +t1) = pair
  report(t0, (Done{h}, t1))

def eval_digest(pair: Mnist.Params & U32) -> U32:
  (params, +h) = pair
  h

def construct_job(+reps: Nat, +rows: U32) -> IO(Unit):
  +warm = construct_loop(1n, rows, 2166136261)
  do IO<Unit>:
    t0 : Nat <- IO.now()
    pair : U32 & Nat <- stamped(U32, construct_loop(reps, rows, warm))
    report_pure(t0, pair)

def eval_timed(+reps: Nat, +rows: U32, warm: Mnist.Params & U32) -> IO(Unit):
  do IO<Unit>:
    t0 : Nat <- IO.now()
    pair : U32 & Nat <- stamped(U32, eval_digest(eval_next(reps, rows, warm)))
    report_pure(t0, pair)

def eval_job(+reps: Nat, +rows: U32) -> IO(Unit):
  do IO<Unit>:
    params : Mnist.Params <- mnist_params()
    eval_timed(reps, rows, eval_next(1n, rows, (params, 2166136261)))

# Workloads.

def unary(+reps: Nat, +o: Elementwise.Operation, +n: U32) -> IO(Unit):
  do IO<Unit>:
    a : Core.Dynamic <- tensor([n], 1)
    zero : Core.Dynamic <- built(Core.construct([], [0.0]))
    time_op(reps, Bin{o}, a, zero)

def binary(+reps: Nat, +o: Elementwise.Operation, +ad: List<&2, U32>, +bd: List<&2, U32>) -> IO(Unit):
  do IO<Unit>:
    a : Core.Dynamic <- tensor(ad, 1)
    b : Core.Dynamic <- tensor(bd, 2)
    time_op(reps, Bin{o}, a, b)

def single(+reps: Nat, +op: Op, +ad: List<&2, U32>) -> IO(Unit):
  do IO<Unit>:
    a : Core.Dynamic <- tensor(ad, 1)
    b : Core.Dynamic <- built(Core.construct([], [0.0]))
    time_op(reps, op, a, b)

def binary_product(+reps: Nat, +form: Form, +sq: List<&2, U32>) -> IO(Unit):
  do IO<Unit>:
    a : Core.Dynamic <- tensor(sq, 1)
    b : Core.Dynamic <- tensor(sq, 2)
    time_op(reps, Product{form}, a, b)

type Workload is Data:
  WAdd{}
  WSub{}
  WMul{}
  WAddBroadcast{}
  WNeg{}
  WRelu{}
  WClone{}
  WBroadcastTo{}
  WReshape{}
  WTranspose{}
  WPermute{}
  WSum{}
  WReduceCols{}
  WReduceRows{}
  WSumToSizeCols{}
  WSumToSizeRows{}
  WMatmul{}
  WMatmulNT{}
  WMatmulTN{}
  WLinearForward{}
  WLinearBackward{}
  WReluBackward{}
  WMseForward{}
  WMseBackward{}
  WEntropyForward{}
  WEntropyBackward{}
  WSgd{}
  WMlpBackward{}
  WMnistStep{}
  WMulBackward{}
  WSumBackward{}
  WMatmulBackward{}
  WConstruct{}
  WMnistEval{}

type Named is Data:
  Named{name: String, workload: Workload}

def names() -> List<&2, Named>:
  [Named{"add", WAdd{}}, Named{"sub", WSub{}}, Named{"mul", WMul{}}, Named{"add_bcast", WAddBroadcast{}}, Named{"neg", WNeg{}},
    Named{"relu", WRelu{}}, Named{"clone", WClone{}}, Named{"broadcast_to", WBroadcastTo{}}, Named{"reshape", WReshape{}},
    Named{"transpose", WTranspose{}}, Named{"permute", WPermute{}}, Named{"sum", WSum{}}, Named{"reduce_cols", WReduceCols{}},
    Named{"reduce_rows", WReduceRows{}}, Named{"sum_to_size_cols", WSumToSizeCols{}}, Named{"sum_to_size_rows", WSumToSizeRows{}},
    Named{"matmul", WMatmul{}}, Named{"matmul_nt", WMatmulNT{}}, Named{"matmul_tn", WMatmulTN{}}, Named{"linear_fwd", WLinearForward{}},
    Named{"linear_bwd", WLinearBackward{}}, Named{"relu_bwd", WReluBackward{}}, Named{"mse_fwd", WMseForward{}},
    Named{"mse_bwd", WMseBackward{}}, Named{"ce_fwd", WEntropyForward{}}, Named{"ce_bwd", WEntropyBackward{}}, Named{"sgd", WSgd{}},
    Named{"mlp_bwd", WMlpBackward{}}, Named{"mnist_step", WMnistStep{}},
    Named{"mul_bwd", WMulBackward{}}, Named{"sum_bwd", WSumBackward{}}, Named{"matmul_bwd", WMatmulBackward{}},
    Named{"construct", WConstruct{}}, Named{"mnist_eval", WMnistEval{}}]

def lookup(+name: String, table: List<&2, Named>) -> Maybe<&2, Workload>:
  match table:
    case Nil{}:
      None{}
    case Con{Named{+key, +w}, rest}:
      Bool.pick(Maybe<&2, Workload>, String.eq(name, key), Some{w}, lookup(name, rest))

def run_with(w: Workload, +n: U32, +reps: Nat, +sq: List<&2, U32>, +ce: List<&2, U32>,
  +layer: List<&2, List<&2, U32>>) -> IO(Unit):
  match w:
    case WAdd{}:
      binary(reps, Elementwise.Add{}, [n], [n])
    case WSub{}:
      binary(reps, Elementwise.Sub{}, [n], [n])
    case WMul{}:
      binary(reps, Elementwise.Mul{}, [n], [n])
    case WAddBroadcast{}:
      binary(reps, Elementwise.Add{}, sq, [n])
    case WNeg{}:
      unary(reps, Elementwise.NegateLeft{}, n)
    case WRelu{}:
      unary(reps, Elementwise.ReluLeft{}, n)
    case WClone{}:
      single(reps, Clone{}, [n])
    case WBroadcastTo{}:
      single(reps, BroadcastTo{sq}, [n])
    case WReshape{}:
      single(reps, Reshape{[(n * n : U32)], sq}, sq)
    case WTranspose{}:
      single(reps, Transpose{}, sq)
    case WPermute{}:
      single(reps, Permute{[2, 0, 1]}, [4, n, (n / 4 : U32)])
    case WSum{}:
      single(reps, Sum{}, [n])
    case WReduceCols{}:
      single(reps, ReduceTo{[n]}, sq)
    case WReduceRows{}:
      single(reps, ReduceTo{[n, 1]}, sq)
    case WSumToSizeCols{}:
      single(reps, SumToSize{[n]}, sq)
    case WSumToSizeRows{}:
      single(reps, SumToSize{[n, 1]}, sq)
    case WMatmul{}:
      binary_product(reps, Plain{}, sq)
    case WMatmulNT{}:
      binary_product(reps, NT{}, sq)
    case WMatmulTN{}:
      binary_product(reps, TN{}, sq)
    case WLinearForward{}:
      tape_job(reps, layer, Bare{}, Linear{4n}, [4n], [])
    case WLinearBackward{}:
      tape_job(reps, layer, ThenLinear{}, Sweep{}, [], [0n, 1n, 2n])
    case WReluBackward{}:
      tape_job(reps, [[n]], ThenRelu{}, Sweep{}, [], [0n])
    case WMseForward{}:
      tape_job(reps, [[n], [n]], Bare{}, Mse{2n}, [2n], [])
    case WMseBackward{}:
      tape_job(reps, [[n], [n]], ThenMse{}, Sweep{}, [], [0n, 1n])
    case WEntropyForward{}:
      tape_job(reps, [[n, 10]], Bare{}, Entropy{ce, 1n}, [1n], [])
    case WEntropyBackward{}:
      tape_job(reps, [[n, 10]], ThenEntropy{ce}, Sweep{}, [], [0n])
    case WSgd{}:
      tape_job(reps, [[n], [n]], ThenMulSwept{}, Descend{}, [0n], [])
    case WMlpBackward{}:
      tape_job(reps, [[32, 64], [32], [10, 32], [10], [n, 64]], ThenMlp{ce}, Sweep{}, [], [0n, 1n, 2n, 3n, 4n])
    case WMnistStep{}:
      mnist_job(reps, n)
    case WMulBackward{}:
      tape_job(reps, [[n], [n]], ThenMul{}, Sweep{}, [], [0n, 1n])
    case WSumBackward{}:
      tape_job(reps, [[n]], ThenSum{}, Sweep{}, [], [0n])
    case WMatmulBackward{}:
      tape_job(reps, [sq, sq], ThenMatmul{}, Sweep{}, [], [0n, 1n])
    case WConstruct{}:
      construct_job(reps, n)
    case WMnistEval{}:
      eval_job(reps, n)

def run(+w: Workload, +n: U32, +reps: Nat) -> IO(Unit):
  run_with(w, n, reps, [n, n], targets(U32.to_nat(n), 10, 0), [[64, n], [128, n], [128]])

def usage() -> IO(Unit):
  IO.die(Unit, 1, "usage: bench <workload> <n> <reps>")

def parsed_reps(+v: Workload, +k: U32, reps: Maybe<&2, Nat>) -> IO(Unit):
  match reps:
    case None{}:
      IO.die(Unit, 1, "bad reps")
    case Some{+r}:
      run(v, k, r)

def parsed_n(+v: Workload, n: Maybe<&2, U32>, reps: Maybe<&2, Nat>) -> IO(Unit):
  match n:
    case None{}:
      IO.die(Unit, 1, "bad n")
    case Some{+k}:
      parsed_reps(v, k, reps)

def parsed(w: Maybe<&2, Workload>, n: Maybe<&2, U32>, reps: Maybe<&2, Nat>) -> IO(Unit):
  match w:
    case None{}:
      IO.die(Unit, 1, "unknown workload")
    case Some{+v}:
      parsed_n(v, n, reps)

def dispatch(args: List<String>) -> IO(Unit):
  match args:
    case Con{program, Con{+name, Con{n, Con{reps, rest}}}}:
      parsed(lookup(name, names()), U32.read(n), Nat.read(reps))
    case _:
      usage()

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    dispatch(args)
