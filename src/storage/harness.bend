import Base
import ./main.bend as Storage

type Step is Data:
  Read{index: U32}
  Write{index: U32, bits: U32}
  Copy{}
  ReadCopy{index: U32}
  WriteCopy{index: U32, bits: U32}

type Case is Data:
  Case{length: U32, bits: U32, steps: List<&2, Step>}

type State is Type:
  State{storage: Storage.Storage, copied: Maybe<Storage.Storage>}

def from_bits(word: U32) -> F32:
  U32{bits} = word
  F32{bits}

def error(error: Storage.Error) -> String:
  match error:
    case Storage.TooLarge{}:
      "e:TooLarge"
    case Storage.OutOfBounds{}:
      "e:OutOfBounds"
    case Storage.InvalidIndex{}:
      "e:InvalidIndex"

def read_text(result: Result<&2, &2, Storage.Error, F32>) -> String:
  match result:
    case Done{value}:
      "v:" ++ U32.show(F32.bits(value))
    case Fail{e}:
      error(e)

def write_text(result: Result<&2, &2, Storage.Error, Unit>) -> String:
  match result:
    case Done{unit}:
      "ok"
    case Fail{e}:
      error(e)

def finish_read(copied: Maybe<Storage.Storage>, pair: Storage.Storage & Result<&2, &2, Storage.Error, F32>) -> IO(State):
  (storage, result) = pair
  do IO<State>:
    IO.print(read_text(result))
    return State{storage, copied}

def finish_write(copied: Maybe<Storage.Storage>, pair: Storage.Storage & Result<&2, &2, Storage.Error, Unit>) -> IO(State):
  (storage, result) = pair
  do IO<State>:
    IO.print(write_text(result))
    return State{storage, copied}

def finish_copy(pair: Storage.Storage & Storage.Storage) -> IO(State):
  (storage, copied) = pair
  do IO<State>:
    IO.print("copy")
    return State{storage, Some{copied}}

def finish_read_copy(storage: Storage.Storage, pair: Storage.Storage & Result<&2, &2, Storage.Error, F32>) -> IO(State):
  (copied, result) = pair
  do IO<State>:
    IO.print(read_text(result))
    return State{storage, Some{copied}}

def finish_write_copy(storage: Storage.Storage, pair: Storage.Storage & Result<&2, &2, Storage.Error, Unit>) -> IO(State):
  (copied, result) = pair
  do IO<State>:
    IO.print(write_text(result))
    return State{storage, Some{copied}}

def read_copy(+index: U32, storage: Storage.Storage, copied: Maybe<Storage.Storage>) -> IO(State):
  match copied:
    case None{}:
      do IO<State>:
        IO.print("e:NoCopy")
        return State{storage, None{}}
    case Some{value}:
      finish_read_copy(storage, Storage.get(value, index))

def write_copy(+index: U32, bits: U32, storage: Storage.Storage, copied: Maybe<Storage.Storage>) -> IO(State):
  match copied:
    case None{}:
      do IO<State>:
        IO.print("e:NoCopy")
        return State{storage, None{}}
    case Some{value}:
      finish_write_copy(storage, Storage.set(value, index, from_bits(bits)))

def step(step: Step, state: State) -> IO(State):
  match step state:
    case Read{index} State{storage, copied}:
      finish_read(copied, Storage.get(storage, index))
    case Write{index, bits} State{storage, copied}:
      finish_write(copied, Storage.set(storage, index, from_bits(bits)))
    case Copy{} State{storage, copied}:
      finish_copy(Storage.clone(storage))
    case ReadCopy{index} State{storage, copied}:
      read_copy(index, storage, copied)
    case WriteCopy{index, bits} State{storage, copied}:
      write_copy(index, bits, storage, copied)

def steps(actions: List<&2, Step>, state: State) -> IO(Unit):
  match actions:
    case Nil{}:
      IO.pure(Unit, Unit{})
    case head <> tail:
      do IO<Unit>:
        next : State <- step(head, state)
        steps(tail, next)

def initialized(actions: List<&2, Step>, pair: Storage.Storage & Nat) -> IO(Unit):
  (storage, length) = pair
  do IO<Unit>:
    IO.print("new:" ++ Nat.show(length))
    steps(actions, State{storage, None{}})

def start(actions: List<&2, Step>, result: Result<&2, &1, Storage.Error, Storage.Storage>) -> IO(Unit):
  match result:
    case Fail{e}:
      IO.print(error(e))
    case Done{storage}:
      initialized(actions, Storage.length(storage))

def run(cases: List<&2, Case>) -> IO(Unit):
  match cases:
    case Nil{}:
      IO.pure(Unit, Unit{})
    case Case{length, bits, actions} <> tail:
      do IO<Unit>:
        start(actions, Storage.new(length, from_bits(bits)))
        run(tail)
