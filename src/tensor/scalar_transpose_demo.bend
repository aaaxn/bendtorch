import Base
import ./core.bend as Core
import ./transforms.bend as Transforms
import ./transform_demo.bend as Demo

def after_transpose(+a: U32, +b: U32, pair: Core.Dynamic & Result<&2, &1, Transforms.Error, Core.Dynamic>) -> IO(Unit):
  (source, result) = pair
  match result:
    case Fail{error}:
      do IO<Unit>:
        IO.print("transpose:error=" ++ U32.show(Transforms.error_code(error)))
        source : Core.Dynamic <- Demo.report("retained", source)
        return Unit{}
    case Done{output}:
      do IO<Unit>:
        source : Core.Dynamic <- Demo.report("retained", source)
        output : Core.Dynamic <- Demo.report("result", output)
        Demo.after_inverse(Transforms.transpose_dynamic(output, a, b))

def created(name: String, +a: U32, +b: U32, result: Result<&2, &1, Core.Error, Core.Dynamic>) -> IO(Unit):
  match result:
    case Fail{error}:
      IO.print("construction:error=" ++ U32.show(Core.error_code(error)))
    case Done{tensor}:
      do IO<Unit>:
        IO.print(name)
        after_transpose(a, b, Transforms.transpose_dynamic(tensor, a, b))

def case_scalar(name: String, value: F32, +a: U32, +b: U32) -> IO(Unit):
  created(name, a, b, Core.construct([], [value]))

def written(pair: Core.Dynamic & Result<&2, &2, Core.Error, Unit>) -> IO(Core.Dynamic):
  (owner, status) = pair
  match status:
    case Fail{error}:
      do IO<Core.Dynamic>:
        IO.print("write:error=" ++ U32.show(Core.error_code(error)))
        return owner
    case Done{unit}:
      do IO<Core.Dynamic>:
        IO.print("write:done")
        return owner

def ownership(pair: Core.Dynamic & Result<&2, &1, Transforms.Error, Core.Dynamic>) -> IO(Unit):
  (source, result) = pair
  match result:
    case Fail{error}:
      IO.print("transpose:error=" ++ U32.show(Transforms.error_code(error)))
    case Done{output}:
      do IO<Unit>:
        IO.print("scalar_ownership")
        output : Core.Dynamic <- written(Core.write_dynamic(output, 0, 99.0))
        source : Core.Dynamic <- Demo.report("retained_after_write", source)
        output : Core.Dynamic <- Demo.report("output_after_write", output)
        return Unit{}

def ownership_created(result: Result<&2, &1, Core.Error, Core.Dynamic>) -> IO(Unit):
  match result:
    case Fail{error}:
      IO.print("construction:error=" ++ U32.show(Core.error_code(error)))
    case Done{tensor}:
      ownership(Transforms.transpose_dynamic(tensor, 0, 0))

def main() -> IO(Unit):
  do IO<Unit>:
    case_scalar("scalar", 7.0, 0, 0)
    case_scalar("scalar_negative_zero", Demo.from_word(2147483648), 0, 0)
    case_scalar("scalar_subnormal", Demo.from_word(1), 0, 0)
    case_scalar("scalar_axis_a", 7.0, 1, 0)
    case_scalar("scalar_axis_b", 7.0, 0, 1)
    case_scalar("scalar_axis_both", 7.0, 1, 1)
    case_scalar("scalar_u32_a", 7.0, 4294967295, 0)
    case_scalar("scalar_u32_b", 7.0, 0, 4294967295)
    ownership_created(Core.construct([], [7.0]))
