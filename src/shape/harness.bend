import Base
import ./core.bend as Shape
import ./flat.bend as Flat

# Tail recursive: the JS lane runs on the host call stack, and the
# request stream is megabytes long.
def split.go(bytes: List<&2, U32>, +separator: U32, +piece: List<&2, U32>, +pieces: List<&2, List<&2, U32>>) -> List<&2, List<&2, U32>>:
  match bytes:
    case Nil{}:
      piece <> pieces
    case +byte <> tail:
      +cut = U32.is_eq(byte, separator)
      split.go(tail, separator, Bool.pick(List<&2, U32>, cut, Nil{}, byte <> piece), Bool.pick(List<&2, List<&2, U32>>, cut, piece <> pieces, pieces))

def split.restore(reversed: List<&2, List<&2, U32>>, pieces: List<&2, List<&2, U32>>) -> List<&2, List<&2, U32>>:
  match reversed:
    case Nil{}:
      pieces
    case head <> tail:
      split.restore(tail, List.reverse(&2, U32, head) <> pieces)

# Every item, a line or field or number, is preceded by its separator, so an
# empty list and a list holding one empty item differ.
def items.drop(pieces: List<&2, List<&2, U32>>) -> List<&2, List<&2, U32>>:
  match pieces:
    case Nil{}:
      Nil{}
    case head <> tail:
      tail

def items(bytes: List<&2, U32>, +separator: U32) -> List<&2, List<&2, U32>>:
  items.drop(split.restore(split.go(bytes, separator, Nil{}, Nil{}), Nil{}))

def number(digits: List<&2, U32>, +value: U32) -> U32:
  match digits:
    case Nil{}:
      value
    case digit <> tail:
      number(tail, U32.add(U32.mul(value, 10), U32.sub(digit, 48)))

def numbers(pieces: List<&2, List<&2, U32>>) -> List<&2, U32>:
  match pieces:
    case Nil{}:
      Nil{}
    case head <> tail:
      number(head, 0) <> numbers(tail)

def words(bytes: List<&2, U32>) -> List<&2, U32>:
  numbers(items(bytes, 44))

def coordinates.go(pieces: List<&2, List<&2, U32>>) -> List<&2, List<&2, U32>>:
  match pieces:
    case Nil{}:
      Nil{}
    case head <> tail:
      words(head) <> coordinates.go(tail)

def coordinates(bytes: List<&2, U32>) -> List<&2, List<&2, U32>>:
  coordinates.go(items(bytes, 59))

def shown(values: List<&2, U32>) -> List<&2, String>:
  match values:
    case Nil{}:
      Nil{}
    case head <> tail:
      U32.show(head) <> shown(tail)

def show_words(values: List<&2, U32>) -> String:
  String.join(shown(values), ",")

def error_text(error: Shape.Error) -> String:
  "error:" ++ U32.show(Shape.error_code(error))

def metadata(tag: String, +shape: Shape.Shape) -> String:
  String.concat([tag, ":", show_words(Shape.dimensions(shape)), ":", U32.show(Shape.rank(shape)), ":", U32.show(Shape.count(shape))])

def index_text(-shape: Shape.Shape, result: Result<&2, &2, Shape.Error, Shape.FlatIndex<shape>>) -> String:
  match result:
    case Fail{error}:
      error_text(error)
    case Done{index}:
      "index:" ++ U32.show(Shape.index_value(shape, index))

def coordinate_text(-shape: Shape.Shape, result: Result<&2, &2, Shape.Error, Shape.CheckedCoordinate<shape>>) -> String:
  match result:
    case Fail{error}:
      error_text(error)
    case Done{coordinate}:
      "coordinate:" ++ show_words(Shape.coordinate_values(shape, coordinate))

def flatten_lines(+shape: Shape.Shape, values: List<&2, List<&2, U32>>, rest: List<&2, String>) -> List<&2, String>:
  match values:
    case Nil{}:
      rest
    case head <> tail:
      index_text(shape, Shape.flatten_checked(shape, head)) <> flatten_lines(shape, tail, rest)

def unflatten_lines(+shape: Shape.Shape, indices: List<&2, U32>) -> List<&2, String>:
  match indices:
    case Nil{}:
      Nil{}
    case head <> tail:
      coordinate_text(shape, Shape.unflatten_checked(shape, head)) <> unflatten_lines(shape, tail)

def shape_lines(-input: List<&2, U32>, result: Shape.ConstructResult<input>, values: List<&2, List<&2, U32>>, indices: List<&2, U32>) -> List<&2, String>:
  match result:
    case Shape.SError{error}:
      [error_text(error)]
    case Shape.SBuilt{+shape, preserved}:
      metadata("shape", shape) <> flatten_lines(shape, values, unflatten_lines(shape, indices))

def projection_lines(+plan: Shape.BroadcastPlan, values: List<&2, List<&2, U32>>, rest: List<&2, String>) -> List<&2, String>:
  match values:
    case Nil{}:
      rest
    case +head <> tail:
      String.concat([coordinate_text(Shape.left_shape(plan), Shape.project_left_checked(plan, head)), " ", coordinate_text(Shape.right_shape(plan), Shape.project_right_checked(plan, head))]) <> projection_lines(plan, tail, rest)

def flat_text(+plan: Shape.BroadcastPlan, +map: Shape.Index<Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan))>, result: Result<&2, &2, Shape.Error, Shape.FlatIndex<Shape.output_shape(plan)>>) -> String:
  match result:
    case Fail{error}:
      error_text(error)
    case Done{+index}:
      String.concat(["index:", U32.show(Shape.index_value(Shape.left_shape(plan), Flat.project_left_index(plan, map, index))), " index:", U32.show(Shape.index_value(Shape.right_shape(plan), Flat.project_right_index(plan, map, index)))])

def flat_lines(+plan: Shape.BroadcastPlan, +map: Shape.Index<Shape.extents(Shape.left_shape(plan)), Shape.extents(Shape.right_shape(plan)), Shape.extents(Shape.output_shape(plan))>, indices: List<&2, U32>) -> List<&2, String>:
  match indices:
    case Nil{}:
      Nil{}
    case head <> tail:
      flat_text(plan, map, Shape.flat_checked(Shape.output_shape(plan), U32.to_nat(head))) <> flat_lines(plan, map, tail)

def broadcast_lines.plan(result: Result<&2, &2, Shape.Error, Shape.BroadcastPlan>, values: List<&2, List<&2, U32>>, indices: List<&2, U32>) -> List<&2, String>:
  match result:
    case Fail{error}:
      [error_text(error)]
    case Done{+plan}:
      metadata("broadcast", Shape.output_shape(plan)) <> projection_lines(plan, values, flat_lines(plan, Shape.index_map(plan), indices))

def broadcast_lines.right(left: Shape.Shape, -input: List<&2, U32>, result: Shape.ConstructResult<input>, values: List<&2, List<&2, U32>>, indices: List<&2, U32>) -> List<&2, String>:
  match result:
    case Shape.SError{error}:
      [error_text(error)]
    case Shape.SBuilt{right, preserved}:
      broadcast_lines.plan(Shape.broadcast_checked(left, right), values, indices)

def broadcast_lines.left(-input: List<&2, U32>, result: Shape.ConstructResult<input>, +right: List<&2, U32>, values: List<&2, List<&2, U32>>, indices: List<&2, U32>) -> List<&2, String>:
  match result:
    case Shape.SError{error}:
      [error_text(error)]
    case Shape.SBuilt{left, preserved}:
      broadcast_lines.right(left, right, Shape.construct(right), values, indices)

def request.shape(shape: Bool, first: List<&2, U32>, second: List<&2, U32>, third: List<&2, U32>) -> List<&2, String>:
  match shape:
    case True{}:
      +dimensions = words(first)
      shape_lines(dimensions, Shape.construct(dimensions), coordinates(second), words(third))
    case False{}:
      ["malformed"]

def request.broadcast(shape: Bool, first: List<&2, U32>, second: List<&2, U32>, third: List<&2, U32>, fourth: List<&2, U32>) -> List<&2, String>:
  match shape:
    case True{}:
      ["malformed"]
    case False{}:
      +left = words(first)
      broadcast_lines.left(left, Shape.construct(left), words(second), coordinates(third), words(fourth))

def is_shape(tag: List<&2, U32>) -> Bool:
  match tag:
    case Nil{}:
      False{}
    case head <> tail:
      U32.is_eq(head, 115)

def request(fields: List<&2, List<&2, U32>>) -> List<&2, String>:
  match fields:
    case tag <> first <> second <> third <> Nil{}:
      request.shape(is_shape(tag), first, second, third)
    case tag <> first <> second <> third <> fourth <> Nil{}:
      request.broadcast(is_shape(tag), first, second, third, fourth)
    case fields:
      ["malformed"]

def run(lines: List<&2, List<&2, U32>>) -> IO(Unit):
  match lines:
    case Nil{}:
      IO.pure(Unit, Unit{})
    case head <> tail:
      do IO<Unit>:
        IO.print(String.join(request(items(head, 32)), "\n"))
        run(tail)

def read(pair: File & Result<&1, &1, U32 & String, List<&2, U32>>) -> IO(Unit):
  (file, result) = pair
  do IO<Unit>:
    bytes : List<&2, U32> <- IO.pass(List<&2, U32>, result)
    File.close(file)
    run(items(bytes, 10))

def sized(pair: File & Result<&1, &1, U32 & String, U32>) -> IO(Unit):
  (file, result) = pair
  do IO<Unit>:
    size : U32 <- IO.pass(U32, result)
    read_result : File & Result<&1, &1, U32 & String, List<&2, U32>> <- File.read_bytes(file, size)
    read(read_result)

def open(arguments: List<String>) -> IO(Unit):
  match arguments:
    case program <> path <> Nil{}:
      do IO<Unit>:
        file : File <- IO.try(File, File.open(path, "r"))
        size : File & Result<&1, &1, U32 & String, U32> <- File.size(file)
        sized(size)
    case arguments:
      IO.die(Unit, 2, "usage: harness <requests>")

def main() -> IO(Unit):
  do IO<Unit>:
    arguments : List<String> <- IO.args()
    open(arguments)
