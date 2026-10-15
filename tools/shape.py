import itertools
import json
import math
import sys
from pathlib import Path

from tools.common import (
    ROOT,
    Evidence,
    GateError,
    bend,
    check_laws,
    check_sources,
    native,
    required_exports,
)

CAPACITY = 1 << 20
MAX_U32 = (1 << 32) - 1


def shape_oracle(dimensions: list[int]) -> dict:
    if any(value < 0 or value > MAX_U32 for value in dimensions):
        return {"transport_error": "U32Range"}
    if len(dimensions) > 4:
        return {"error": 1}
    count = math.prod(dimensions)
    if count > CAPACITY:
        return {"error": 2}
    return {"dimensions": dimensions, "rank": len(dimensions), "count": count}


def flatten_oracle(dimensions: list[int], coordinates: list[int]) -> dict:
    if len(dimensions) != len(coordinates):
        return {"error": 3}
    if any(
        value < 0 or value >= size
        for value, size in zip(coordinates, dimensions, strict=True)
    ):
        return {"error": 4}
    index = 0
    for size, value in zip(dimensions, coordinates, strict=True):
        index = index * size + value
    return {"value": index}


def unflatten_oracle(dimensions: list[int], index: int) -> dict:
    if index < 0 or index >= math.prod(dimensions):
        return {"error": 5}
    values = []
    for size in reversed(dimensions):
        index, value = divmod(index, size)
        values.append(value)
    return {"value": list(reversed(values))}


def broadcast_oracle(left: list[int], right: list[int]) -> dict:
    for dimensions in (left, right):
        result = shape_oracle(dimensions)
        if "error" in result or "transport_error" in result:
            return result
    output = []
    for a, b in itertools.zip_longest(reversed(left), reversed(right), fillvalue=1):
        if a == b:
            output.append(a)
        elif a == 1:
            output.append(b)
        elif b == 1:
            output.append(a)
        else:
            return {"error": 6}
    return shape_oracle(list(reversed(output)))


def projection_oracle(dimensions: list[int], coordinates: list[int]) -> list[int]:
    aligned = coordinates[len(coordinates) - len(dimensions) :]
    return [
        0 if size == 1 else value
        for size, value in zip(dimensions, aligned, strict=True)
    ]


def flat_projection_oracle(
    left: list[int], right: list[int], output: list[int], index: int
) -> str:
    coordinates = unflatten_oracle(output, index)
    if "error" in coordinates:
        return render("index", coordinates)
    return " ".join(
        render(
            "index",
            flatten_oracle(side, projection_oracle(side, coordinates["value"])),
        )
        for side in (left, right)
    )


def all_coordinates(dimensions: list[int]) -> list[list[int]]:
    return [
        list(values)
        for values in itertools.product(*(range(size) for size in dimensions))
    ]


def boundary_coordinates(dimensions: list[int]) -> list[list[int]]:
    zero = [0] * len(dimensions)
    result = [zero]
    if math.prod(dimensions):
        result.append([size - 1 for size in dimensions])
    result.append([0] * (len(dimensions) + 1))
    if dimensions:
        result.append([0] * (len(dimensions) - 1))
        for axis, size in enumerate(dimensions):
            values = zero.copy()
            values[axis] = size
            result.append(values)
    return result


def render(tag: str, result: dict) -> str:
    if "error" in result:
        return f"error:{result['error']}"
    if tag == "index":
        return f"index:{result['value']}"
    if tag == "coordinate":
        return "coordinate:" + ",".join(map(str, result["value"]))
    dimensions = ",".join(map(str, result["dimensions"]))
    return f"{tag}:{dimensions}:{result['rank']}:{result['count']}"


def expected_lines(request: dict) -> list[str]:
    if request["action"] == "shape":
        dimensions = request["dimensions"]
        result = shape_oracle(dimensions)
        if "error" in result:
            return [render("shape", result)]
        return [
            render("shape", result),
            *(
                render("index", flatten_oracle(dimensions, coordinates))
                for coordinates in request["coordinates"]
            ),
            *(
                render("coordinate", unflatten_oracle(dimensions, index))
                for index in request["indices"]
            ),
        ]
    left, right = request["left"], request["right"]
    result = broadcast_oracle(left, right)
    if "error" in result:
        return [render("broadcast", result)]
    lines = [render("broadcast", result)]
    for coordinates in request["coordinates"]:
        valid = flatten_oracle(result["dimensions"], coordinates)
        projected = [
            render(
                "coordinate",
                valid
                if "error" in valid
                else {"value": projection_oracle(side, coordinates)},
            )
            for side in (left, right)
        ]
        lines.append(" ".join(projected))
    lines.extend(
        flat_projection_oracle(left, right, result["dimensions"], index)
        for index in request["indices"]
    )
    return lines


def encode(request: dict) -> str:
    def words(values: list[int]) -> str:
        if any(value < 0 or value > MAX_U32 for value in values):
            raise GateError(f"The request stream carries U32 values only: {values}")
        return "".join(f",{value}" for value in values)

    coordinates = "".join(";" + words(values) for values in request["coordinates"])
    if request["action"] == "shape":
        dimensions, indices = request["dimensions"], request["indices"]
        fields = ["s", words(dimensions), coordinates, words(indices)]
    else:
        fields = [
            "b",
            words(request["left"]),
            words(request["right"]),
            coordinates,
            words(request["indices"]),
        ]
    return "".join(" " + field for field in fields)


def stream(requests: list[dict]) -> str:
    return "".join("\n" + encode(request) for request in requests)


TRANSPORT_REJECTIONS = [
    {"action": "shape", "dimensions": [-1], "coordinates": [], "indices": []},
    {"action": "shape", "dimensions": [MAX_U32 + 1], "coordinates": [], "indices": []},
    {"action": "shape", "dimensions": [2], "coordinates": [[-1]], "indices": []},
    {"action": "shape", "dimensions": [2], "coordinates": [], "indices": [MAX_U32 + 1]},
]


def check_transport() -> int:
    for dimensions in ([-1], [MAX_U32 + 1]):
        if shape_oracle(dimensions) != {"transport_error": "U32Range"}:
            raise GateError(f"Oracle accepted non-U32 dimensions: {dimensions}")
    for request in TRANSPORT_REJECTIONS:
        try:
            encode(request)
        except GateError:
            continue
        raise GateError(f"The request stream accepted a non-U32 value: {request!r}")
    return len(TRANSPORT_REJECTIONS)


def exhaustive_requests() -> tuple[list[dict], dict]:
    shapes = [
        list(dimensions)
        for rank in range(4)
        for dimensions in itertools.product(range(5), repeat=rank)
    ] + [list(dimensions) for dimensions in itertools.product(range(3), repeat=4)]
    requests = []
    for dimensions in shapes:
        count = math.prod(dimensions)
        requests.append(
            {
                "action": "shape",
                "dimensions": dimensions,
                "coordinates": all_coordinates(dimensions)
                + boundary_coordinates(dimensions),
                "indices": [*range(count), count, MAX_U32],
            }
        )
    for left, right in itertools.product(shapes, repeat=2):
        result = broadcast_oracle(left, right)
        coordinates, indices = [], []
        if "error" not in result:
            coordinates = all_coordinates(result["dimensions"]) + boundary_coordinates(
                result["dimensions"]
            )
            indices = [*range(result["count"]), result["count"], MAX_U32]
        requests.append(
            {
                "action": "broadcast",
                "left": left,
                "right": right,
                "coordinates": coordinates,
                "indices": indices,
            }
        )

    adversarial = [
        [1, 1, 1, 1],
        [4, 3, 2, 1],
        [0, 4, 3, 2],
        [4, 3, 2, 0],
        [CAPACITY],
        [1024, 1024],
        [2, 512, 1024],
        [16, 16, 16, 256],
        [1, 1, 1, CAPACITY],
        [CAPACITY + 1],
        [1024, 1025],
        [65536, 65536],
        [MAX_U32, MAX_U32],
        [MAX_U32, MAX_U32, 1],
        [MAX_U32, MAX_U32, 0],
        [MAX_U32, MAX_U32, MAX_U32, 0],
        [MAX_U32, MAX_U32, 0, MAX_U32],
        [0, MAX_U32, MAX_U32, MAX_U32],
        [CAPACITY, 2, 0],
        [1, 1, 1, 1, 1],
        [MAX_U32, MAX_U32, MAX_U32, MAX_U32, 0],
    ]
    for dimensions in adversarial:
        metadata = shape_oracle(dimensions)
        indices = [0]
        if "error" not in metadata:
            count = metadata["count"]
            indices = [0, max(0, count - 1), count, MAX_U32]
        requests.append(
            {
                "action": "shape",
                "dimensions": dimensions,
                "coordinates": boundary_coordinates(dimensions),
                "indices": indices,
            }
        )
    extra_pairs = [
        ([0, 3], [1, 3]),
        ([1, 3], [0, 3]),
        ([0, 3], [2, 3]),
        ([2, 1, 3, 1], [1, 4, 1, 2]),
        ([1, 1, 1, 1], []),
        ([1024, 1], [1, 2048]),
        ([CAPACITY], [1]),
        ([MAX_U32, MAX_U32, 0], [1, 1, 1]),
        ([], [MAX_U32, MAX_U32, MAX_U32, 0]),
    ]
    for left, right in extra_pairs:
        result = broadcast_oracle(left, right)
        coordinates, indices = [], []
        if "error" not in result:
            coordinates = boundary_coordinates(result["dimensions"])
            count = result["count"]
            indices = [0, max(0, count - 1), count, MAX_U32]
        requests.append(
            {
                "action": "broadcast",
                "left": left,
                "right": right,
                "coordinates": coordinates,
                "indices": indices,
            }
        )
    return requests, {
        "exhaustive_shapes": len(shapes),
        "exhaustive_broadcast_pairs": len(shapes) ** 2,
        "adversarial_shapes": len(adversarial),
        "additional_broadcast_pairs": len(extra_pairs),
    }


def coverage(requests: list[dict]) -> dict:
    counts = {
        "requests": len(requests),
        "flatten_checks": 0,
        "unflatten_checks": 0,
        "projection_checks": 0,
        "flat_projection_checks": 0,
    }
    for request in requests:
        if request["action"] == "shape":
            if "error" not in shape_oracle(request["dimensions"]):
                counts["flatten_checks"] += len(request["coordinates"])
                counts["unflatten_checks"] += len(request["indices"])
        elif "error" not in broadcast_oracle(request["left"], request["right"]):
            counts["projection_checks"] += 2 * len(request["coordinates"])
            counts["flat_projection_checks"] += 2 * len(request["indices"])
    return counts


def verify_output(actual: str, requests: list[dict], lane: str) -> None:
    rows = actual.splitlines()
    expected = [
        (index, line)
        for index, request in enumerate(requests)
        for line in expected_lines(request)
    ]
    for position, (index, reference) in enumerate(expected):
        observed = rows[position] if position < len(rows) else None
        if observed != reference:
            raise GateError(
                f"{lane}: Bend/std-library mismatch at request {index}: "
                f"{requests[index]!r}; expected {reference!r}; observed {observed!r}"
            )
    if len(rows) != len(expected):
        raise GateError(
            f"{lane}: expected {len(expected)} observations, got {len(rows)}"
        )


def run_gate(tool_root: Path, output: Path) -> None:
    check_sources(tool_root)
    evidence = Evidence(output, tool_root)
    check_laws(
        evidence,
        "shape",
        ROOT / "src/shape/PROOF.bend",
        required_exports(ROOT / "verification/shape.json"),
    )
    evidence.run(
        "pytorch-absent",
        [
            sys.executable,
            "-c",
            "import importlib.util; "
            "assert importlib.util.find_spec('torch') is None; "
            "print('PyTorch is absent from the root execution environment')",
        ],
        expected_output="PyTorch is absent from the root execution environment\n",
    )
    requests, counts = exhaustive_requests()
    counts["transport_rejections"] = check_transport()
    path = output / "requests.txt"
    path.write_text(stream(requests))
    for lane in ("native", "javascript"):
        target = output / ("shapes" if lane == "native" else "shapes.js")
        evidence.run(
            f"shape-{lane}-build",
            bend(tool_root, ROOT / "src/shape/harness.bend", "-o", target),
        )
        command = (
            native(target, str(path))
            if lane == "native"
            else [bend(tool_root)[0], str(target), str(path)]
        )
        result = evidence.run(f"shape-{lane}-run", command)
        verify_output(result.stdout, requests, lane)
    counts.update(coverage(requests))
    (output / "coverage.json").write_text(json.dumps(counts, indent=2) + "\n")
    evidence.save()
    print(f"Shape proof, native, and JS checks passed: {output}")
