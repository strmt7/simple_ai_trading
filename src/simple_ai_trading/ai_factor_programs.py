"""Auditable, non-executable expression programs for AI-proposed factors."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from typing import Mapping, Sequence

import numpy as np


FACTOR_PROGRAM_SCHEMA_VERSION = "ai-factor-programs-v1"
ALLOWED_FUNCTIONS = (
    "abs",
    "clip",
    "maximum",
    "minimum",
    "safe_divide",
    "sign",
    "tanh",
)
_PROGRAM_FIELDS = {
    "name",
    "expression",
    "mechanism",
    "failure_mode",
    "expected_horizon",
}
_ACTION_PROGRAM_FIELDS = _PROGRAM_FIELDS | {"action_symmetry"}
_FUNCTION_ARITY = {
    "abs": 1,
    "clip": 3,
    "maximum": 2,
    "minimum": 2,
    "safe_divide": 2,
    "sign": 1,
    "tanh": 1,
}
_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
_MAX_AST_NODES = 128
_MAX_AST_DEPTH = 24
_MAX_TEXT_LENGTH = 1_000
_MAX_CONSTANT_ABS = 10_000.0


@dataclass(frozen=True)
class FactorProgram:
    model: str
    name: str
    expression: str
    canonical_expression: str
    mechanism: str
    failure_mode: str
    expected_horizon: str
    program_sha256: str

    def asdict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ActionConditionedFactorProgram:
    """A strict factor program with an auditable long/short symmetry statement."""

    model: str
    name: str
    expression: str
    canonical_expression: str
    mechanism: str
    failure_mode: str
    expected_horizon: str
    action_symmetry: str
    base_program_sha256: str
    program_sha256: str

    def asdict(self) -> dict[str, str]:
        return asdict(self)

    def as_factor_program(self) -> FactorProgram:
        return FactorProgram(
            model=self.model,
            name=self.name,
            expression=self.expression,
            canonical_expression=self.canonical_expression,
            mechanism=self.mechanism,
            failure_mode=self.failure_mode,
            expected_horizon=self.expected_horizon,
            program_sha256=self.base_program_sha256,
        )


@dataclass(frozen=True)
class FactorTransform:
    schema_version: str
    source_feature_names: tuple[str, ...]
    output_feature_names: tuple[str, ...]
    programs: tuple[FactorProgram, ...]
    lower_bounds: tuple[float, ...]
    upper_bounds: tuple[float, ...]
    training_rows: int
    transform_sha256: str

    def asdict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "source_feature_names": list(self.source_feature_names),
            "output_feature_names": list(self.output_feature_names),
            "programs": [program.asdict() for program in self.programs],
            "lower_bounds": list(self.lower_bounds),
            "upper_bounds": list(self.upper_bounds),
        }


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _validate_text(
    value: object,
    label: str,
    *,
    maximum: int,
    minimum: int = 1,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"factor {label} must be text")
    text = value.strip()
    if (
        len(text) < minimum
        or len(text) > maximum
        or any(ord(character) < 32 for character in text)
    ):
        raise ValueError(f"factor {label} is invalid")
    return text


def _numeric_literal(node: ast.AST) -> float | None:
    if isinstance(node, ast.Constant) and not isinstance(node.value, bool) and isinstance(
        node.value, (int, float)
    ):
        return float(node.value)
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.UAdd, ast.USub))
        and isinstance(node.operand, ast.Constant)
        and not isinstance(node.operand.value, bool)
        and isinstance(node.operand.value, (int, float))
    ):
        value = float(node.operand.value)
        return value if isinstance(node.op, ast.UAdd) else -value
    return None


def _validate_node(
    node: ast.AST,
    *,
    features: set[str],
    depth: int,
    count: list[int],
) -> None:
    count[0] += 1
    if count[0] > _MAX_AST_NODES or depth > _MAX_AST_DEPTH:
        raise ValueError("factor expression is too complex")
    if isinstance(node, ast.Expression):
        _validate_node(node.body, features=features, depth=depth + 1, count=count)
        return
    if isinstance(node, ast.Name):
        if node.id not in features:
            raise ValueError(f"factor uses an unknown feature: {node.id}")
        return
    if isinstance(node, ast.Constant):
        if (
            isinstance(node.value, bool)
            or not isinstance(node.value, (int, float))
            or not math.isfinite(float(node.value))
            or abs(float(node.value)) > _MAX_CONSTANT_ABS
        ):
            raise ValueError("factor constant is invalid")
        return
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.UAdd, ast.USub)):
            raise ValueError("factor unary operator is forbidden")
        _validate_node(node.operand, features=features, depth=depth + 1, count=count)
        return
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
            raise ValueError("factor binary operator is forbidden")
        _validate_node(node.left, features=features, depth=depth + 1, count=count)
        _validate_node(node.right, features=features, depth=depth + 1, count=count)
        return
    if isinstance(node, ast.Call):
        if (
            not isinstance(node.func, ast.Name)
            or node.func.id not in _FUNCTION_ARITY
            or node.keywords
            or len(node.args) != _FUNCTION_ARITY[node.func.id]
        ):
            raise ValueError("factor function call is forbidden")
        if node.func.id == "clip":
            lower, upper = node.args[1:]
            lower_value = _numeric_literal(lower)
            upper_value = _numeric_literal(upper)
            if lower_value is None or upper_value is None:
                raise ValueError("factor clip bounds must be numeric constants")
            if lower_value >= upper_value:
                raise ValueError("factor clip bounds are not ordered")
        for argument in node.args:
            _validate_node(
                argument,
                features=features,
                depth=depth + 1,
                count=count,
            )
        return
    raise ValueError(f"factor syntax is forbidden: {type(node).__name__}")


def _feature_unit(name: str) -> str:
    return "bps" if name.endswith("_bps") else "dimensionless"


def _compatible_additive_unit(left: str, right: str) -> str:
    if left == right:
        return left
    if left == "scalar":
        return right
    if right == "scalar":
        return left
    raise ValueError(f"factor combines incompatible additive units: {left} and {right}")


def _infer_unit(node: ast.AST) -> str:
    if isinstance(node, ast.Expression):
        return _infer_unit(node.body)
    if isinstance(node, ast.Name):
        return _feature_unit(node.id)
    if isinstance(node, ast.Constant):
        return "scalar"
    if isinstance(node, ast.UnaryOp):
        return _infer_unit(node.operand)
    if isinstance(node, ast.BinOp):
        left = _infer_unit(node.left)
        right = _infer_unit(node.right)
        if isinstance(node.op, (ast.Add, ast.Sub)):
            return _compatible_additive_unit(left, right)
        if left in {"scalar", "dimensionless"}:
            return right
        if right in {"scalar", "dimensionless"}:
            return left
        return f"product({left},{right})"
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise RuntimeError("validated factor AST changed during unit inference")
    function = node.func.id
    units = [_infer_unit(argument) for argument in node.args]
    if function in {"abs", "clip"}:
        return units[0]
    if function in {"maximum", "minimum"}:
        return _compatible_additive_unit(units[0], units[1])
    if function == "safe_divide":
        if units[0] == units[1]:
            return "dimensionless"
        if units[1] in {"scalar", "dimensionless"}:
            return units[0]
        return f"ratio({units[0]},{units[1]})"
    if function in {"sign", "tanh"}:
        return "dimensionless"
    raise RuntimeError("validated factor function changed during unit inference")


def validate_factor_program(
    value: Mapping[str, object],
    *,
    model: str,
    feature_names: Sequence[str],
) -> FactorProgram:
    """Validate one strict mapping and return its canonical AST identity."""

    if set(value) != _PROGRAM_FIELDS:
        raise ValueError("factor program fields differ from the strict schema")
    model_name = _validate_text(model, "model", maximum=128)
    name = _validate_text(value["name"], "name", maximum=64)
    if not _NAME_PATTERN.fullmatch(name):
        raise ValueError("factor name must be lower snake case")
    expression = _validate_text(
        value["expression"], "expression", maximum=_MAX_TEXT_LENGTH
    )
    mechanism = _validate_text(
        value["mechanism"], "mechanism", maximum=800, minimum=20
    )
    failure_mode = _validate_text(
        value["failure_mode"], "failure_mode", maximum=800, minimum=20
    )
    expected_horizon = _validate_text(
        value["expected_horizon"], "expected_horizon", maximum=80
    )
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError) as exc:
        raise ValueError("factor expression cannot be parsed") from exc
    _validate_node(
        tree,
        features=set(feature_names),
        depth=0,
        count=[0],
    )
    _infer_unit(tree)
    referenced_features = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    descriptive_text = f"{name} {mechanism}".lower()
    for symbol in ("btcusdt", "ethusdt", "solusdt"):
        if symbol in descriptive_text and not any(
            feature.startswith(f"{symbol}_") for feature in referenced_features
        ):
            raise ValueError(
                f"factor description names {symbol} without a matching feature"
            )
    canonical_expression = ast.unparse(tree).strip()
    identity = {
        "model": model_name,
        "name": name,
        "canonical_expression": canonical_expression,
        "mechanism": mechanism,
        "failure_mode": failure_mode,
        "expected_horizon": expected_horizon,
    }
    return FactorProgram(
        model=model_name,
        name=name,
        expression=expression,
        canonical_expression=canonical_expression,
        mechanism=mechanism,
        failure_mode=failure_mode,
        expected_horizon=expected_horizon,
        program_sha256=_sha256(identity),
    )


def validate_action_conditioned_factor_program(
    value: Mapping[str, object],
    *,
    model: str,
    feature_names: Sequence[str],
) -> ActionConditionedFactorProgram:
    """Validate an action-aligned program without relaxing the base AST contract."""

    if set(value) != _ACTION_PROGRAM_FIELDS:
        raise ValueError("action-conditioned factor fields differ from the strict schema")
    base = validate_factor_program(
        {field: value[field] for field in _PROGRAM_FIELDS},
        model=model,
        feature_names=feature_names,
    )
    action_symmetry = _validate_text(
        value["action_symmetry"],
        "action_symmetry",
        maximum=800,
        minimum=20,
    )
    normalized = action_symmetry.lower()
    if "long" not in normalized or "short" not in normalized:
        raise ValueError("factor action_symmetry must explain both long and short rows")
    identity = {
        **base.asdict(),
        "action_symmetry": action_symmetry,
    }
    identity.pop("program_sha256")
    return ActionConditionedFactorProgram(
        model=base.model,
        name=base.name,
        expression=base.expression,
        canonical_expression=base.canonical_expression,
        mechanism=base.mechanism,
        failure_mode=base.failure_mode,
        expected_horizon=base.expected_horizon,
        action_symmetry=action_symmetry,
        base_program_sha256=base.program_sha256,
        program_sha256=_sha256(identity),
    )


def parse_factor_response(
    response_text: str,
    *,
    model: str,
    feature_names: Sequence[str],
    maximum_factors: int,
) -> tuple[FactorProgram, ...]:
    """Parse exact JSON. Markdown fences and free-text repair are intentionally absent."""

    programs, rejections = parse_factor_response_ledger(
        response_text,
        model=model,
        feature_names=feature_names,
        maximum_factors=maximum_factors,
    )
    if rejections:
        raise ValueError(rejections[0]["reason"])
    return programs


def parse_factor_response_ledger(
    response_text: str,
    *,
    model: str,
    feature_names: Sequence[str],
    maximum_factors: int,
) -> tuple[tuple[FactorProgram, ...], tuple[dict[str, object], ...]]:
    """Validate strict JSON while preserving per-program rejection evidence."""

    if not 1 <= int(maximum_factors) <= 32:
        raise ValueError("maximum factor count is invalid")
    try:
        payload = json.loads(response_text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("factor response is not strict JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"factors"}:
        raise ValueError("factor response root differs from the strict schema")
    rows = payload["factors"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= maximum_factors:
        raise ValueError("factor response count is invalid")
    programs: list[FactorProgram] = []
    rejections: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            rejections.append(
                {"index": index, "name": None, "reason": "factor response item is not an object"}
            )
            continue
        try:
            programs.append(
                validate_factor_program(
                    row,
                    model=model,
                    feature_names=feature_names,
                )
            )
        except ValueError as exc:
            rejections.append(
                {
                    "index": index,
                    "name": str(row.get("name", "")) or None,
                    "reason": str(exc),
                }
            )
    return tuple(programs), tuple(rejections)


def parse_action_conditioned_factor_response_ledger(
    response_text: str,
    *,
    model: str,
    feature_names: Sequence[str],
    maximum_factors: int,
) -> tuple[
    tuple[ActionConditionedFactorProgram, ...],
    tuple[dict[str, object], ...],
]:
    """Parse strict action-conditioned JSON and retain every rejection reason."""

    if not 1 <= int(maximum_factors) <= 32:
        raise ValueError("maximum factor count is invalid")
    try:
        payload = json.loads(response_text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("factor response is not strict JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"factors"}:
        raise ValueError("factor response root differs from the strict schema")
    rows = payload["factors"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= maximum_factors:
        raise ValueError("factor response count is invalid")
    programs: list[ActionConditionedFactorProgram] = []
    rejections: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            rejections.append(
                {
                    "index": index,
                    "name": None,
                    "reason": "factor response item is not an object",
                }
            )
            continue
        try:
            programs.append(
                validate_action_conditioned_factor_program(
                    row,
                    model=model,
                    feature_names=feature_names,
                )
            )
        except ValueError as exc:
            rejections.append(
                {
                    "index": index,
                    "name": str(row.get("name", "")) or None,
                    "reason": str(exc),
                }
            )
    return tuple(programs), tuple(rejections)


def _evaluate_node(node: ast.AST, features: Mapping[str, np.ndarray]) -> np.ndarray:
    if isinstance(node, ast.Expression):
        return _evaluate_node(node.body, features)
    if isinstance(node, ast.Name):
        return np.asarray(features[node.id], dtype=np.float64)
    if isinstance(node, ast.Constant):
        first = next(iter(features.values()))
        return np.full(np.asarray(first).shape, float(node.value), dtype=np.float64)
    if isinstance(node, ast.UnaryOp):
        value = _evaluate_node(node.operand, features)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = _evaluate_node(node.left, features)
        right = _evaluate_node(node.right, features)
        with np.errstate(over="ignore", invalid="ignore"):
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            return left * right
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise RuntimeError("validated factor AST changed")
    arguments = [_evaluate_node(argument, features) for argument in node.args]
    function = node.func.id
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if function == "abs":
            return np.abs(arguments[0])
        if function == "clip":
            return np.clip(arguments[0], arguments[1], arguments[2])
        if function == "maximum":
            return np.maximum(arguments[0], arguments[1])
        if function == "minimum":
            return np.minimum(arguments[0], arguments[1])
        if function == "safe_divide":
            output = np.zeros_like(arguments[0], dtype=np.float64)
            denominator = arguments[1]
            np.divide(
                arguments[0],
                denominator,
                out=output,
                where=np.abs(denominator) >= 1e-6,
            )
            return output
        if function == "sign":
            return np.sign(arguments[0])
        if function == "tanh":
            return np.tanh(arguments[0])
    raise RuntimeError("validated factor function changed")


def evaluate_factor_program(
    program: FactorProgram,
    features: np.ndarray,
    feature_names: Sequence[str],
) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[1] != len(feature_names)
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("factor feature matrix is invalid")
    tree = ast.parse(program.canonical_expression, mode="eval")
    columns = {name: matrix[:, index] for index, name in enumerate(feature_names)}
    values = np.asarray(_evaluate_node(tree, columns), dtype=np.float64)
    if values.shape != (matrix.shape[0],) or not np.isfinite(values).all():
        raise ValueError(f"factor produced nonfinite values: {program.name}")
    return values


def _output_name(program: FactorProgram) -> str:
    model = re.sub(r"[^a-z0-9]+", "_", program.model.lower()).strip("_")
    return f"ai_{model}_{program.name}"


def _transform_payload(transform: FactorTransform) -> dict[str, object]:
    payload = transform.asdict()
    payload.pop("transform_sha256")
    return payload


def fit_factor_transform(
    programs: Sequence[FactorProgram],
    features: np.ndarray,
    feature_names: Sequence[str],
    training_mask: np.ndarray,
) -> tuple[FactorTransform, np.ndarray, tuple[dict[str, str], ...]]:
    """Fit training-only clipping bounds and include every valid unique program."""

    matrix = np.asarray(features, dtype=np.float64)
    training = np.asarray(training_mask, dtype=bool)
    names = tuple(str(name) for name in feature_names)
    if (
        matrix.ndim != 2
        or matrix.shape[1] != len(names)
        or training.shape != (matrix.shape[0],)
        or np.count_nonzero(training) < 1_024
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("factor transform source is invalid")

    accepted: list[FactorProgram] = []
    output_names: list[str] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []
    columns: list[np.ndarray] = []
    rejections: list[dict[str, str]] = []
    seen_expressions: set[str] = set()
    seen_names: set[str] = set(names)
    for program in programs:
        output_name = _output_name(program)
        if program.canonical_expression in seen_expressions:
            rejections.append(
                {
                    "model": program.model,
                    "name": program.name,
                    "program_sha256": program.program_sha256,
                    "reason": "duplicate_canonical_expression",
                }
            )
            continue
        if output_name in seen_names:
            rejections.append(
                {
                    "model": program.model,
                    "name": program.name,
                    "program_sha256": program.program_sha256,
                    "reason": "duplicate_output_feature_name",
                }
            )
            continue
        try:
            values = evaluate_factor_program(program, matrix, names)
        except ValueError as exc:
            rejections.append(
                {
                    "model": program.model,
                    "name": program.name,
                    "program_sha256": program.program_sha256,
                    "reason": str(exc),
                }
            )
            continue
        selected = values[training]
        lower, upper = np.quantile(selected, (0.005, 0.995))
        if (
            not math.isfinite(float(lower))
            or not math.isfinite(float(upper))
            or upper - lower <= 1e-12
        ):
            rejections.append(
                {
                    "model": program.model,
                    "name": program.name,
                    "program_sha256": program.program_sha256,
                    "reason": "degenerate_training_distribution",
                }
            )
            continue
        accepted.append(program)
        output_names.append(output_name)
        lower_bounds.append(float(lower))
        upper_bounds.append(float(upper))
        columns.append(np.clip(values, lower, upper).astype(np.float32))
        seen_expressions.add(program.canonical_expression)
        seen_names.add(output_name)
    if not accepted:
        raise ValueError("no AI factor program survived the safety transform")
    factor_matrix = np.column_stack(columns).astype(np.float32)
    provisional = FactorTransform(
        schema_version=FACTOR_PROGRAM_SCHEMA_VERSION,
        source_feature_names=names,
        output_feature_names=tuple(output_names),
        programs=tuple(accepted),
        lower_bounds=tuple(lower_bounds),
        upper_bounds=tuple(upper_bounds),
        training_rows=int(np.count_nonzero(training)),
        transform_sha256="",
    )
    transform = FactorTransform(
        **{
            **provisional.__dict__,
            "transform_sha256": _sha256(_transform_payload(provisional)),
        }
    )
    return transform, factor_matrix, tuple(rejections)


def apply_factor_transform(
    transform: FactorTransform,
    features: np.ndarray,
    feature_names: Sequence[str],
) -> np.ndarray:
    if (
        transform.schema_version != FACTOR_PROGRAM_SCHEMA_VERSION
        or tuple(feature_names) != transform.source_feature_names
        or transform.transform_sha256 != _sha256(_transform_payload(transform))
        or len(transform.programs) != len(transform.output_feature_names)
        or len(transform.programs) != len(transform.lower_bounds)
        or len(transform.programs) != len(transform.upper_bounds)
    ):
        raise ValueError("AI factor transform identity is invalid")
    columns = []
    for program, lower, upper in zip(
        transform.programs,
        transform.lower_bounds,
        transform.upper_bounds,
        strict=True,
    ):
        values = evaluate_factor_program(program, features, feature_names)
        columns.append(np.clip(values, lower, upper).astype(np.float32))
    return np.column_stack(columns).astype(np.float32)


__all__ = [
    "ALLOWED_FUNCTIONS",
    "FACTOR_PROGRAM_SCHEMA_VERSION",
    "ActionConditionedFactorProgram",
    "FactorProgram",
    "FactorTransform",
    "apply_factor_transform",
    "evaluate_factor_program",
    "fit_factor_transform",
    "parse_action_conditioned_factor_response_ledger",
    "parse_factor_response",
    "parse_factor_response_ledger",
    "validate_action_conditioned_factor_program",
    "validate_factor_program",
]
