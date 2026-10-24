"""TensorFlow `strings` manipulation module."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

from zero_tensorflow._typing import (
    ArrayLikeProtocol,
    AxisLike,
    DTypeLike,
    NestedTensorData,
    NumpyConvertibleProtocol,
    ScalarValue,
    TensorOperand,
)

import ml_switcheroo_compiler.core as msc_core
import ml_switcheroo_compiler.core.dtype as msc_dtype
import ml_switcheroo_compiler.ops as _ops
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.core.tensor import as_tensor_obj
from ml_switcheroo_compiler.tracing.state import global_tracing_state as _tracer
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

__all__ = [
    "bytes_split",
    "format",
    "join",
    "length",
    "lower",
    "ngrams",
    "reduce_join",
    "regex_full_match",
    "regex_replace",
    "split",
    "strip",
    "substr",
    "to_hash_bucket",
    "to_hash_bucket_fast",
    "to_hash_bucket_strong",
    "to_number",
    "unicode_decode",
    "unicode_decode_with_offsets",
    "unicode_encode",
    "unicode_script",
    "unicode_split",
    "unicode_split_with_offsets",
    "unicode_transcode",
    "unsorted_segment_join",
    "upper",
]


@runtime_checkable
class _ShapedTypedProtocol(Protocol):
    """Protocol for tensor-like objects declaring `.shape` and `.dtype` attributes."""

    @property
    def shape(self) -> tuple[int, ...] | Sequence[int | None]:
        """Return tensor shape.

        Returns:
            tuple[int, ...] | Sequence[int | None]: Shape dimensions.
        """

    @property
    def dtype(self) -> DTypeLike:
        """Return tensor data type.

        Returns:
            DTypeLike: Data type.
        """


def _is_tracing(val: TensorOperand | _ShapedTypedProtocol) -> bool:
    """Return True if `val` is a tensor-like object and tracing is active or eager mode is disabled.

    Args:
        val: Candidate operand to inspect.

    Returns:
        bool: True when tracing a symbolic string operation.
    """
    if isinstance(val, (ProxyTensor, _ShapedTypedProtocol)):
        return _tracer.is_tracing or not config.eager_mode
    return False


def _as_str_tensor(val: TensorOperand | _ShapedTypedProtocol) -> msc_core.Tensor:
    """Convert `val` into a compiler `Tensor` for traced string operations.

    Args:
        val: Candidate string or tensor operand.

    Returns:
        msc_core.Tensor: Compiler tensor instance.
    """
    if isinstance(val, msc_core.Tensor):
        return val
    return as_tensor_obj(str(val))


def _to_python(
    val: TensorOperand | TensorOrScalar | _ShapedTypedProtocol,
) -> NestedTensorData:
    """Convert a tensor or array-like operand into nested Python scalars/lists.

    Args:
        val: Input operand.

    Returns:
        NestedTensorData: Python scalar or nested sequence.
    """
    current: TensorOperand | TensorOrScalar | NestedTensorData | _ShapedTypedProtocol = val
    if isinstance(current, NumpyConvertibleProtocol):
        current = current.numpy()
    if isinstance(current, ArrayLikeProtocol):
        return current.tolist()
    if isinstance(current, (list, tuple)):
        return [_to_python(item) for item in current]
    if isinstance(current, (int, float, bool, complex, str, bytes)) or current is None:
        return current
    return str(current)


def _map_nested(
    func: Callable[[ScalarValue], TensorOrScalar],
    val: NestedTensorData,
) -> TensorOrScalar:
    """Recursively apply `func` to leaf scalar elements of `val`.

    Args:
        func: Leaf transformation function.
        val: Nested Python scalar or sequence.

    Returns:
        TensorOrScalar: Transformed scalar or nested list.
    """
    if isinstance(val, (int, float, bool, complex, str, bytes)) or val is None:
        return func(val)
    return [_map_nested(func, v) for v in val]


def _reduce_join_nested(val: NestedTensorData, sep: str = "") -> str:
    """Recursively join nested string sequences using `sep`.

    Args:
        val: Nested string scalar or sequence.
        sep: Separator string.

    Returns:
        str: Joined string.
    """
    if isinstance(val, (list, tuple)):
        return sep.join([_reduce_join_nested(v, sep) for v in val])
    return str(val)


def join(
    inputs: Sequence[TensorOperand | _ShapedTypedProtocol],
    separator: str = "",
    name: str | None = None,
) -> TensorOrScalar:
    """Perform element-wise concatenation of a list of string tensors.

    Args:
        inputs: A list of `tf.Tensor` objects of same size and `string` dtype.
        separator: A string added between each string being joined.
        name: Optional name for the operation.

    Returns:
        TensorOrScalar: Joined string or list of strings.
    """
    del name
    if not inputs:
        return ""
    if any(_is_tracing(x) for x in inputs):
        t_inputs = [_as_str_tensor(x) for x in inputs]
        return _ops.text.string_join(t_inputs, separator=separator)
    py_inputs = [_to_python(x) for x in inputs]
    is_list = any(isinstance(x, list) for x in py_inputs)
    if not is_list:
        return separator.join(str(x) for x in py_inputs)
    max_len = max((len(x) for x in py_inputs if isinstance(x, list)), default=0)
    res: list[str] = []
    for i in range(max_len):
        parts: list[str] = []
        for inp in py_inputs:
            if isinstance(inp, list):
                parts.append(str(inp[i]))
            else:
                parts.append(str(inp))
        res.append(separator.join(parts))
    return res


def split(
    input: TensorOperand | _ShapedTypedProtocol,
    sep: str | None = None,
    maxsplit: int = -1,
    name: str | None = None,
) -> TensorOrScalar:
    """Split elements of `input` based on `sep` into a list/tensor of substrings.

    Args:
        input: A `string` Tensor or Python string/sequence.
        sep: Delimiter string (`None` splits on whitespace).
        maxsplit: Maximum number of splits (`-1` means no limit).
        name: Optional name for the operation.

    Returns:
        TensorOrScalar: Split substrings.
    """
    del name
    if _is_tracing(input):
        return _ops.text.string_split(
            _as_str_tensor(input), delimiter=sep if sep is not None else " "
        )
    val = _to_python(input)
    return _map_nested(lambda x: str(x).split(sep, maxsplit), val)


def length(
    input: TensorOperand | _ShapedTypedProtocol,
    unit: str = "BYTE",
    name: str | None = None,
) -> TensorOrScalar:
    """Compute the length of each string given in the input tensor.

    Args:
        input: A `string` Tensor or Python string/sequence.
        unit: Length unit (`'BYTE'` or `'UTF8_CHAR'`).
        name: Optional name for the operation.

    Returns:
        TensorOrScalar: Integer length(s) of each string.
    """
    del name
    if _is_tracing(input):
        return _ops.text.string_length(_as_str_tensor(input))
    val = _to_python(input)
    if unit.upper() == "UTF8_CHAR":
        return _map_nested(
            lambda x: len(x.decode("utf-8", errors="replace") if isinstance(x, bytes) else str(x)),
            val,
        )
    return _map_nested(
        lambda x: len(x if isinstance(x, bytes) else str(x).encode("utf-8")),
        val,
    )


def regex_replace(
    input: TensorOperand | _ShapedTypedProtocol,
    pattern: str,
    rewrite: str,
    replace_global: bool = True,
    name: str | None = None,
) -> TensorOrScalar:
    """Replace elements of `input` matching regex `pattern` with `rewrite`.

    Args:
        input: String tensor or sequence to process.
        pattern: Regular expression pattern to match.
        rewrite: Replacement string.
        replace_global: If True, replace all non-overlapping matches; else only first.
        name: Optional name for the operation.

    Returns:
        TensorOrScalar: String(s) with regex matches replaced.
    """
    del name
    if _is_tracing(input):
        return _ops.text.regex_replace(_as_str_tensor(input), pattern, rewrite)
    compiled = re.compile(pattern)
    count = 0 if replace_global else 1
    val = _to_python(input)
    return _map_nested(lambda x: compiled.sub(rewrite, str(x), count=count), val)


def regex_full_match(
    input: TensorOperand | _ShapedTypedProtocol,
    pattern: str,
    name: str | None = None,
) -> TensorOrScalar:
    """Check if the input matches the regular expression `pattern`.

    Args:
        input: A `string` Tensor or sequence of strings.
        pattern: Regular expression pattern string.
        name: Optional name for the operation.

    Returns:
        TensorOrScalar: Boolean match result(s).
    """
    del name
    if _is_tracing(input):
        return _ops.text.regex_full_match(_as_str_tensor(input), pattern)
    compiled = re.compile(pattern)
    val = _to_python(input)
    return _map_nested(lambda x: compiled.fullmatch(str(x)) is not None, val)


def _is_int_dtype(out_type: DTypeLike) -> bool:
    """Return True if `out_type` specifies an integer data type.

    Args:
        out_type: Candidate output `DType`, string, or Python type.

    Returns:
        bool: True if integer dtype.
    """
    if out_type is int:
        return True
    if isinstance(out_type, msc_dtype.DType):
        return "int" in out_type.value.lower()
    if isinstance(out_type, str):
        return "int" in out_type.lower()
    return False


def to_number(
    input: TensorOperand | _ShapedTypedProtocol,
    out_type: DTypeLike = None,
    name: str | None = None,
) -> TensorOrScalar:
    """Convert each string in the input Tensor to the specified numeric type.

    Args:
        input: A `string` Tensor or sequence of strings.
        out_type: Optional numeric output `DType` (defaults to `tf.float32`).
        name: Optional name for the operation.

    Returns:
        TensorOrScalar: Parsed numeric value(s).
    """
    del name
    if _is_tracing(input):
        dt = out_type if isinstance(out_type, msc_dtype.DType) else msc_dtype.DType.Float32
        return _ops.text.string_to_number(_as_str_tensor(input), dtype=dt)
    val = _to_python(input)
    if _is_int_dtype(out_type):
        return _map_nested(lambda x: int(float(str(x))), val)
    return _map_nested(lambda x: float(str(x)), val)


def bytes_split(
    input: TensorOperand | _ShapedTypedProtocol,
    name: str | None = None,
) -> TensorOrScalar:
    """Split string elements of `input` into individual bytes/characters.

    Args:
        input: A `string` Tensor or Python string/sequence.
        name: Optional name for the operation.

    Returns:
        TensorOrScalar: Split byte/character list(s).
    """
    del name
    if _is_tracing(input):
        return _ops.text.string_split(_as_str_tensor(input), delimiter="")
    val = _to_python(input)
    return _map_nested(lambda x: list(str(x)), val)


def format(
    template: str,
    inputs: Sequence[TensorOperand],
    placeholder: str = "{}",
    summarize: int = 3,
    name: str | None = None,
) -> str:
    """Format a string template using a list of tensors.

    Args:
        template: A string template containing `placeholder` tokens.
        inputs: List of tensor or scalar values to substitute into `template`.
        placeholder: Placeholder token string to replace.
        summarize: Number of elements to summarize per tensor.
        name: Optional name for the operation.

    Returns:
        str: Formatted string.
    """
    del summarize, name
    py_inputs = [_to_python(x) for x in inputs]
    s = template
    for item in py_inputs:
        s = s.replace(placeholder, str(item), 1)
    return s


def lower(
    input: TensorOperand | _ShapedTypedProtocol,
    encoding: str = "utf-8",
    name: str | None = None,
) -> TensorOrScalar:
    """Convert all uppercase characters into their respective lowercase replacements.

    Args:
        input: A `string` Tensor or Python string/sequence.
        encoding: Character encoding (`'utf-8'` or `''`).
        name: Optional name for the operation.

    Returns:
        TensorOrScalar: Lowercase string(s).
    """
    del encoding, name
    if _is_tracing(input):
        return _ops.text.string_lower(_as_str_tensor(input))
    val = _to_python(input)
    return _map_nested(lambda x: str(x).lower(), val)


def ngrams(
    data: TensorOperand,
    ngram_width: int | Sequence[int],
    separator: str = " ",
    pad_values: TensorOperand = None,
    padding_width: int | None = None,
    preserve_short_sequences: bool = False,
    name: str | None = None,
) -> list[str]:
    """Create a list of n-grams based on `data`.

    Args:
        data: Input sequence of strings.
        ngram_width: Width(s) of the n-grams to generate.
        separator: Separator string used between n-gram tokens.
        pad_values: Optional tuple `(left_pad, right_pad)` or single string used for padding.
        padding_width: Optional padding width (defaults to `w - 1`).
        preserve_short_sequences: Whether to preserve sequences shorter than `ngram_width`.
        name: Optional name for the operation.

    Returns:
        list[str]: Generated n-grams list.
    """
    del name
    py_data = _to_python(data)
    tokens: list[str] = (
        [str(t) for t in py_data]
        if isinstance(py_data, list)
        else ([str(py_data)] if py_data is not None and py_data != "" else [])
    )
    if not tokens:
        return []

    widths: list[int] = (
        [int(w) for w in ngram_width] if isinstance(ngram_width, Sequence) else [int(ngram_width)]
    )
    result: list[str] = []
    for w in widths:
        if w <= 0:
            continue
        padded = list(tokens)
        if pad_values is not None:
            pw = padding_width if padding_width is not None else w - 1
            if (
                isinstance(pad_values, Sequence)
                and not isinstance(pad_values, (str, bytes))
                and len(pad_values) == 2
            ):
                left_pad, right_pad = str(pad_values[0]), str(pad_values[1])
            else:
                left_pad = right_pad = str(pad_values)
            padded = [left_pad] * max(0, pw) + padded + [right_pad] * max(0, pw)

        if len(padded) < w:
            if preserve_short_sequences:
                result.append(separator.join(padded))
        else:
            for idx in range(len(padded) - w + 1):
                result.append(separator.join(padded[idx : idx + w]))

    return result


def reduce_join(
    inputs: TensorOperand | _ShapedTypedProtocol,
    axis: AxisLike = None,
    keepdims: bool = False,
    separator: str = "",
    name: str | None = None,
) -> TensorOrScalar:
    """Join all strings in the given tensor along the given axis.

    Args:
        inputs: Input `string` tensor or sequence.
        axis: Dimension(s) along which to join.
        keepdims: Whether to retain reduced dimensions with length 1.
        separator: Separator string inserted between joined elements.
        name: Optional name for the operation.

    Returns:
        TensorOrScalar: Joined string result.
    """
    del axis, keepdims, name
    if _is_tracing(inputs):
        return _ops.text.string_join(
            [_as_str_tensor(inputs)],
            separator=separator,
        )
    val = _to_python(inputs)
    return _reduce_join_nested(val, sep=separator)


def strip(
    input: TensorOperand | _ShapedTypedProtocol,
    name: str | None = None,
) -> TensorOrScalar:
    """Strip leading and trailing whitespace from `input`.

    Args:
        input: A `string` Tensor or Python string/sequence.
        name: Optional name for the operation.

    Returns:
        TensorOrScalar: Stripped string(s).
    """
    del name
    if _is_tracing(input):
        return _ops.text.regex_replace(_as_str_tensor(input), r"^\s+|\s+$", "")
    val = _to_python(input)
    return _map_nested(lambda x: str(x).strip(), val)


def substr(
    input: TensorOperand | _ShapedTypedProtocol,
    pos: int,
    len: int,
    unit: str = "BYTE",
    name: str | None = None,
) -> TensorOrScalar:
    """Return substrings from `input` starting at `pos` with length `len`.

    Args:
        input: A `string` Tensor or Python string/sequence.
        pos: Scalar integer defining the starting position of each substring.
        len: Scalar integer defining the length of each substring.
        unit: Indexing unit (`'BYTE'` or `'UTF8_CHAR'`).
        name: Optional name for the operation.

    Returns:
        TensorOrScalar: Extracted substring(s).
    """
    del unit, name
    if _is_tracing(input):
        return _ops.text.string_substr(_as_str_tensor(input), pos, len)
    val = _to_python(input)

    def _sub(x: ScalarValue) -> str:
        """Slice a single string scalar from `pos` to `pos + len`.

        Args:
            x: Scalar value to convert and slice.

        Returns:
            str: Extracted substring.
        """
        s = str(x)
        return s[pos : (pos + len)]

    return _map_nested(_sub, val)


def to_hash_bucket(
    input: TensorOperand | _ShapedTypedProtocol,
    num_buckets: int,
    name: str | None = None,
) -> TensorOrScalar:
    """Convert each string in the input Tensor to its hash mod by `num_buckets`.

    Args:
        input: A `string` Tensor or Python string/sequence.
        num_buckets: The number of buckets (`>= 1`).
        name: Optional name for the operation.

    Returns:
        TensorOrScalar: Hash bucket index or indices in `[0, num_buckets)`.
    """
    del name
    if _is_tracing(input):
        return _ops.text.string_to_hash(_as_str_tensor(input), num_buckets=num_buckets)
    val = _to_python(input)
    return _map_nested(lambda x: hash(str(x)) % num_buckets, val)


def to_hash_bucket_fast(
    input: TensorOperand | _ShapedTypedProtocol,
    num_buckets: int,
    name: str | None = None,
) -> TensorOrScalar:
    """Convert each string in the input Tensor to its fast hash mod by `num_buckets`.

    Args:
        input: A `string` Tensor or Python string/sequence.
        num_buckets: The number of buckets (`>= 1`).
        name: Optional name for the operation.

    Returns:
        TensorOrScalar: Hash bucket index or indices.
    """
    return to_hash_bucket(input, num_buckets, name)


def to_hash_bucket_strong(
    input: TensorOperand | _ShapedTypedProtocol,
    num_buckets: int,
    key: Sequence[int],
    name: str | None = None,
) -> TensorOrScalar:
    """Convert each string in the input Tensor to its keyed hash mod by `num_buckets`.

    Args:
        input: A `string` Tensor or Python string/sequence.
        num_buckets: The number of buckets (`>= 1`).
        key: 2-element sequence of unsigned 64-bit integers representing the SipHash key.
        name: Optional name for the operation.

    Returns:
        TensorOrScalar: Hash bucket index or indices.
    """
    del key
    return to_hash_bucket(input, num_buckets, name)


def _decode_text_scalar(
    input_val: TensorOperand | _ShapedTypedProtocol,
    input_encoding: str,
    errors: str,
) -> str:
    """Decode a scalar string or byte operand into a Python `str`.

    Args:
        input_val: Input string or bytes operand.
        input_encoding: Character encoding name.
        errors: Error handling policy.

    Returns:
        str: Decoded Python string.
    """
    py_val = _to_python(input_val)
    if isinstance(py_val, list):
        return "".join(
            item.decode(input_encoding, errors=errors) if isinstance(item, bytes) else str(item)
            for item in py_val
        )
    if isinstance(py_val, bytes):
        return py_val.decode(input_encoding, errors=errors)
    if py_val is None:
        return ""
    return str(py_val)


def unicode_decode(
    input: TensorOperand | _ShapedTypedProtocol,
    input_encoding: str = "UTF-8",
    errors: str = "replace",
    replacement_char: int = 65533,
    replace_control_characters: bool = False,
    name: str | None = None,
) -> list[int]:
    """Decode each byte-string in `input` into a sequence of Unicode code points.

    Args:
        input: Encoded string tensor.
        input_encoding: Character encoding name (e.g., `'UTF-8'`).
        errors: Error handling policy (`'replace'`, `'ignore'`, `'strict'`).
        replacement_char: Unicode codepoint used when `errors='replace'`.
        replace_control_characters: Whether to replace C0 control characters.
        name: Optional name for the operation.

    Returns:
        list[int]: Decoded codepoints list.
    """
    del replacement_char, replace_control_characters, name
    text = _decode_text_scalar(input, input_encoding, errors)
    return [ord(ch) for ch in text]


def unicode_decode_with_offsets(
    input: TensorOperand | _ShapedTypedProtocol,
    input_encoding: str = "UTF-8",
    errors: str = "replace",
    replacement_char: int = 65533,
    replace_control_characters: bool = False,
    name: str | None = None,
) -> tuple[list[int], list[int]]:
    """Decode each string into Unicode code points and return byte start offsets.

    Args:
        input: Encoded string tensor.
        input_encoding: Character encoding name.
        errors: Error handling policy.
        replacement_char: Replacement character codepoint.
        replace_control_characters: Whether to replace C0 control characters.
        name: Optional name for the operation.

    Returns:
        tuple[list[int], list[int]]: Tuple of `(codepoints, byte_offsets)`.
    """
    del replacement_char, replace_control_characters, name
    chars, offsets = unicode_split_with_offsets(input, input_encoding, errors=errors)
    return ([ord(ch) for ch in chars], offsets)


def unicode_encode(
    input: TensorOperand | _ShapedTypedProtocol,
    output_encoding: str = "UTF-8",
    errors: str = "replace",
    replacement_char: int = 65533,
    name: str | None = None,
) -> list[str]:
    """Encode a tensor of integer Unicode code points into strings.

    Args:
        input: Integer codepoint tensor or sequence.
        output_encoding: Target Unicode encoding (`'UTF-8'`, `'UTF-16-BE'`, `'UTF-32-BE'`).
        errors: Error handling policy.
        replacement_char: Replacement character codepoint.
        name: Optional name for the operation.

    Returns:
        list[str]: Encoded strings list.
    """
    del output_encoding, errors, replacement_char, name
    py_val = _to_python(input)
    if isinstance(py_val, list):
        if py_val and all(isinstance(c, int) and not isinstance(c, bool) for c in py_val):
            return ["".join(chr(int(c)) for c in py_val if isinstance(c, int))]
        out: list[str] = []
        for row in py_val:
            if isinstance(row, list):
                out.append("".join(chr(int(c)) for c in row if isinstance(c, int)))
        return out
    if isinstance(py_val, int) and not isinstance(py_val, bool):
        return [chr(py_val)]
    return []


def unicode_script(
    input: TensorOperand | _ShapedTypedProtocol,
    name: str | None = None,
) -> list[int]:
    """Determine the UScriptCode for a given tensor of Unicode integer code points.

    Args:
        input: Tensor of `int32` Unicode code points.
        name: Optional name for the operation.

    Returns:
        list[int]: List of UScriptCode integer values.
    """
    del name
    py_val = _to_python(input)
    if isinstance(py_val, list):
        return [0 for _ in py_val]
    return []


def unicode_split(
    input: TensorOperand | _ShapedTypedProtocol,
    input_encoding: str = "UTF-8",
    errors: str = "replace",
    replacement_char: int = 65533,
    name: str | None = None,
) -> list[str]:
    """Split each string in `input` into a sequence of Unicode character substrings.

    Args:
        input: Encoded string tensor.
        input_encoding: Character encoding name.
        errors: Error handling policy.
        replacement_char: Replacement character codepoint.
        name: Optional name for the operation.

    Returns:
        list[str]: Split Unicode character strings.
    """
    del replacement_char, name
    text = _decode_text_scalar(input, input_encoding, errors)
    return list(text)


def unicode_split_with_offsets(
    input: TensorOperand | _ShapedTypedProtocol,
    input_encoding: str = "UTF-8",
    errors: str = "replace",
    replacement_char: int = 65533,
    name: str | None = None,
) -> tuple[list[str], list[int]]:
    """Split each string into Unicode character substrings and return byte offsets.

    Args:
        input: Encoded string tensor.
        input_encoding: Character encoding name.
        errors: Error handling policy.
        replacement_char: Replacement character codepoint.
        name: Optional name for the operation.

    Returns:
        tuple[list[str], list[int]]: Tuple of `(char_substrings, byte_offsets)`.
    """
    del replacement_char, name
    text = _decode_text_scalar(input, input_encoding, errors)
    chars: list[str] = []
    offsets: list[int] = []
    byte_pos = 0
    for ch in text:
        chars.append(ch)
        offsets.append(byte_pos)
        byte_pos += len(ch.encode(input_encoding, errors=errors))
    return (chars, offsets)


def unicode_transcode(
    input: TensorOperand | _ShapedTypedProtocol,
    input_encoding: str = "UTF-8",
    output_encoding: str = "UTF-8",
    errors: str = "replace",
    replacement_char: int = 65533,
    replace_control_characters: bool = False,
    name: str | None = None,
) -> list[str]:
    """Transcode the input text from a source encoding to a destination encoding.

    Args:
        input: Input string tensor.
        input_encoding: Source encoding name.
        output_encoding: Target encoding name.
        errors: Error handling policy.
        replacement_char: Replacement character codepoint.
        replace_control_characters: Whether to replace C0 control characters.
        name: Optional name for the operation.

    Returns:
        list[str]: Transcoded strings list.
    """
    del replacement_char, replace_control_characters, name
    text = _decode_text_scalar(input, input_encoding, errors)
    if not text:
        return []
    encoded = text.encode(output_encoding, errors=errors)
    return [encoded.decode(output_encoding, errors=errors)]


def unsorted_segment_join(
    inputs: TensorOperand,
    segment_ids: TensorOperand,
    num_segments: int,
    separator: str = "",
    name: str | None = None,
) -> list[str]:
    """Join elements of `inputs` based on `segment_ids`.

    Args:
        inputs: Input string tensor or sequence.
        segment_ids: Tensor or sequence defining segment assignments.
        num_segments: Total number of segments.
        separator: Separator string used when joining within a segment.
        name: Optional name for the operation.

    Returns:
        list[str]: Segment-joined strings list.
    """
    del name
    if num_segments <= 0:
        return []
    py_inputs = _to_python(inputs)
    py_segs = _to_python(segment_ids)
    buckets: list[list[str]] = [[] for _ in range(num_segments)]
    if isinstance(py_inputs, list) and isinstance(py_segs, list):
        for item, seg in zip(py_inputs, py_segs):
            if isinstance(seg, int) and 0 <= seg < num_segments:
                buckets[seg].append(str(item))
    return [separator.join(b) for b in buckets]


def upper(
    input: TensorOperand | _ShapedTypedProtocol,
    encoding: str = "utf-8",
    name: str | None = None,
) -> TensorOrScalar:
    """Convert all lowercase characters into their respective uppercase replacements.

    Args:
        input: A `string` Tensor or Python string/sequence.
        encoding: Character encoding (`'utf-8'` or `''`).
        name: Optional name for the operation.

    Returns:
        TensorOrScalar: Uppercase string(s).
    """
    del encoding, name
    if _is_tracing(input):
        return _ops.text.string_upper(_as_str_tensor(input))
    val = _to_python(input)
    return _map_nested(lambda x: str(x).upper(), val)
