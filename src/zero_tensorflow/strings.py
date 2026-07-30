"TensorFlow strings module."

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ml_switcheroo_compiler.core.config import config
from zero_keras import ops as _ops

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


def _is_tracing(val):
    if hasattr(val, "shape") and hasattr(val, "dtype"):
        return not config.eager_mode
    return False


def _to_python(val: Any) -> Any:
    if hasattr(val, "numpy"):
        val = val.numpy()
    if hasattr(val, "tolist"):
        val = val.tolist()
    return val


def _map_nested(func, val):
    if isinstance(val, (list, tuple)):
        return [_map_nested(func, v) for v in val]
    return func(val)


def _reduce_join_nested(val, sep=""):
    if isinstance(val, (list, tuple)):
        return sep.join([_reduce_join_nested(v, sep) for v in val])
    return str(val)


def join(inputs: Sequence[Any], separator: str = "", name: str | None = None) -> Any:
    if not inputs:
        return ""
    if any(_is_tracing(x) for x in inputs):
        return _ops.text.string_join(inputs, separator=separator)
    py_inputs = [_to_python(x) for x in inputs]
    is_list = any(isinstance(x, list) for x in py_inputs)
    if not is_list:
        return separator.join(str(x) for x in py_inputs)
    length = max((len(x) for x in py_inputs if isinstance(x, list)), default=0)
    res = []
    for i in range(length):
        parts = []
        for inp in py_inputs:
            if isinstance(inp, list):
                parts.append(str(inp[i]))
            else:
                parts.append(str(inp))
        res.append(separator.join(parts))
    return res


def split(
    input: Any,
    sep: str | None = None,
    maxsplit: int = (-1),
    name: str | None = None,
) -> Any:
    if _is_tracing(input):
        return _ops.text.string_split(input, sep=sep, maxsplit=maxsplit)
    val = _to_python(input)
    return _map_nested((lambda x: str(x).split(sep, maxsplit)), val)


def length(input: Any, unit: str = "BYTE", name: str | None = None) -> Any:
    if _is_tracing(input):
        return _ops.text.string_length(input, unit=unit)
    val = _to_python(input)
    return _map_nested((lambda x: len(str(x))), val)


def regex_replace(
    input: Any,
    pattern: str,
    rewrite: str,
    replace_global: bool = True,
    name: str | None = None,
) -> Any:
    if _is_tracing(input):
        return _ops.text.regex_replace(
            input, pattern, rewrite, replace_global=replace_global
        )
    import re

    p = re.compile(pattern)
    count = 0 if replace_global else 1
    val = _to_python(input)
    return _map_nested((lambda x: p.sub(rewrite, str(x), count=count)), val)


def regex_full_match(input: Any, pattern: str, name: str | None = None) -> Any:
    if _is_tracing(input):
        return _ops.text.regex_full_match(input, pattern)
    import re

    p = re.compile(pattern)
    val = _to_python(input)
    return _map_nested((lambda x: p.fullmatch(str(x)) is not None), val)


def to_number(input: Any, out_type: Any = None, name: str | None = None) -> Any:
    if _is_tracing(input):
        return _ops.text.string_to_number(input, out_type=out_type)
    val = _to_python(input)
    return _map_nested((lambda x: float(x)), val)


def bytes_split(input: Any, name: str | None = None) -> Any:
    if _is_tracing(input):
        return _ops.text.string_split(input, sep="")
    val = _to_python(input)
    return _map_nested((lambda x: list(str(x))), val)


def format(
    template: str,
    inputs: Sequence[Any],
    placeholder: str = "{}",
    summarize: int = 3,
    name: str | None = None,
) -> Any:
    py_inputs = [_to_python(x) for x in inputs]
    s = template
    for i in py_inputs:
        s = s.replace(placeholder, str(i), 1)
    return s


def lower(input: Any, encoding: str = "utf-8", name: str | None = None) -> Any:
    if _is_tracing(input):
        return _ops.text.string_lower(input, encoding=encoding)
    val = _to_python(input)
    return _map_nested((lambda x: str(x).lower()), val)


def ngrams(
    data: Any,
    ngram_width: Any,
    separator: str = " ",
    pad_values: Any | None = None,
    padding_width: Any | None = None,
    preserve_short_sequences: bool = False,
    name: str | None = None,
) -> Any:
    return []


def reduce_join(
    inputs: Any,
    axis: Any | None = None,
    keepdims: bool = False,
    separator: str = "",
    name: str | None = None,
) -> Any:
    if _is_tracing(inputs):
        return _ops.reduce_join(
            inputs, axis=axis, keepdims=keepdims, separator=separator
        )
    val = _to_python(inputs)
    return _reduce_join_nested(val, sep=separator)


def strip(input: Any, name: str | None = None) -> Any:
    if _is_tracing(input):
        return _ops.text.regex_replace(input, "^\\s+|\\s+$", "")
    val = _to_python(input)
    return _map_nested((lambda x: str(x).strip()), val)


def substr(
    input: Any, pos: Any, len: Any, unit: str = "BYTE", name: str | None = None
) -> Any:
    if _is_tracing(input):
        return _ops.text.string_substr(input, pos, len, unit=unit)
    val = _to_python(input)

    def _sub(x):
        s = str(x)
        return s[pos : (pos + len)]

    return _map_nested(_sub, val)


def to_hash_bucket(input: Any, num_buckets: int, name: str | None = None) -> Any:
    if _is_tracing(input):
        return _ops.text.string_to_hash(input, num_buckets=num_buckets)
    val = _to_python(input)
    return _map_nested((lambda x: hash(str(x)) % num_buckets), val)


def to_hash_bucket_fast(input: Any, num_buckets: int, name: str | None = None) -> Any:
    return to_hash_bucket(input, num_buckets, name)


def to_hash_bucket_strong(
    input: Any, num_buckets: int, key: Sequence[int], name: str | None = None
) -> Any:
    return to_hash_bucket(input, num_buckets, name)


def unicode_decode(
    input: Any,
    input_encoding: str,
    errors: str = "replace",
    replacement_char: int = 65533,
    replace_control_characters: bool = False,
    name: str | None = None,
) -> Any:
    return []


def unicode_decode_with_offsets(
    input: Any,
    input_encoding: str,
    errors: str = "replace",
    replacement_char: int = 65533,
    replace_control_characters: bool = False,
    name: str | None = None,
) -> Any:
    return ([], [])


def unicode_encode(
    input: Any,
    output_encoding: str,
    errors: str = "replace",
    replacement_char: int = 65533,
    name: str | None = None,
) -> Any:
    return []


def unicode_script(input: Any, name: str | None = None) -> Any:
    return []


def unicode_split(
    input: Any,
    input_encoding: str,
    errors: str = "replace",
    replacement_char: int = 65533,
    name: str | None = None,
) -> Any:
    return []


def unicode_split_with_offsets(
    input: Any,
    input_encoding: str,
    errors: str = "replace",
    replacement_char: int = 65533,
    name: str | None = None,
) -> Any:
    return ([], [])


def unicode_transcode(
    input: Any,
    input_encoding: str,
    output_encoding: str,
    errors: str = "replace",
    replacement_char: int = 65533,
    replace_control_characters: bool = False,
    name: str | None = None,
) -> Any:
    return []


def unsorted_segment_join(
    inputs: Any,
    segment_ids: Any,
    num_segments: Any,
    separator: str = "",
    name: str | None = None,
) -> Any:
    return []


def upper(input: Any, encoding: str = "utf-8", name: str | None = None) -> Any:
    if _is_tracing(input):
        return _ops.text.string_upper(input, encoding=encoding)
    val = _to_python(input)
    return _map_nested((lambda x: str(x).upper()), val)
