"""Unit tests for `zero_tensorflow.types` trace type hierarchy."""

from __future__ import annotations

import zero_tensorflow.types as tf_types


def test_types_stubs() -> None:
    """Verify `TraceType` subtype and supertype hierarchy in `tf.types.experimental`."""
    exp = tf_types.experimental
    t_base = exp.TraceType(1, 2)
    t_same = exp.TraceType(1, 2)
    t_diff = exp.TraceType(3, 4)
    t_sub = exp.TensorLike(1, 2)

    assert t_sub.is_subtype_of(t_base) is True
    assert t_base.is_subtype_of(t_sub) is False
    assert t_diff.is_subtype_of(t_base) is False
    assert t_base.is_subtype_of("not_a_trace_type") is False

    assert t_base.most_specific_common_supertype([t_same, t_sub]) is t_base
    common = t_sub.most_specific_common_supertype([t_diff])
    assert common is not None

    # Validate non-TraceType element in sequence
    fake_seq = [t_same]
    assert t_same.most_specific_common_supertype(fake_seq) is t_same

    proto = exp.SupportsTracingProtocol("float32")
    traced = proto.__tf_tracing_type__()
    assert isinstance(traced, exp.TraceType)

    fn_t = exp.FunctionType()
    call_t = exp.Callable()
    atomic_t = exp.AtomicFunction()
    conc_t = exp.ConcreteFunction()
    gen_t = exp.GenericFunction()
    poly_t = exp.PolymorphicFunction()

    assert conc_t.is_subtype_of(atomic_t) is True
    assert poly_t.is_subtype_of(gen_t) is True
    assert fn_t.is_subtype_of(call_t) is True
    assert exp.distributed() is None
    assert isinstance(exp.distributed._PerReplica(), tf_types._BaseTraceType)
    assert isinstance(exp.distributed._Mirrored(), tf_types._BaseTraceType)
