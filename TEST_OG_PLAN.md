# Official Test Suite Porting Plan

This document outlines the step-by-step delivery plan for porting the official TensorFlow and Keras test suites to validate `zero-tensorflow`. The goal is to ensure 1-to-1 API compatibility and verify that outputs match identically (or are `allclose` where floating-point math diverges slightly).

## Phase 1: Setup and Infrastructure
- [x] Create a dedicated directory for official tests (e.g., `tests/official_tf/` and `tests/official_keras/`).
- [x] Identify the specific tags or commits in the official `tensorflow/tensorflow` and `keras-team/keras` repositories to port tests from (ensuring version alignment).
- [x] Update `requirements-test.txt` with required testing dependencies (e.g., `absl-py`, `parameterized`, `mock`, `pytest-mock`).
- [x] Create a global test compat/fixture layer that aliases `tensorflow` to `zero_tensorflow` during pytest execution so that official test files can run without mass string-replacements of `import tensorflow as tf`.
- [x] Implement a stubbing mechanism for unsupported boilerplate components (like `tf.config`, `tf.debugging`, CPU/GPU device placement, or hardware assertions).

## Phase 2: Core TensorFlow Tests (`tf.*`)
- [x] Port `tf.Tensor` and basic math operation tests (`tf.math`).
  - [x] Verify operator overloads (`__add__`, `__mul__`, etc.).
  - [x] Ensure `numpy.testing.assert_allclose` is used for floating-point comparisons.
- [x] Port `tf.nn` test suite.
  - [x] Ensure activations, convolutions, and pooling operations match expectations.
- [x] Port `tf.data` test suite.
  - [x] Verify `tf.data.Dataset` behaviors, iterators, and transformations.
- [x] Port `tf.Variable` and gradient tracking tests (e.g., `GradientTape`).
- [x] Map out failing tests related to unsupported operations and add them to `TENSORFLOW_TODO.md`. Optionally mark them as `@pytest.mark.skip(reason="Not yet implemented")`.

## Phase 3: Keras Tests (`tf.keras.*`)
- [x] Port `tf.keras.layers` test suite (Core, Convolutional, Pooling, etc.).
- [x] Port `tf.keras.models` and `Functional/Sequential` API tests.
- [x] Port `tf.keras.optimizers` test suite.
  - [x] Verify weight updates precisely match official implementations.
- [x] Port `tf.keras.losses` and `tf.keras.metrics` test suites.
- [x] Port `tf.keras.initializers` and `tf.keras.regularizers` tests.

## Phase 4: CI & Continuous Testing
- [x] Run the complete test suite locally to establish a baseline pass rate.
- [x] Investigate and fix any tests failing purely due to precision differences (adjust `atol`/`rtol`).
- [x] Integrate the new test suite into `.github/workflows/ci.yml`.
- [x] Set up a reporting mechanism to track the percentage of the official test suite that passes, ensuring it strictly increases over time.