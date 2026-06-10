# Plan: Replacing Internal Keras with External `zero-keras`

This document outlines the detailed steps required to migrate `zero-tensorflow` from its internal `keras` implementation to the standalone `zero-keras` package (located at `https://github.com/SamuelMarks/zero-keras`). It includes steps for verifying feature parity, as the two implementations have diverged over time.

## 1. Feature Parity & External Codebase Updates
Before swapping out the internal module, we must ensure `zero-keras` contains all features currently supported by `zero_tensorflow/keras`. 

**Note:** A preliminary diff reveals that while `zero-keras` is generally larger and contains new files (`activations.py`, `core_layers.py`), some internal files like `initializers.py` and `schedules.py` are slightly larger, indicating potential missing features in the external package.

* [x] Compare `src/zero_tensorflow/keras/initializers.py` with `../zero-keras/src/zero_keras/initializers.py`
  * [x] Identify missing initializer classes or parameters.
  * [x] Port missing features to `../zero-keras`.
* [x] Compare `src/zero_tensorflow/keras/optimizers/schedules.py` with `../zero-keras/src/zero_keras/optimizers/schedules.py`
  * [x] Identify missing learning rate schedules or parameters.
  * [x] Port missing features to `../zero-keras`.
* [x] Compare `src/zero_tensorflow/keras/layers.py` with `../zero-keras/src/zero_keras/layers.py` (and `core_layers.py`)
  * [x] Ensure all layer definitions, arguments, and behaviors match or exceed the internal implementation.
  * [x] Check for differences in Base `Layer` and `Model` implementations.
* [x] Compare `src/zero_tensorflow/keras/losses.py` with `../zero-keras/src/zero_keras/losses.py`
  * [x] Verify parity of loss functions.
* [x] Compare `src/zero_tensorflow/keras/metrics.py` with `../zero-keras/src/zero_keras/metrics.py`
  * [x] Verify parity of metrics and stateful metric tracking.
* [x] Compare `src/zero_tensorflow/keras/optimizers/__init__.py` with `../zero-keras/src/zero_keras/optimizers/__init__.py`
  * [x] Verify parity of optimizers (e.g., Adam, SGD, RMSprop) and their step logic.
* [x] Ensure `../zero-keras` test suite passes and provides adequate coverage for ported features.
* [x] Push updates to the `master` branch of `https://github.com/SamuelMarks/zero-keras`.

## 2. Dependency Management
Once the external package is up-to-date, add it to this repository's dependencies.

* [x] Open `pyproject.toml`.
* [x] Add `zero-keras @ git+https://github.com/SamuelMarks/zero-keras.git@master` to the `dependencies` list.
* [x] If developing locally first, optionally test with `zero-keras @ file://localhost/...` before finalizing the git URL.
* [x] Run local installation (`pip install -e .[test]`) to verify the dependency resolves correctly.

## 3. Internal Clean-up (Removing Local Keras)
Remove the redundant internal package to avoid conflicts.

* [x] Delete the `src/zero_tensorflow/keras/` directory.
  * `rm -rf src/zero_tensorflow/keras/`
* [x] Verify no stray `__pycache__` or `.pyc` files remain.

## 4. Aliasing and Exports
Update the main module to point `zero_tensorflow.keras` to the new external dependency.

* [x] Open `src/zero_tensorflow/__init__.py`.
* [x] Remove the internal import: `from . import keras`.
* [x] Add standard library import: `import sys`.
* [x] Import the external package: `import zero_keras as keras`.
* [x] Add `sys.modules` alias to support deep imports (e.g., `from zero_tensorflow.keras import layers`):
  * `sys.modules["zero_tensorflow.keras"] = keras`
* [x] Update `__all__` list if necessary (should remain `"keras"`).

## 5. Scripts and Code Generation Updates
Ensure utility scripts use the correct module paths.

* [x] Review `generate_layers.py`.
* [x] Update code generation strings to use `from zero_keras import layers` instead of `from zero_tensorflow.keras import layers` if standardizing, or leave as-is relying on the alias.
* [x] Run `generate_layers.py` to ensure it still executes without errors.

## 6. Test Suite Migration
Resolve the local test suite for keras.

* [x] Review `tests/keras/` to see if there are any `zero-tensorflow` specific integration tests that should be kept.
* [x] If the tests are purely unit tests for keras functionality, rely on the `zero-keras` upstream test suite.
* [x] Delete redundant unit tests: `rm -rf tests/keras/` (or specific files within).
* [x] Run the remaining `zero-tensorflow` test suite (`pytest`) to ensure core components (`data`, `nn`, `math`, `Tensor`) still function correctly and the aliased `keras` module loads properly.
