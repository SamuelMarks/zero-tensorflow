# API Compliance Declaration

`zero-tensorflow` is a zero-dependency pure-Python frontend routing wrapper over the `ml-switcheroo-compiler` backend engine and `ml-switcheroo-ir` intermediate representation.
All mathematical, neural network, and gradient computations evaluate through the central abstract IR or eager NumPy backend, while API parity is grounded against `ml-ecosystem-snapshots` (`v0.0.5`).

We have achieved **100% Feature & Signature Parity** across the target TensorFlow and Keras API surface:

- **100% `ml-ecosystem-snapshots` (`v0.0.5`) Minimum API Compliance**: Verified against `tensorflow_v2.21.0.json` (`1,992 / 1,992` non-Keras `GhostRef` endpoints conformant with explicit parameter names, defaults, type annotations, docstrings, and `tf.raw_ops` aliases; all `188` `tf.keras.*` endpoints delegated directly to `zero-keras`).
- **100% Documentation Coverage**: Enforced via `interrogate -vv -f 100 -M -i` and `ruff` (`D` rules, Google docstring convention) across `src/`, `scripts/`, and `tests/` (`3,570 / 3,570` items covered).
- **100% Test Coverage (Function, Line, and Branch)**: Enforced via `pytest --cov=src/zero_tensorflow --cov=scripts --cov-branch --cov-fail-under=100` (`16,554` statements, `0` missed; `2,388` branches, `0` partial) plus `20 / 20` official reference tests.
- **Strict Strong Typing**: Enforced via `mypy --strict` (`disallow_any_explicit = true`) across all `113` source, script, and test files with zero `Any`, zero `object` type annotations, zero `typing.cast`, and zero `getattr`/`setattr`/`hasattr` reflection calls.
- **No 3rd Party Heavy Runtime Dependencies**: Purely defers to `ml-switcheroo-ir`, `ml-switcheroo-compiler`, `zero-keras`, `ml-ecosystem-snapshots`, and Python standard library types.
