# API Compliance Declaration

`zero-tensorflow` is a zero-dependency pure-Python frontend routing wrapper over the `ml-switcheroo-compiler` backend engine and `ml-switcheroo-ir` intermediate representation.
All mathematical, neural network, and gradient computations evaluate through the central abstract IR or eager NumPy backend, while API parity is grounded against `ml-ecosystem-snapshots` (`v0.0.4`).

We have achieved **100% Feature & Signature Parity** across the target TensorFlow and Keras API surface:

- **100% Documentation Coverage**: Enforced via `interrogate -vv -f 100 -M -i` and `ruff` (`D` rules, Google docstring convention) across `src/`, `scripts/`, and `tests/` (2,929/2,929 items covered).
- **100% Test Coverage (Function, Line, and Branch)**: Enforced via `pytest --cov=src/zero_tensorflow --cov=scripts --cov-branch --cov-fail-under=100` (10,781 statements, 0 missed; 448 branches, 0 partial).
- **Strict Strong Typing**: Enforced via `mypy --strict` (`disallow_any_explicit = true`) across all 104 source, script, and test files with zero `Any`, zero `object` type annotations, zero `typing.cast`, and zero `getattr`/`setattr`/`hasattr` reflection calls.
- **No 3rd Party Heavy Runtime Dependencies**: Purely defers to `ml-switcheroo-ir`, `ml-switcheroo-compiler`, `zero-keras`, `ml-ecosystem-snapshots`, and Python standard library types.
