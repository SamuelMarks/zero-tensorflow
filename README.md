# Zero Framework API Shell

> **Note:** This repository is an API-compatible shell. All underlying math, autodiff, and graph execution has been migrated to the [ml-switcheroo-compiler](https://github.com/SamuelMarks/ml-switcheroo-compiler) backend. This repository purely implements frontend routing and syntactic parity for the target framework.

# zero-tensorflow

[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/SamuelMarks/zero-tensorflow/actions/workflows/ci.yml/badge.svg)](https://github.com/SamuelMarks/zero-tensorflow/actions)
[![Test Coverage](https://img.shields.io/badge/test_coverage-100%25-brightgreen.svg)](#)
[![Doc Coverage](https://img.shields.io/badge/doc_coverage-100%25-brightgreen.svg)](#)
[![Official API Parity](https://img.shields.io/badge/official_api_parity-60%25-orange.svg)](#)

## What is zero-tensorflow?

[`zero-tensorflow`](src/zero_tensorflow) is a zero-dependency, pure Python implementation of the [TensorFlow](https://github.com/tensorflow/tensorflow) (OG repo) API surface, complete with full support for Keras via the [zero-keras](https://github.com/SamuelMarks/zero-keras) dependency replicating [Keras](https://github.com/keras-team/keras) (OG repo). It provides the familiar API constructs—like `Tensor`, `Variable`, `math` operations, `GradientTape`, and Keras layers/models—but is entirely stripped of all C++ dependencies, protobuf bindings, and the heavyweight artifacts of the genuine `tensorflow` library.

## Why does this project exist?

This project was built to address the **N-to-M translation problem** in Machine Learning and serves as a Tier 4 frontend in the **Abstract ML Machine Ecosystem** (the [`ml-switcheroo` architecture](ARCHITECTURE.md)). 

The N-to-M translation problem describes the difficulty of supporting $N$ frameworks ([TensorFlow](https://www.tensorflow.org/), [PyTorch](https://pytorch.org/), [JAX](https://github.com/google/jax)) against $M$ execution targets ([WASM](https://webassembly.org/), [WebGPU](https://www.w3.org/TR/webgpu/), [TensorRT](https://developer.nvidia.com/tensorrt)). Writing bespoke translators for every combination is unsustainable. 

### Core Motivations

1. **Avoiding Dependency Hell:** The genuine `tensorflow` package is notoriously large and difficult to install on constrained systems. Many tooling pipelines only need to define, parse, or trace a model architecture to export it. `zero-tensorflow` allows you to construct and evaluate these models anywhere Python can run, relying solely on standard libraries and [`numpy`](https://numpy.org/).
2. **Universal Source-to-Source Compilation:** By acting as a shim, `zero-tensorflow` intercepts mathematical and neural network operations. It delegates tracing to `ml-switcheroo-compiler`, which records execution onto a tape and outputs a strict [ONNX](https://onnx.ai/)-based Logical Graph Dialect (`ml-switcheroo-ir`). From this IR, the ecosystem can statically allocate memory and generate [LEB128](https://en.wikipedia.org/wiki/LEB128) WASM or [WGSL](https://www.w3.org/TR/WGSL/) shaders for in-browser execution.
3. **Cross-Framework Proving Grounds:** State in object-oriented machine learning code (like `tf.Variable` or `nn.Parameter`) complicates compilation. `zero-tensorflow` automatically lifts state into purely functional graph inputs and outputs. This allows us to train identical model architectures across `zero-tensorflow`, `zero-torch`, and `zero-jax` in our unified `zero-zoo` to definitively prove float-for-float "Golden Seed" algorithmic equivalence.

### How it Works

When executing eagerly, `zero-tensorflow` transparently passes operations down to `numpy`. When a user decorates a function or invokes compilation (via `.backward()` equivalent or tracing mechanics), the operations are overloaded to emit `LogicalNode` proxy objects. The underlying `ml-switcheroo-compiler` handles reverse-mode automatic differentiation, common subexpression elimination, and statically binds the exact source Python AST to the generated Intermediate Representation.

---

## Documentation & Plans

- [Architecture Guide](ARCHITECTURE.md)
- [Compliance Information](COMPLIANCE.md)
- [Semantic Plan](SEMANTIC_PLAN.md)
- [TensorFlow TODO](TENSORFLOW_TODO.md)
- [Zero Keras Plan](ZERO_KERAS_PLAN.md)

---

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
