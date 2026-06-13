import json
import inspect
import sys
import zero_tensorflow as ztf
import typing


def check_parity(ztf_obj, tf_snapshot, path="tf"):
    errors = []

    # Use __all__ if available, else get members
    if hasattr(ztf_obj, "__all__"):
        names = ztf_obj.__all__
        members = [(n, getattr(ztf_obj, n)) for n in names if hasattr(ztf_obj, n)]
    else:
        members = inspect.getmembers(ztf_obj)

    for name, obj in members:
        if name.startswith("_"):
            continue

        # Ignore typing
        if obj is typing.Any or obj is typing.Optional or obj is typing.Callable:
            continue
        if name in ("Any", "Optional", "Callable", "ArrayIterator", "PyIterator"):
            continue

        # skip imported modules that aren't part of the core API
        if inspect.ismodule(obj) and not obj.__name__.startswith("zero_tensorflow"):
            continue

        if name not in tf_snapshot:
            # Special case aliases
            if path == "tf" and name == "keras":
                continue  # Keras is separate or tf.keras
            errors.append(f"Missing in official TF: {path}.{name}")
            continue

        tf_info = tf_snapshot[name]

        # We don't strictly enforce type because sometimes ztf uses class instead of module (e.g. math)
        if inspect.ismodule(obj) or inspect.isclass(obj):
            if inspect.isclass(obj) and name in ("math", "nn", "data"):
                # ztf implemented these as classes, we should check their methods against tf module contents
                for sub_name, sub_obj in inspect.getmembers(obj):
                    if sub_name.startswith("_"):
                        continue
                    if callable(sub_obj) or inspect.isclass(sub_obj):
                        if sub_name not in tf_info.get("contents", {}):
                            # some exceptions
                            if sub_name == "pow":
                                continue
                            errors.append(
                                f"Missing in official TF: {path}.{name}.{sub_name}"
                            )
            elif tf_info["type"] == "module":
                errors.extend(
                    check_parity(obj, tf_info.get("contents", {}), f"{path}.{name}")
                )

    return errors


def main():
    try:
        with open("snapshots/tf_api.json", "r") as f:
            tf_snapshot = json.load(f)
    except FileNotFoundError:
        print("snapshots/tf_api.json not found. Run generate_api_snapshot.py first.")
        sys.exit(1)

    errors = check_parity(ztf, tf_snapshot)

    if errors:
        print(f"API Parity check failed with {len(errors)} errors:")
        for err in errors:
            print("  " + err)
        sys.exit(1)
    else:
        print(
            "API Parity check passed. All zero-tensorflow APIs exist in official TensorFlow."
        )


if __name__ == "__main__":
    main()
