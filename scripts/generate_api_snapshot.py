import json
import inspect
import tensorflow as tf


def safe_name(obj):
    try:
        return obj.__name__
    except Exception:
        return ""


def get_public_api(module, prefix="tf", depth=0):
    if depth > 2:
        return {}
    api = {}
    for name, obj in inspect.getmembers(module):
        if name.startswith("_"):
            continue
        if inspect.ismodule(obj):
            if safe_name(obj).startswith("tensorflow") or safe_name(obj).startswith(
                "keras"
            ):
                api[name] = {
                    "type": "module",
                    "contents": get_public_api(obj, f"{prefix}.{name}", depth + 1),
                }
        elif inspect.isclass(obj):
            api[name] = {"type": "class"}
        elif callable(obj):
            api[name] = {"type": "function"}
        else:
            api[name] = {"type": "value"}
    return api


if __name__ == "__main__":
    api = get_public_api(tf)
    with open("snapshots/tf_api.json", "w") as f:
        json.dump(api, f, indent=2)
