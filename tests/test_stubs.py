import inspect

import zero_tensorflow as tf


def _call_stubs(obj, visited=None):
    if visited is None:
        visited = set()
    if id(obj) in visited:
        return
    visited.add(id(obj))

    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            attr = getattr(obj, name)
        except Exception:  # noqa: BLE001, S112
            continue

        if (
            inspect.isfunction(attr)
            or inspect.ismethod(attr)
            or isinstance(attr, staticmethod)
        ):
            try:
                attr()
            except NotImplementedError:
                pass
            except Exception:  # noqa: BLE001, S110
                pass
        elif inspect.isclass(attr):
            _call_stubs(attr, visited)

            # also try calling class methods that might be bound
            for n, v in inspect.getmembers(attr):
                if n.startswith("_"):
                    continue
                if inspect.isfunction(v) or inspect.ismethod(v):
                    try:
                        v()
                    except NotImplementedError:
                        pass
                    except Exception:  # noqa: BLE001, S110
                        pass


def test_stubs():
    _call_stubs(tf)
