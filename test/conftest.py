"""
Pytest configuration: install lightweight stubs for GPU-only modules so that
unit tests can import sgocr code without a physical GPU or heavy ML libraries.

PIL and requests are actually installed in this environment, so we let them
through as real modules. Only torch, torchvision, transformers, and the AI
client packages (openai, anthropic, google) are absent and need stubs.
"""
from __future__ import annotations

import sys
import types


def _make_stub_type(name: str) -> type:
    """Return a callable stub class usable as a base class or constructor.

    Instance methods are stubbed so the object can be used in common
    patterns (context manager, iteration, etc.).
    """
    short = name.rsplit(".", 1)[-1]
    return type(
        short,
        (),
        {
            "__init__": lambda self, *a, **kw: None,
            # Instance-level attribute access returns a new stub instance.
            "__getattr__": lambda self, k: _make_stub_type(f"{name}.{k}")(),
            "__class_getitem__": classmethod(lambda cls, item: cls),
            "__iter__": lambda self: iter([]),
            "__len__": lambda self: 0,
            "__bool__": lambda self: True,
            "__call__": lambda self, *a, **kw: _make_stub_type(name)(),
            "__enter__": lambda self: self,
            "__exit__": lambda self, *a: False,
        },
    )


def _make_stub(name: str) -> types.ModuleType:
    m = types.ModuleType(name)

    def _getattr(self: types.ModuleType, k: str) -> object:
        full = f"{name}.{k}"
        # Prefer a pre-installed stub module over synthesising a new type.
        if full in sys.modules:
            return sys.modules[full]
        # Return a callable stub type so patterns like `class Foo(Dataset):`
        # work without errors.
        return _make_stub_type(full)

    m.__class__ = type(
        "_StubModule",
        (types.ModuleType,),
        {"__getattr__": _getattr},
    )
    return m


# Only stub modules that are genuinely absent. PIL and requests are
# installed in this environment, so we do NOT stub them.
_GPU_STUBS = [
    "torch",
    "torch.nn",
    "torch.cuda",
    "torch.utils",
    "torch.utils.data",
    "torchvision",
    "torchvision.transforms",
    "transformers",
    "openai",
    "anthropic",
    "google",
    "google.generativeai",
    "google.api_core",
    "google.api_core.exceptions",
]

for _mod in _GPU_STUBS:
    if _mod not in sys.modules:
        sys.modules[_mod] = _make_stub(_mod)
