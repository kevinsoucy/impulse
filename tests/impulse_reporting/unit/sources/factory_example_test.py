import importlib.util
import sys
from pathlib import Path

from impulse_reporting.sources.registry import _clear_registry, resolve_source


def test_factory_example_imports_and_registers_an_adapter():
    _clear_registry()
    example = Path(__file__).parents[4] / "examples" / "source_adapter" / "factory_telemetry.py"
    spec = importlib.util.spec_from_file_location("factory_telemetry_example", example)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    source = resolve_source("factory-telemetry")
    assert source.name == "factory-telemetry"
    assert [dimension.name for dimension in source.list_dimensions(None)] == ["plant", "line"]
