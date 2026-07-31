# Source adapters

A source adapter is the public setup boundary between a deployment's physical data model and
Impulse analysis. It discovers business dimensions and logical channels, validates selections, and
returns the ordinary `impulse_reporting.core.report.Report`. Solvers continue to own physical reads
and reshaping; TSAL, events, aggregations, and reporting remain unchanged.

```python
from impulse_reporting.sources import registered_sources, resolve_source

print(registered_sources())
source = resolve_source()  # configured default, or the only installed source
print(source.list_dimensions(spark))
print(source.list_dimension_values(spark, "plant"))
print(source.list_channels(spark, container_filters={"plant": ["Berlin"]}))

report = source.create_report(
    spark,
    name="analysis",
    container_filters={"plant": ["Berlin"]},
    channels=["temperature"],
    sink=None,
)
temperature = report.get_db().query.channel_with_alias(channel_alias="temperature")
result = report.get_db().query.select(temperature.mean().alias("mean_temperature")).solve(
    spark, solver=report.get_solver()
)
```

Registration is explicit and import-driven:

```python
from impulse_reporting.sources import register_source

@register_source("factory-telemetry", default=True)
class FactoryTelemetrySource(SourceAdapter):
    ...
```

Impulse never scans files or guesses an import. A deployment imports its adapter package during
environment setup. If multiple sources exist, mark one default or select by name.

See `examples/source_adapter/factory_telemetry.py` for a complete neutral adapter. It includes
multi-value dimension filters translated into native tag-filter DNF, logical aliases with priority,
sinkless report creation, and a custom solver registered independently of all agent skills.
