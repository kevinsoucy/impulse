"""Notebook bootstrap for the BYOD demo.

`connect()` creates the demo's widgets, reads them, and builds the
`BYODConfig` — collapsing the per-notebook configuration block to a single
call so the notebook body stays focused on the demo narrative.

    ctx = connect(dbutils)
    cfg = ctx["cfg"]

The four base widgets (`adapter`, `dataset_version`, `catalog`,
`schema_prefix`) carry the job's `base_parameters` (and supply interactive
defaults). Pass `extra_widgets` for notebook-specific inputs, and
`report_table_suffix` to also build the Impulse `report_config`.
"""

from __future__ import annotations

from lib.byod_config import BYODConfig

# Base widgets every notebook needs. Job runs override these via the task's
# `base_parameters`; interactive runs fall back to the defaults here.
_BASE_WIDGETS = (
    ("adapter",         "nuscenes",        "Adapter name (e.g. nuscenes, a2d2)"),
    ("dataset_version", "v1.0-mini",       "Adapter-specific variant"),
    ("catalog",         "main",            "UC catalog (must exist)"),
    ("schema_prefix",   "lakevision_demo", "Schema prefix"),
)


def _build_report_config(cfg: BYODConfig, table_prefix: str) -> dict:
    """The Impulse Report source/sink config. Metadata tables are unchanged
    from 1.0 — only the Series API is new — so the source block still reads the
    `container_*`/`channel_*` tables."""
    return {
        "source": {
            "container_metrics_table": cfg.t_container_metrics,
            "container_tags_table":    cfg.t_container_tags,
            "channel_metrics_table":   cfg.t_channel_metrics,
            "channel_tags_table":      cfg.t_channel_tags,
            "channels_uri":            cfg.t_channels,
        },
        "unity_sink": {
            "catalog":      cfg.catalog,
            "schema":       cfg.schema_gold,
            "table_prefix": table_prefix,
        },
        "query_engine": {"solver": "DeltaSolver", "data_type": "RAW"},
        # container_id only — the channel start_dt/stop_dt vs event start_ts/stop_ts
        # naming mismatch rules out a time dimension here.
        "measurement_dimensions": ["container_id"],
    }


def connect(dbutils, *, extra_widgets=(), report_table_suffix=None, with_adapter=False) -> dict:
    """Create + read the demo widgets and build the `BYODConfig`.

    Parameters
    ----------
    dbutils
        The notebook's dbutils handle.
    extra_widgets
        Notebook-specific widgets as `(name, default, label)` tuples. A 4th
        element (a list of choices) makes it a dropdown instead of a text box.
    report_table_suffix
        When set, also build `report_config` with a gold sink table prefix of
        ``f"{adapter}_{suffix}"`` (e.g. ``"demo"`` for 02, ``"explore"`` for 05).
    with_adapter
        When True, also resolve + instantiate the adapter as ``ctx["adapter"]``.

    Returns
    -------
    dict
        Every widget value keyed by name, plus ``"cfg"``, and — when requested —
        ``"report_config"`` and ``"adapter"``.
    """
    specs = list(_BASE_WIDGETS) + list(extra_widgets)
    for spec in specs:
        name, default, label = spec[0], spec[1], spec[2]
        if len(spec) > 3:
            dbutils.widgets.dropdown(name, default, list(spec[3]), label)
        else:
            dbutils.widgets.text(name, default, label)
    vals = {spec[0]: dbutils.widgets.get(spec[0]) for spec in specs}

    cfg = BYODConfig.for_adapter(
        adapter_name=vals["adapter"],
        dataset_version=vals["dataset_version"] or None,
        catalog=vals["catalog"],
        schema_prefix=vals["schema_prefix"],
    )

    ctx: dict = dict(vals)
    ctx["cfg"] = cfg
    if report_table_suffix is not None:
        ctx["report_config"] = _build_report_config(cfg, f"{cfg.adapter_name}_{report_table_suffix}")
    if with_adapter:
        from lib.adapter import resolve
        ctx["adapter"] = resolve(vals["adapter"])(cfg)

    print(f"adapter={cfg.adapter_name}  dataset={cfg.dataset_version}  "
          f"sink={cfg.catalog}.{cfg.schema_gold}.*")
    return ctx
