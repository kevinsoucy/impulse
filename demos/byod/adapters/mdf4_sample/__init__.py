"""mdf4_sample — MDF4 reference adapter directory (Phase 1: conversion-only).

Phase 1 ships the PandaSet → MDF4 conversion script and a round-trip test that
proves the MDF4 layout is structurally correct. The Adapter class, config.yaml,
loader, and per-modality mappers land in Phase 2 once the centralized MDF4
reader (impulse.readers.mdf4) is available.
"""
