---
sidebar_position: 1
---

# Motivation

## The Challenge: Large-Scale Measurement Data

Automotive testing, industrial IoT, and other measurement-intensive domains generate enormous volumes of time-series
sensor data.
A single vehicle validation campaign can produce petabytes of recordings spread across thousands of measurement files,
each containing hundreds of channels sampled at varying rates.

Extracting actionable insights from this data is critical.
Engineers need to understand how systems behave under specific operating conditions, identify anomalies,
and produce standardized reports that feed into design decisions.
Yet the sheer scale and structural complexity of measurement data make ad-hoc analysis approaches unsustainable.

## Capabilities of Impulse

### Unified Query Language for Measurement Time-Series

Measurement data in a lakehouse architecture is typically stored across multiple normalized tables:
container metadata, channel metadata, tags, metrics, and the raw time-series blobs themselves.

Because of the complexity and amount of data involved, answering even simple questions, becomes a non-trivial
engineering task.
Most of the engineers have a deep technical background, but aren't necessarily experts in distributed computing.

To be able to extract relevant insights without the need for PySpark expertise, domain experts need a unified query
language that abstracts away the underlying data model and execution engine.

Impulse introduces **TSAL (Time Series Analytics Language)**,
a Pythonic expression language that abstracts away the underlying data model. Selecting a physical channel,
deriving a virtual signal, or defining an event condition becomes a single, readable expression:

```python
# Select a physical channel by tags
eng_rpm = db.query.channel(channel_name='Engine RPM', brand='CAR_BRAND', model='MODEL_TEST')

# Derive a virtual signal
avg_temp = (amb_air_temp + intake_air_temp) / 2

# Define an event as a boolean expression over signals
high_rpm = (eng_rpm > 2000) & (eng_rpm < 5000)
```

No complex joins or complex PySpark code is needed.

A subtle but important point: an event expression does not yield a value at every
timestamp -- it yields the **time windows** during which the condition held.
Those windows are themselves composable. Combining them with `&`, `|`, and `~`
expresses that several conditions overlapped, alternated, or excluded one another
in time. Answering *when* a situation occurred across an entire campaign stays a
single readable expression rather than a hand-written scan.

### Bridging Domain Logic and Distributed Computing

Domain experts, think in terms of **channels**, **events**, and **aggregations**.
They do not think in terms of Spark DataFrames, partitioning strategies, or user-defined functions.
Forcing domain experts to learn distributed computing internals creates a steep learning curve and slows down analysis
cycles.

Impulse translates declarative TSAL expressions into optimized Spark execution through custom query solvers.
Domain experts are now able to write expressions; the framework handles distribution, serialization, and execution
in an optimal manner.

### Repetitive Report Patterns

Every measurement data analysis follows a recognizable pattern:

1. Configure the data source connection
2. Filter containers by metadata (vehicle type, test campaign, date range)
3. Select and transform channels of interest
4. Define time windows (events) relevant to the analysis
5. Compute aggregations (histograms, statistics) within those events
6. Persist or export the results

Impulse's **Report** orchestrator standardizes this entire workflow. A JSON configuration defines the data
source and sink. Events and aggregations are declared as composable objects.
Two method calls execute the full pipeline:

```python
my_report.determine_report()  # compute all events and aggregations
my_report.persist_results()  # write results to the Gold layer
```

This reduces a typical analysis from hundreds of lines of ad-hoc Spark code to a concise, declarative report definition.

### Support for Complex Datasets and Advanced Analytics

Measurement data often consists out of thousands of channels, which are sampled at different rates, and have complex interdependencies.
To allow comprehensive analysis, Impulse supports complex datasets with multiple channels and advanced analytics.

With the out of the box support for interpolation, resampling, and windowing operations, domain experts can easily align
channels with different sampling rates, derive new signals, and define complex events.

This is necessary to truly be able to understand and analyze the recorded data and identify areas of interest for further investigation.

### Events Across Independent Signals and Entities

Not every question lives in a single channel. A situation of interest often arises
only when independent signals coincide -- one subsystem crossing a threshold while
another holds a particular state. Because every event expression evaluates to time
windows, Impulse composes them directly: the overlap of two independent conditions
is itself an event, written as one expression.

Measurement data is also increasingly dense. Alongside scalar channels, domains
such as ADAS object detection, industrial defect inspection, or ECU diagnostics
produce **per-frame tables**, where many entities -- detected objects, inspection
stations, individual control units -- share the same timestamp, each carrying its
own row of attributes. Impulse registers such a table as a **series** in a few
lines, and domain experts author the same predicate expressions over it as over
channels:

```python
ot = db.query.series('object_tracks')

# A close cyclist while the ego vehicle is moving fast
cyclist_close = (ot.detection_class == 1) & (ot.distance_m < 8.0)
near_miss_at_speed = (db.query.signal('Vehicle Speed Sensor') > 30) & cyclist_close
```

Crucially, a series can report not only *whether* a situation occurred, but *which*
entity was involved and when. A near-miss is traced back to the specific object
that triggered it, and when several entities trigger within the same window, each
is reported on its own. This turns fleet-scale recordings into precise,
entity-level findings without leaving the declarative API.

## The Vision

Impulse connects Silver layer measurement data to Gold layer analytical outputs through a single,
declarative, domain-friendly API.
It replaces scattered ad-hoc pipelines with a standardized workflow that scales from a single measurement file to
petabytes of sensor data -- while keeping the analytical logic readable and maintainable by the
domain experts who understand the data best.
