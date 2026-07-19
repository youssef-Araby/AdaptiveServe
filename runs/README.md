# Run Namespaces

Experiment outputs are separated by status so completed evidence, planned work,
and historical artifacts cannot be mistaken for one another.

| Namespace | Status | Contents |
| --- | --- | --- |
| [`p0/`](p0/) | Completed evidence | Corrected 2026-07-16 P0 run, including C0-C6 outputs, joined datasets, CV results, figures, and provenance anchors |
| [`longbench16_24k/`](longbench16_24k/) | Planned; no results yet | Reserved output namespace for the 16-task, 3,750-example, 24K-token evaluation defined in [`dataset.md`](../dataset.md) |
| [`legacy/`](legacy/) | Superseded | Pre-P0 and forensic artifacts retained only for reproducibility and historical inspection |

New experiments must use their own namespace. They must not overwrite
`p0/` artifacts or reuse its completion markers.
