# strata-labeller

The semi-automatic labelling loop, on the catalog, with Label Studio as
the review tool. Label a small seed set, train a model, let it pre-label
the rest, then only correct what it got wrong. Each round the model
improves and there is less to fix.

```bash
uv add strata-labeller
```

Depends on `strata-labels`, `strata-catalog`, `strata-modelling`, typer
and the Label Studio SDK. Everything that knows Label Studio's shape stops
in one module; past it a sample is a catalog row and an annotation is a
`strata.labels` value.

## The loop

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────────┐
│  Hand-label a   │────▶│  Train a model   │────▶│  Push predictions    │
│  seed set       │     │  on a frozen     │     │  to Label Studio     │
│                 │     │  dataset version │     │  as pre-annotations  │
└─────────────────┘     └──────────────────┘     └──────────────────────┘
        ▲                                                    │
        │                                                    ▼
        │        export corrected labels           ┌──────────────────────┐
        └──────────── to the catalog ──────────────│  Review & correct in │
                                                   │  Label Studio        │
                                                   └──────────────────────┘
```

1. `prepare` converts a corpus into what the catalog holds, when it did not arrive that way
2. `ingest` registers files in the catalog, content-addressed, so re-running over a growing directory is safe
3. `push` scores everything unreviewed and sends the least confident to Label Studio with the model's guess attached
4. Review in Label Studio: confirm what is right, fix what is not
5. `export` pulls the corrections back into the catalog
6. `train` freezes a dataset version, materialises it, and fine-tunes from it
7. Repeat

The review queue is ordered by an active-learning strategy over each
prediction's confidences. Samples the model found nothing in are drawn as
a second pool in a declared share, so neither request crowds the other
out. Samples two people answered differently go first.

## Projects

A project is a directory holding everything belonging to one labelling
job. The catalog is the asset; the project is the work.

```
my-project/
├── project.toml        # label schema, collections, model
├── model.py            # optional: a model this project carries
├── datasets/           # materialised dataset versions
└── runs/               # runs, metrics and checkpoints
```

```toml
[label_config]
template = "image_classification"    # or text_classification, text_span, image_bbox
classes = ["cat", "dog"]
# multi_label = true      # span projects: one region may carry several labels
# overlapping = true      # span projects: two regions may intersect

[data]
type = "image"                       # a registered sample type
# source_root = "data/source"        # a corpus that needs converting first
# preparer = "eml"

# [[data.features]]                  # what the model is told besides the bytes
# name = "species"
# source = "label_set"               # or "metadata"
# ref = "plant-species"

[catalog]
name = "images"                      # which catalog on this host
collections = ["my_images"]          # which collections this job draws from

[model]
ref = "multilabel"                   # a registered name, or model.py:MyModel

[model.params]
num_epochs = 4

[model.fresh_params]                 # applied on a cold start
num_epochs = 10
```

The label set is authoritative for classes: the file seeds them once, and
`class add` extends them, append-only.

## Setup

A `config.toml` on the machine says where things are, nothing about a
job:

```toml
[label_studio]
url = "http://localhost:8080"

[catalog]
root = "catalog"

[modelling]
url = ""                             # a URL sends rounds to a GPU host
```

Credentials come from the environment: `LABEL_STUDIO_API_KEY`, and
`STRATA_MODELLING_TOKEN` for a remote host. `docker-compose.yml` at the
repository root starts a Label Studio for development.

```bash
auto-labeller new cats --class cat --class dog
cp -r /path/to/images/* projects/cats/data/raw/
auto-labeller ingest -p cats
auto-labeller init -p cats                 # creates the Label Studio project
```

Then label a seed set and run `train`, `push`, `export` in a loop.

## Commands

| `auto-labeller` | |
|---|---|
| `new` | scaffold a project |
| `templates` | list label config templates |
| `projects` | list projects and their counts |
| `class add` / `class list` | extend or inspect a project's classes |
| `prepare` | convert a corpus into what this project ingests |
| `ingest` | register files from the project's data root |
| `init` | create the Label Studio project and fill it |
| `push` | send unreviewed samples, least confident first |
| `export` | pull corrections back into the catalog |
| `train` | freeze a version, materialise it, train; `--job ID` reattaches to a remote round |
| `report` | training history, or one run in detail; `--json` for a chart or a script |
| `unskip` | return skipped samples to the queue |
| `relink` | repoint tasks at their current sample URLs, re-signing them |
| `catalog-check` | ask the blob server and the modelling host which catalog they serve |
| `import-rounds` | one-way migration from the pre-catalog format |

Most take `-p/--project`; all take `--config`.

## Decisions

Recorded in the umbrella repository's `docs/adr/`: ids mean nothing
outside their catalog (0008), disagreement is recorded (0009), the review
queue is two pools (0012), and Label Studio stops at the adapter (0013).

## Tests

```bash
uv run pytest packages/labeller
```
