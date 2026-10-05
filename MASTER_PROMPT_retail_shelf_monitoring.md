# MASTER PROMPT: Retail Shelf Monitoring System (end-to-end)

> Paste everything below the line into Claude Code (or any coding agent) opened at
> `D:\Degree_GLS\sem 7\DLL\DL-PROJECT`. Put the reference repo next to it, e.g. at
> `D:\Degree_GLS\sem 7\DLL\DL-PROJECT\reference\Retail-Shelf-Monitoring-main\`
> (unzip it there) so the agent can read it.

---

## ROLE
You are a senior ML/CV engineer. Build a complete, runnable, end-to-end retail shelf
monitoring system in the current directory. Work autonomously: plan, write every file,
run it, fix errors, and stop only when the acceptance criteria at the bottom pass.
Do not ask questions; record assumptions in README.md.

Environment: Windows 10/11, Python 3.10 or 3.11, NVIDIA GPU optional (CPU fallback is
mandatory), no Docker required, no admin rights required. Use `pathlib`, never hard-coded
absolute paths. Use `python -m venv .venv`.

## REFERENCE
A reference implementation is in `./reference/Retail-Shelf-Monitoring-main/`
(Alijanloo/Retail-Shelf-Monitoring, MIT licence). READ IT FIRST: README.md,
docs/technical_report.md, docs/project_tree.md, docs/shelf_aligner.md,
config.example.yaml, and the modules under `retail_shelf_monitoring/`.
Re-implement the same system in your own code in a NEW package; do not copy files
verbatim and do not edit the reference. Keep its good ideas, fix its weaknesses
(listed below), and add the parts it is missing.

### What the reference does (keep this design)
- Clean Architecture layers: `entities` (pydantic models), `usecases` (business logic +
  interfaces), `adaptors` (ML, repos, tracking), `frameworks` (config, DB, logging,
  inference engines, UI), plus a dependency-injection `container.py`.
- Pipeline: video/CCTV stream -> keyframe selection -> shelf identification/alignment
  (ORB/SIFT + RANSAC homography against stored reference shelf images) -> YOLO product
  detection -> SKU recognition (MobileNetV3 embedding + FAISS nearest neighbour) ->
  grid mapping (DBSCAN on box y-centres into rows, sorted left-to-right) -> compare to
  a stored planogram -> cell states OK / EMPTY(OOS) / MISPLACED / UNKNOWN -> SORT tracking
  for non-keyframes -> temporal consensus (alert only after N consecutive frames) ->
  alerts stored in Redis -> desktop UI (PySide6) with staff confirm/dismiss.
- Multi-threaded app: capture thread, inference thread, alert-analysis thread, alert thread.
- Entities: SKU, Planogram(grid of rows/items with bbox + sku_id), Detection, Frame,
  Alert(alert_id, shelf_id, row_idx, item_idx, alert_type, expected_sku, detected_sku,
  first_seen, last_seen, confirmed, confirmed_by, dismissed, evidence_paths,
  consecutive_frames).
- Config via YAML (`ml`, `sku_detection`, `grid`, `database`, `redis`, `logging`).

### Weaknesses in the reference you MUST fix
1. Missing assets: it ships no weights, no FAISS index, no dataset, no training code.
   -> Add full training/indexing scripts (see Training section) and a demo mode that
   works with zero downloads.
2. Hard-coded shelf: `stream_processing.py` forces `shelf_id = "shelf_517"` and has the
   aligner commented out; keyframe interval hard-coded to 30. -> Make shelf alignment
   real and configurable, with a single-shelf mode that skips alignment.
3. Wrong import (`from ast import Dict`), `oos_count` always 0 while a separate
   `empty_count` exists, doc/code folder mismatches (grid, shelf_aligner live under
   usecases but docs say adaptors/vision). -> Clean, consistent structure and naming,
   one canonical set of cell-state names.
4. Windows-hostile dependencies: `faiss-gpu`, TensorRT, OpenVINO, mandatory Docker
   Postgres + Redis. -> Use `faiss-cpu`; ONNX Runtime / PyTorch via ultralytics as the
   default inference engine; make OpenVINO/TensorRT optional extras behind the same
   interface; SQLite (SQLAlchemy) default with optional PostgreSQL; in-process alert
   store default with optional Redis. Selected by config, same interfaces.
5. Tests only cover parts. -> Cover every use case and adaptor, plus one real
   end-to-end test on synthetic data.

## NO-GPU PROFILE (the developer has no local GPU)
The reference repo ships NO model weights, so there is no pretrained model to reuse. Design
everything so that it runs on CPU, and so that heavy training can be done elsewhere:
- Inference must run in real time-ish on CPU: default to `yolo11n` (config switch to `s`),
  `imgsz` 640 (allow 480/416), ONNX Runtime or OpenVINO export, keyframe interval >= 15,
  SORT between keyframes. Report CPU ms/frame in docs/results.md.
- Baseline weights: auto-download stock COCO `yolo11n.pt` via ultralytics when no custom
  weights exist, and log a warning that accuracy on dense shelves will be limited.
- CPU-feasible training path (default): fine-tune `yolo11n` on the small custom/Roboflow set
  and on a configurable SKU-110K SUBSET (e.g. `--max-images 500`), few epochs, small imgsz,
  `workers` low, `device=cpu`. Never attempt full SKU-110K training on CPU.
- GPU-offload path: generate `training/colab_train.ipynb` (also runnable on Kaggle) that
  clones the repo, mounts/uploads the dataset, runs the full two-stage training on a free
  cloud GPU, and exports `best.pt` + `best.onnx` to download into `outputs/models/`.
- Replace `faiss-gpu`/`torch+cu118` instructions with CPU wheels everywhere
  (`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`).
- `device: auto` in config: use CUDA if available, else CPU, with no code changes.

## PRODUCT GOAL
Given a shelf camera stream, video file, or still images, the system must:
1. Detect every product on the shelf (YOLO11, single class "product").
2. Identify each product's SKU by embedding + FAISS lookup (fall back to "unknown_sku").
3. Build the shelf grid and compare it with the stored planogram.
4. Report per cell: OK, EMPTY (out-of-stock), MISPLACED, UNKNOWN.
5. Raise an alert only when a condition persists for N consecutive keyframes.
6. Persist alerts + evidence images; let staff confirm/dismiss in the UI.
7. Report shelf metrics: product count, fill %, planogram compliance %, OOS count.

## TECH STACK
Python, PyTorch + torchvision, ultralytics (YOLO11n/s; allow `yolo11m` via config),
OpenCV, scikit-learn (DBSCAN/KMeans), faiss-cpu, onnxruntime, pydantic v2,
dependency-injector, SQLAlchemy 2, PyYAML, PySide6 (desktop UI), loguru or stdlib logging,
pytest + pytest-cov, black/isort/mypy config in pyproject.toml. Optional extras:
openvino, tensorrt, psycopg2-binary, redis.

## REPOSITORY LAYOUT (create exactly; package name `shelf_monitor`)
```
DL-PROJECT/
├─ README.md, pyproject.toml, requirements.txt, .gitignore, config.example.yaml
├─ reference/                      (provided; read-only)
├─ shelf_monitor/
│  ├─ entities/        common.py sku.py planogram.py detection.py frame.py alert.py
│  ├─ usecases/
│  │  ├─ interfaces/   inference_model.py repositories.py tracker.py alert_store.py
│  │  ├─ grid/         clustering.py grid_detector.py
│  │  ├─ shelf_aligner/ feature_matcher.py homography.py shelf_aligner.py
│  │  ├─ planogram_generation.py   detection_processing.py   cell_state_computation.py
│  │  ├─ temporal_consensus.py     alert_generation.py       stream_processing.py
│  │  └─ metrics.py                # fill %, compliance %, counts
│  ├─ adaptors/
│  │  ├─ ml/           yolo_detector.py sku_recognizer.py (embedding + FAISS)
│  │  ├─ tracking/     sort.py
│  │  ├─ repositories/ sqlite_planogram_repo.py postgres_planogram_repo.py
│  │  │                memory_alert_store.py redis_alert_store.py
│  │  └─ keyframe_selector.py
│  ├─ frameworks/
│  │  ├─ config.py database.py logging_config.py exceptions.py
│  │  ├─ pytorch_models/embedding_net.py     # MobileNetV3-Large -> L2-normalised 256-d
│  │  ├─ inference_engines/ onnx_runtime.py (default) openvino_model.py tensor_rt.py (optional)
│  │  └─ ui/ main_window.py threads/{capture,inference,alert_analysis,alert}_thread.py
│  │         widgets/{video_widget,alert_panel,planogram_view}.py resources/styles.qss
│  ├─ container.py     # dependency injection wiring
│  ├─ cli.py           # headless commands (see below)
│  └─ __main__.py
├─ training/
│  ├─ prepare_sku110k.py        # SKU-110K -> YOLO format (+ instructions if not downloaded)
│  ├─ prepare_roboflow.py       # merge Roboflow YOLO export (custom store images) into dataset
│  ├─ train_detector.py         # stage 1 pretrain (SKU-110K) -> stage 2 fine-tune (custom)
│  ├─ train_embedding.py        # metric learning (triplet/ArcFace) on product crops
│  ├─ build_sku_index.py        # crops -> embeddings -> FAISS index + sku_labels.json
│  ├─ export_onnx.py            # best.pt -> ONNX (+ optional OpenVINO IR)
│  └─ evaluate.py               # mAP50, mAP50-95, P/R, PR curve, Top-1/Top-5 SKU accuracy
├─ scripts/  make_demo_data.py  run_demo.py
├─ data/     (gitignored large files; data/README.md explains layout)
├─ tests/    unit/{entities,usecases,adaptors,frameworks}/  integration/test_end_to_end.py
├─ docs/     architecture.md  technical_report.md  results.md  (+ mermaid diagrams)
├─ notebooks/ exploration.ipynb
└─ outputs/  models/ runs/ evidence/ reports/
```

## DETAILED REQUIREMENTS

### 1. Detection
- `YoloDetector` implements `InferenceModel` (`predict(frame) -> list[Detection]`);
  confidence/NMS/device/imgsz/max_det from config (`max_det >= 300` for dense shelves).
- Default engine: ultralytics PyTorch or ONNX Runtime; engines selectable via config.
- Graceful CPU fallback; clear error if weights are missing, pointing to the demo mode.

### 2. SKU recognition
- `EmbeddingNet`: MobileNetV3-Large backbone, global pool, Linear(256), L2-normalise.
- `SkuRecognizer`: crop -> resize 224 -> embedding -> FAISS (`faiss-cpu`, inner product)
  top-k with a similarity threshold below which the SKU is `unknown_sku`.
- Batch crops for speed. Cache embeddings per tracker id so each tracked product is
  re-identified only every K frames.

### 3. Shelf alignment (make it real)
- Store reference shelf images + precomputed ORB/SIFT features.
- Match each keyframe against references (Lowe ratio 0.75, RANSAC reproj 5.0, max 5000
  features); validate homography (det in [0.1, 10], perspective terms < 0.002, condition
  number < 10); confidence = inlier ratio; accept above `min_alignment_confidence` (0.3).
- Warp detections into the reference coordinate system.
- Config flag `single_shelf_mode: true` to skip alignment and use a fixed `shelf_id`.

### 4. Grid + planogram
- `GridDetector`: DBSCAN (default, `eps` 15, `min_samples` 2; KMeans option) on y-centres
  -> rows sorted top-to-bottom; items sorted left-to-right.
- `PlanogramGeneration`: from a reference image run detection + SKU recognition, build
  `{shelf_id, rows:[{row_idx, items:[{item_idx, bbox, sku_id}]}]}`, store via repository.
- `match_grids`: position tolerance (default 1) -> matches, mismatches, missing.

### 5. Cell states, consensus, alerts
- States (single canonical enum): OK, EMPTY, MISPLACED, UNKNOWN.
  `summary = {ok, empty, misplaced, unknown, total_cells, compliance_pct, fill_pct}`.
- `TemporalConsensus`: per-cell counters; alert only after `min_consecutive_frames`
  (default 3); clear after M clean frames; handle occlusion and motion blur.
- `AlertGeneration`: create/update alerts, de-duplicate per (shelf, row, item, type), save
  an evidence crop/frame to `outputs/evidence/`, expose confirm/dismiss.

### 6. Stream processing
- `StreamProcessing`: keyframe every `keyframe_interval` frames (configurable) or when an
  intensity-difference threshold triggers; on non-keyframes propagate boxes with SORT
  (Kalman + Hungarian IoU matching). Async-friendly; no hard-coded ids.

### 7. Desktop UI (PySide6) + threads
- Live video with detection boxes, grid overlay and per-cell colour (green OK, red EMPTY,
  amber MISPLACED, grey UNKNOWN).
- Alert panel: list, filter, confirm/dismiss with staff id, evidence preview.
- Planogram view and a metrics strip (fill %, compliance %, OOS count, FPS).
- Four worker threads as in the reference, communicating via Qt signals/queues.
- Menu: open video / webcam / RTSP URL, load planogram, create planogram from image.

### 8. Headless CLI (`python -m shelf_monitor <command>`)
`make-planogram --image ref.jpg --shelf-id S1`, `analyze-image --image x.jpg --shelf-id S1`,
`analyze-video --video v.mp4 --shelf-id S1 --out outputs/runs/`, `ui`, `demo`.
Every command writes annotated output plus a JSON report.

### 9. Config
`config.example.yaml` (copy to `config.yaml`): `app`, `database` (default sqlite file;
postgres optional), `alerts` (backend: memory|redis), `ml`, `sku_detection`, `grid`,
`aligner`, `stream` (keyframe_interval, single_shelf_mode), `consensus`, `logging`.
Validate with pydantic; environment variables override.

### 10. Training (the reference has none; this is the deep-learning core of the project)
- Detector: stage 1 pretrain YOLO11 on SKU-110K (imgsz 640, dense-scene augmentation,
  `max_det` 300+); stage 2 fine-tune from stage-1 weights on custom local-store images
  annotated in Roboflow (YOLO export), lower LR, optional frozen backbone layers.
  Log metrics, keep best.pt in `outputs/models/`. Auto-reduce batch on CUDA OOM.
  Configurable epochs/batch/imgsz/seed.
- Embedding: train `EmbeddingNet` with triplet or ArcFace loss on product crops
  (use the SHAPE product dataset from github.com/rokopi-byte/shelf_management if available,
  otherwise crops from the custom data); report Top-1 / Top-5 retrieval accuracy.
- Ablation to include in `docs/results.md`: (a) COCO-pretrained only, (b) SKU-110K
  pretrain only, (c) SKU-110K pretrain + custom fine-tune. Table of mAP50, mAP50-95,
  precision, recall, inference ms/frame. Add training curves and sample predictions.
- Data scripts must handle missing datasets gracefully: print exact download steps and
  never crash the demo.

### 11. Zero-download demo mode (must work offline)
- `scripts/make_demo_data.py` synthesises shelf images (coloured product boxes on
  shelf rows, with random gaps and swapped products) and a short video where products
  disappear/move over time, with ground-truth labels.
- Demo uses a quickly trained tiny detector/embedding on the synthetic data (or a
  classical fallback detector) so `python -m shelf_monitor demo` runs end to end:
  build planogram -> process video -> raise EMPTY and MISPLACED alerts -> write
  annotated video, alerts JSON, and metrics. Clearly label demo results as synthetic.

### 12. Quality bar
Type hints, docstrings, small functions, no globals, structured logging, reproducible
seeds, all paths via config/pathlib, black/isort clean, mypy-friendly. Every external
dependency (DB, Redis, FAISS, engine) sits behind an interface and is mockable.

## FINAL REQUIREMENTS (HIGHEST PRECEDENCE: override anything above that conflicts)

### A. Division of labour
- The developer will do the REAL training on Google Colab or Kaggle. You must NOT run full
  real training or download SKU-110K (13.6 GB) locally. Locally you only run smoke tests:
  synthetic data, CPU, <= 2 epochs, tiny image counts, a few minutes at most.
- The developer's only manual tasks: (1) run the notebook, (2) download one zip,
  (3) unzip it into `models/`, (4) run the app. NO code or config edits afterwards.

### B. MODEL CONTRACT (fixed folder, fixed filenames, auto-loaded)
Folder: `models/` in the project root (override with env `SHELF_MODELS_DIR` or config
`models_dir`). It ships EMPTY except `models/README.md` and `.gitkeep`. Smoke/demo models
must never be written to `models/`; put them in `outputs/demo_models/` and let only the
`demo` command and the tests point there.

Exact files produced by the notebook and copied into `models/`:

| File | Required | What it is |
|---|---|---|
| `models/product_detector.pt` | yes | fine-tuned YOLO11 product detector (ultralytics weights) |
| `models/product_detector.onnx` | recommended | same detector exported for fast CPU inference (preferred when present) |
| `models/sku_embedding.pt` | for SKU recognition | EmbeddingNet state_dict |
| `models/sku_embedding.onnx` | optional | ONNX export of the embedding network |
| `models/sku_index.faiss` | for SKU recognition | FAISS index of reference SKU embeddings |
| `models/sku_labels.json` | with the index | maps index row -> `{sku_id, name}` |
| `models/model_manifest.json` | yes | all metadata the app needs (below) |

The notebook's last step bundles exactly these into ONE file, `shelf_models.zip`, whose
contents are the files above at the zip root. The developer unzips it into `models/`.

`model_manifest.json` (written by the notebook, validated with pydantic by the app):
`schema_version`, `created_at`, `synthetic` (bool), `detector{file, onnx_file, imgsz,
conf_default, iou_default, max_det, class_names}`, `embedding{file, onnx_file, input_size,
dim, mean, std, backbone}`, `index{file, labels_file, metric, dim, num_vectors,
similarity_threshold}`, `metrics{map50, map50_95, precision, recall, top1, top5}`,
`training{epochs_stage1, epochs_stage2, dataset_notes}`.
The application reads image size, thresholds, normalisation, embedding dim and class names
from the manifest and from the loaded model. NOTHING about the model is hard-coded.

`ModelRegistry` (frameworks layer, behind an interface):
- Resolves paths from the contract, validates the manifest, checks the embedding dim equals
  the FAISS index dim, prefers `.onnx` over `.pt` unless config says otherwise, uses
  `device: auto`.
- Graceful degradation with a loud one-line status banner in CLI/UI/logs:
  1. no detector files -> stock `yolo11n.pt` (BASELINE mode, warning about accuracy);
  2. detector OK but no embedding/index -> POSITION-ONLY mode (OK/EMPTY states only,
     MISPLACED disabled, SKUs shown as `unknown_sku`);
  3. everything present -> FULL mode.
- `python -m shelf_monitor check-models` prints a table: each contract file, found/missing,
  size, mode (BASELINE / POSITION-ONLY / FULL), manifest summary, and runs a 1-image
  inference + 1 embedding + 1 FAISS query self-test. Exit code non-zero on a corrupt file.
- Planogram creation, analyze-image, analyze-video, UI and demo all use the registry.
  Re-enrolling new SKUs locally on CPU: `python -m shelf_monitor enroll-skus --crops <dir>`
  (class-per-folder) rebuilds `sku_index.faiss` + `sku_labels.json` from the loaded
  embedding model, no retraining.

### C. `training/colab_train.ipynb` (create it, complete, runnable on Colab AND Kaggle)
The notebook is THIN: all logic lives in `training/*.py` (already smoke-tested locally);
cells just configure and call them. Required cells, in order:
0. Title + short instructions (what to upload, expected runtime, what to download).
1. CONFIG cell: `SMOKE_TEST=False`, `EPOCHS_STAGE1`, `EPOCHS_STAGE2`, `EMBED_EPOCHS`,
   `IMGSZ=640`, `BATCH='auto'`, `MODEL_SIZE='yolo11s'` (n/s/m), `SKU110K_FRACTION=1.0`,
   `CUSTOM_DATA` (Roboflow API key/workspace/project/version OR path to an uploaded YOLO
   zip, optional), `SKU_CROPS` (optional class-per-folder of reference product images),
   `SEED`.
2. Environment detection (Colab / Kaggle / other), GPU check (fail fast with a clear
   message if no GPU and not SMOKE_TEST), `pip install` from `requirements-train.txt`.
3. Get the project code: upload `project_for_colab.zip` (made by
   `python scripts/package_for_colab.py`) or read it from a Drive / Kaggle input path.
4. Persistence: mount Drive (Colab) or use `/kaggle/working`; checkpoints saved there so a
   disconnected session can resume (`resume=True`).
5. Data: SKU-110K via ultralytics auto-download (honouring `SKU110K_FRACTION`); optional
   custom Roboflow/zip data merged by `training/prepare_roboflow.py`.
6. Stage 1: pretrain detector on SKU-110K.
7. Stage 2: fine-tune on custom data (skipped cleanly if none provided).
8. Evaluate -> `metrics.json`, PR curve and sample predictions saved; optional ablation
   (COCO-only vs SKU-110K vs SKU-110K+custom) behind `RUN_ABLATION`.
9. Embedding: train with ArcFace/triplet on `SKU_CROPS` if given; otherwise a documented
   self-supervised fallback on crops cut from detections.
10. Build FAISS index + `sku_labels.json` from `SKU_CROPS` (or from detected crops).
11. Export ONNX (detector, embedding) and run a parity check against the PyTorch outputs.
12. Validate artifacts exactly as `check-models` would (load, shapes, dim match, query).
13. Write `model_manifest.json` and bundle `shelf_models.zip` with the exact contract names.
14. Download: `google.colab.files.download('shelf_models.zip')` on Colab; on Kaggle print
    where the file is in the Output tab. Print the "NEXT STEPS" text for the developer.
Notebook QA you must perform locally: valid nbformat (`nbformat.validate`), every code
cell passes `ast.parse` (after stripping `!`/`%` lines), no hard-coded local paths, and
the same `training/*.py` entry points the notebook calls are executed end-to-end once on
synthetic data with `SMOKE_TEST=True` (CPU, minutes) to prove the cells' logic works.
Also write `requirements-train.txt` (pinned) and `scripts/package_for_colab.py`.

### D. Drop-in proof (required test)
Take the artifacts produced by the local smoke run, rename/copy them to the contract names
in a temp directory, set `SHELF_MODELS_DIR` to it, and run `check-models`, `analyze-image`
and `analyze-video` with no other change. Add this as an integration test, plus unit tests
for `ModelRegistry` (missing files, wrong dim, corrupt manifest, each of the 3 modes).

### E. Runbook and final message
Write `docs/RUNBOOK.md` and a README section titled "Train -> Download -> Place -> Run" with
exact steps and commands for Windows (cmd/PowerShell). Your FINAL message to the developer
must contain ONLY these four numbered steps with exact commands, nothing else:
1. Train on Colab/Kaggle  2. Download the model  3. Put it in `models/`  4. Run the system
(include the venv/install command once, `check-models`, `ui`, and a CLI example).

## EXECUTION STEPS (do all of them)
1. Read the reference thoroughly; write `docs/architecture.md` summarising what you keep
   and what you change (mermaid diagrams for the pipeline and layer dependencies).
2. Create the venv; write requirements.txt and pyproject.toml (extras: `openvino`,
   `tensorrt`, `postgres`, `redis`, `dev`); install; verify torch/CUDA and faiss import.
3. Implement entities -> interfaces -> use cases -> adaptors -> frameworks -> container
   -> CLI -> UI, writing unit tests alongside each layer.
4. Implement training scripts; run a 2-epoch smoke train on the synthetic data.
5. Run the demo end to end; save outputs under `outputs/runs/`.
6. Run `pytest --cov`; fix everything until green.
7. Do NOT run real training. Finish the training scripts, the notebook and the packaging
   script, validate the notebook (section C), and prove the drop-in contract (section D).
8. Launch the UI once (offscreen/headless check is fine) to confirm it starts.
9. Write README.md (overview, architecture, Windows setup, data prep, training,
   inference, CLI + UI usage, results table, limitations, future work, credits to the
   reference repo and datasets: SKU-110K, SHAPE/SHARD).
10. Write `docs/technical_report.md` in a form suitable for a deep-learning lab submission.

## ACCEPTANCE CRITERIA
- `python -m venv .venv && pip install -e .[dev]` works on a clean Windows machine with
  no Docker, no TensorRT, no OpenVINO, no external DB.
- `python -m shelf_monitor demo` runs offline and produces an annotated video, an alerts
  JSON containing at least one EMPTY and one MISPLACED alert, and a metrics report.
- `python -m shelf_monitor ui` starts the desktop app.
- `pytest` passes, including the end-to-end test; coverage >= 70% on usecases/.
- Training scripts and the notebook's entry points run end to end on synthetic data
  (smoke only); real training is left to the developer's Colab/Kaggle run;
  `docs/results.md` holds the ablation table template (synthetic smoke numbers clearly
  labelled; the notebook writes the real `metrics.json` the developer pastes in).
- No hard-coded shelf ids, absolute paths, or credentials anywhere.
- After the developer unzips `shelf_models.zip` into `models/`, `check-models` reports FULL
  mode and the UI/CLI use the models with zero code or config changes.
- `training/colab_train.ipynb` validates, and `shelf_models.zip` has exactly the contract names.
- Final message: ONLY the four steps in section E (no summary, no metrics, no extras).
