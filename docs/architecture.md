# System Architecture

## Overview

The Retail Shelf Monitoring System is built with **Clean Architecture** principles, separating business logic from infrastructure concerns. Data flows inward through the dependency rule — outer layers depend on inner layers, never the reverse.

---

## Layer Diagram

```mermaid
flowchart TD
    UI["🖥️ PySide6 UI\n(frameworks/ui)"]
    CLI["⌨️ CLI\n(cli.py)"]
    Container["🔧 DI Container\n(container.py)"]
    Adaptors["🔌 Adaptors\n(YOLODetector, SORTTracker,\nSKURecognizer, SQLiteRepo...)"]
    Frameworks["⚙️ Frameworks\n(Config, DB, ModelRegistry,\nInference Engines)"]
    Usecases["📐 Use Cases\n(Detection, Grid, Compliance,\nAlerts, Metrics...)"]
    Entities["🏛️ Entities\n(Frame, Detection, SKU,\nPlanogram, Alert, CellState)"]

    UI --> Container
    CLI --> Container
    Container --> Adaptors
    Container --> Frameworks
    Adaptors --> Usecases
    Frameworks --> Usecases
    Usecases --> Entities
    Adaptors --> Entities
```

---

## Real-Time Processing Pipeline

```mermaid
flowchart LR
    VideoSource["📹 Video Source\n(file / webcam)"]
    Capture["CaptureThread\nOpenCV frame read"]
    Keyframe["KeyframeSelector\nDiff threshold filter"]
    Inference["InferenceThread\nYOLO + SORT + SKU"]
    Grid["GridDetector\nDBSCAN clustering"]
    Aligner["ShelfAligner\nHomography → shelf_id"]
    Compliance["CellStateComputation\nOK/EMPTY/MISPLACED"]
    Consensus["TemporalConsensus\nMin N consecutive frames"]
    Alerts["AlertGeneration\nOOS + Misplacement rules"]
    Store["AlertStore\nMemory / Redis"]
    UI2["UI Dashboard\nAnnotated frames + alerts"]

    VideoSource --> Capture
    Capture --> Keyframe
    Keyframe --> Inference
    Inference --> Grid
    Grid --> Aligner
    Aligner --> Compliance
    Compliance --> Consensus
    Consensus --> Alerts
    Alerts --> Store
    Store --> UI2
```

---

## Model Registry – Operating Modes

```mermaid
flowchart TD
    Start["Application Start"]
    Check["ModelRegistry.scan()"]
    B["BASELINE MODE\nStock yolo11n.pt\nDetection only"]
    P["POSITION-ONLY MODE\nFine-tuned detector\nOK / EMPTY"]
    F["FULL MODE\nDetector + Embedding\n+ FAISS Index\nOK/EMPTY/MISPLACED + SKU ID"]

    Start --> Check
    Check -->|"no .pt files"| B
    Check -->|"product_detector.pt\nonly"| P
    Check -->|"all 7 contract files"| F
```

---

## Training Pipeline

```mermaid
flowchart TD
    Data1["SKU-110K Dataset\n(auto-downloaded)"]
    Data2["Custom Roboflow\n(optional)"]
    Stage1["Stage 1: Train Detector\nYOLOv11n fine-tune\n100 epochs on Colab GPU"]
    Stage2["Stage 2: Train Embedding\nMobileNetV3 + TripletLoss\n50 epochs on Colab GPU"]
    Stage3["Stage 3: Build FAISS Index\nEncode reference crops\n→ sku_index.faiss"]
    Stage4["Stage 4: ONNX Export\nParity check < 1e-4\n→ shelf_models.zip"]
    UserCopy["User copies\nshelf_models.zip\n→ models/"]

    Data1 --> Stage1
    Data2 --> Stage1
    Stage1 --> Stage2
    Stage2 --> Stage3
    Stage3 --> Stage4
    Stage4 --> UserCopy
```

---

## File Structure Detail

```mermaid
graph LR
    Root["DL-PROJECT/"]
    SM["shelf_monitor/"]
    EN["entities/"]
    UC["usecases/"]
    AD["adaptors/"]
    FR["frameworks/"]
    TR["training/"]
    TS["tests/"]
    MO["models/"]
    SC["scripts/"]

    Root --> SM
    Root --> TR
    Root --> TS
    Root --> MO
    Root --> SC
    SM --> EN
    SM --> UC
    SM --> AD
    SM --> FR

    EN --> e1["common.py\nsku.py\ndetection.py\nplanogram.py\nframe.py\nalert.py"]
    UC --> u1["cell_state_computation.py\ntemporal_consensus.py\nalert_generation.py\ndetection_processing.py\nstream_processing.py\nmetrics.py"]
    UC --> u2["grid/\nshelf_aligner/\ninterfaces/"]
    AD --> a1["ml/yolo_detector.py\nml/sku_recognizer.py\ntracking/sort.py\nkeyframe_selector.py\nrepositories/"]
    FR --> f1["config.py\ndatabase.py\nmodel_registry.py\nlogging_config.py\npytorch_models/\ninference_engines/\nui/"]
```

---

## Dependency Injection (container.py)

```mermaid
flowchart TD
    AppContainer["ApplicationContainer\n(dependency_injector\nDeclarativeContainer)"]
    Cfg["config\nAppConfig.load()"]
    DB["database\nSQLAlchemy async engine"]
    Reg["model_registry\nModelRegistry"]
    Det["detector\nYOLODetector"]
    Rec["sku_recognizer\nSKURecognizer"]
    Trk["tracker\nSORTTracker"]
    KF["keyframe_selector\nKeyframeSelector"]
    PlRepo["planogram_repo\nSQLitePlanogramRepo"]
    AlRepo["alert_store\nMemoryAlertStore"]
    Grid2["grid_detector\nGridDetector"]
    Aln["shelf_aligner\nShelfAligner"]
    SP["stream_processor\nStreamProcessor"]

    AppContainer --> Cfg
    AppContainer --> DB
    AppContainer --> Reg
    Reg --> Det
    Reg --> Rec
    Cfg --> Trk
    Cfg --> KF
    DB --> PlRepo
    Det --> SP
    Rec --> SP
    Trk --> SP
    KF --> SP
    Grid2 --> SP
    Aln --> SP
    PlRepo --> SP
    AlRepo --> SP
```

---

## UI Threading Model

```mermaid
sequenceDiagram
    participant CT as CaptureThread
    participant IT as InferenceThread
    participant AT as AlertAnalysisThread
    participant ALT as AlertThread
    participant MW as MainWindow (Qt)

    CT->>IT: frame (via QQueue, drop-oldest)
    IT->>MW: annotated_frame (signal)
    IT->>AT: cell_states (signal)
    AT->>ALT: potential_alerts (signal)
    ALT->>MW: alert_list (signal)
    MW->>MW: update_ui()
```

---

## Alert State Machine

```mermaid
stateDiagram-v2
    [*] --> UNKNOWN
    UNKNOWN --> OK: product detected N frames
    UNKNOWN --> EMPTY: no product N frames
    OK --> EMPTY: product gone N frames
    OK --> MISPLACED: wrong SKU N frames
    EMPTY --> OK: product detected N frames
    MISPLACED --> OK: correct SKU N frames
    MISPLACED --> EMPTY: product removed N frames
```
