# Cs117-Benchmark

## Dataset Setup — BDD100K subset

We use a fixed subset of **BDD100K** (seed 42): 5,000 train and 1,000 val images from the official train split, and 1,000 held-out test images from the official val split. The file lists are in `splits/*.txt`. Everyone must use the same lists.

Only one person builds the subset. Everyone else uses the shared zip.

#### Build the subset (once)

1. Download the Kaggle zip (about 7.6 GB):

```bash
curl -L -o ~/Downloads/solesensei_bdd100k.zip https://www.kaggle.com/api/v1/datasets/download/solesensei/solesensei_bdd100k
```

2. Run the script from the repo root (needs `pip install -r requirements.txt`):

```bash
python -m src.prepare_subset ~/Downloads/solesensei_bdd100k.zip
```

This writes `data/bdd100k_subset.zip` (about 440 MB, git-ignored).

3. Upload `bdd100k_subset.zip` to Google Drive at `MyDrive/BDD100K_Project/bdd100k_subset.zip` and share the folder with the team.

#### Use the subset (everyone, in Colab)

The notebook mounts Drive, unzips the subset to local Colab disk, and calls `load_split(...)`. Nobody else needs the full dataset.

Labels are BDD100K 2018 format. Category names are `bike` and `motor`, not COCO's `bicycle` and `motorcycle`. Map them before scoring.

## Local Development Setup (VS Code)

#### 1. Create a Virtual Environment

From the project root, create a Python virtual environment:

```bash
python3 -m venv .venv
```

#### 2. Activate the Virtual Environment

**macOS / Linux:**

```bash
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
.venv\Scripts\Activate.ps1
```

#### 3. Install Dependencies

Upgrade pip and install the project dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Configure VS Code

1. Open the Command Palette (`Ctrl+Shift+P` on Windows/Linux or `Cmd+Shift+P` on macOS).
2. Search for **Python: Select Interpreter**.
3. Select the Python interpreter inside `.venv`.

This ensures VS Code uses your project's virtual environment.


#### 7. Reusing the Environment

For future development sessions, activate it again:

```bash
source .venv/bin/activate
```

## Project Structure (Example. Not Final)

```text
bdd100k-benchmark/
│
├── notebooks/
│   └── main.ipynb              # Main Google Colab notebook
│
├── src/
│   ├── dataset.py              # Load images, annotations, metadata
│   ├── class_mapping.py        # Shared BDD100K / COCO mapping
│   ├── models.py               # YOLO11 and RT-DETR wrappers
│   ├── evaluation.py           # mAP, precision, recall
│   ├── benchmark.py            # GPU latency, FPS, warm-up
│   ├── experiments.py          # Run model/config comparisons
│   └── visualization.py        # Plots and example predictions
│
├── configs/
│   └── experiment.yaml         # Shared experiment configuration
│
├── splits/
│   ├── development.txt         # Development image filenames
│   └── held_out.txt            # Held-out evaluation filenames
│
├── results/
│   ├── metrics.csv             # Evaluation results
│   └── plots/                  # Generated figures
│
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```
