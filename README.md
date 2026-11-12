# Cs117-Benchmark

## Dataset Setup — BDD100K

This project uses the **BDD100K (Berkeley DeepDrive 100K)** autonomous driving dataset to evaluate YOLO11 and RT-DETR object detection models.

### 1. Download BDD100K

Download the dataset from the official Berkeley DeepDrive website:

**[BDD100K Official Download Page](http://bdd-data.berkeley.edu/download.html)**

Required downloads:

- **100K Images:** Contains the driving images used for training and evaluation.
- **Labels:** Contains ground-truth object detection bounding boxes, class labels, and environmental metadata.

We use the original 2018 annotations because the 2020 Detection Labels download returned JPG files instead of the expected JSON annotations.

### 2. Organize the Dataset

After downloading and extracting the files, organize them locally into the following structure before uploading to Google Drive:

```text
BDD100K_Project/
└── bdd100k_data/
    ├── images/
    │   ├── train/
    │   │   └── *.jpg
    │   └── val/
    │       └── *.jpg
    │
    └── labels/
        ├── train/
        │   └── *.json
        └── val/
            └── *.json
```

### 3. Upload to Google Drive

Upload the `BDD100K_Project` directory to the root of your Google Drive (`MyDrive`).

The dataset should be accessible at:

```text
MyDrive/BDD100K_Project/bdd100k_data/
```



## Local Development Setup (VS Code)

### 1. Create a Virtual Environment

From the project root, create a Python virtual environment:

```bash
python3 -m venv .venv
```

### 2. Activate the Virtual Environment

**macOS / Linux:**

```bash
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
.venv\Scripts\Activate.ps1
```

Once activated, your terminal should display `(.venv)`.

### 3. Install Dependencies

Upgrade pip and install the project dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure VS Code

1. Open the Command Palette (`Ctrl+Shift+P` on Windows/Linux or `Cmd+Shift+P` on macOS).
2. Search for **Python: Select Interpreter**.
3. Select the Python interpreter inside `.venv`.

This ensures VS Code uses your project's virtual environment.


### 7. Reusing the Environment

You only need to create the environment once.

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
