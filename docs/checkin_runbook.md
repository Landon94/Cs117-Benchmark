# CS117 November 17 check-in: measured baseline runbook

This branch prepares a *runnable* measured baseline. It does not claim to have
run YOLO11 or RT-DETR against the actual BDD100K dataset. The assignment
requires a real measurement and the resulting JSON checked into GitHub.

## Procedure (Colab GPU or a local Ultralytics Python environment)

1. Use a runtime with Python, an appropriate PyTorch build, and enough RAM.
2. `pip install -r requirements-checkin.txt`
3. Have the BDD100K validation images in `images.zip` and original BDD100K
   detection annotations in `labels.zip` or an aggregate validation JSON. The
   labeling adapter accepts standard list-of-image label JSON and per-image JSON.
4. Run the **pretrained, unfine-tuned YOLO11m baseline** with the exact command:

```bash
python -m src.run_checkin \
  --images /content/drive/MyDrive/BDD100K_Project/bdd100k_data/images.zip \
  --labels /content/drive/MyDrive/BDD100K_Project/bdd100k_data/labels.zip \
  --model yolo11m.pt --split val --count 1000 --seed 42 \
  --manifest splits/checkin_val.txt --output results/baseline_val.json \
  --predictions-output results/baseline_predictions_val.json
```

5. Commit the *real* `results/baseline_val.json` and the selection manifest.
   The results file records the CLI command, split, image count, model, mAP,
   shared categories, evaluation limitations, and model inference-time summary.
6. If time permits, run the identical split with `--model rtdetr-l.pt` and
   different result output filenames. Use the **same selection manifest**.

## Scope and fairness

- Project question: accuracy-versus-speed tradeoff of YOLO11m and RT-DETR-L
  for driving-scene object detection, under a fixed validation budget.
- Accuracy: 101-point `mAP@0.5:0.95` over eight overlapping COCO/BDD classes:
  person, bicycle, car, motorcycle, bus, train, truck, and traffic light.
- Mapping: BDD `bike` -> COCO `bicycle`; BDD `motor` -> `motorcycle`.
  BDD `rider` and `traffic sign` have no equivalent in the pretrained COCO
  class labels and are excluded. Mention this limitation in the check-in.
- This is a **lightweight, auditable approximation** to COCO-style mAP. It does
  not implement COCO crowd/ignore labels or area-range metrics. It should not
  be labeled as an official COCOeval result.
- Timing is Ultralytics' **per-image inference time**, not full end-to-end
  camera latency. Keep the same resolution and device when comparing models.
- Results must be produced on real data. Never substitute synthetic fixture
  test output for the assignment's measured baseline.

## Check-in rubric mapping

- Item 1: research question and same `mAP@0.5:0.95` metric; describe main/fallback
  decision and real changes from the proposal.
- Item 2: copy the *actual* `measurements.map50_95` value and split/count from
  `results/baseline_val.json`; cite the exact command and file path. Supply
  the proposal's expected number separately (do not invent one).
- Item 3: a table with columns `Model`, `mAP@0.5:0.95`, `Inference ms` and
  `Status` and baseline YOLO11m plus pending RT-DETR-L entries. Every blank
  entry needs an actual owner and date before the Nov 30 / Dec 2 presentation.
- Item 4: list actual blockers, responsible person, and corrective action.
- Item 5: list *actual* work completed by each member, with a file or commit
  of evidence. Merely cherry-picking a prepared commit is not equivalent to
  authoring all of its code; record substantial review, testing, and fixes.
- Item 6: disclose ChatGPT and any other AI tools actually used. Name a
  specific output error that was verified and corrected, not a generic claim.

## Quality checks

```bash
python -m unittest discover -s tests -v
python -m compileall -q src
```

## Before submitting

- Obtain `checkin_template.pptx` from Canvas; fill all boxes without resizing
  or reordering slides, export as PDF, and submit one PDF per Canvas team.
- Make sure a real baseline result is committed and check-in slides cite it.
- Deadline: Tuesday November 17, 2026, 11:58 PM Pacific.
