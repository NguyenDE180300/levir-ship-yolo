# Raw dataset placeholder

Place the extracted **LEVIR-Ship raw dataset** in this directory. Do not commit
the images or labels to Git.

Expected source layout for `prepare_yolo_data.py`:

```text
data/
├── images/   # source PNG images
└── labels/   # matching YOLO TXT labels
```

Build the deterministic YOLO split with:

```bash
python prepare_yolo_data.py --data-root data --out-root yolo_data --split-seed 42
```
