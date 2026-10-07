# Vehicle Sticker Detection (YOLO26)

YOLO-based vehicle sticker detection thesis project.

## Clone

```bash
git clone <your-repo-url>
cd thesis
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

Dataset folders (`images/train`, `images/val`, `labels/train`, `labels/val`) are placeholders — copy your dataset `.jpg` + `.txt` files into them. See `data.yaml`.

Trained weights are included at `models/best.pt`.

## Run

Test camera:

```bash
python camera_test.py
```

Live detection:

```bash
python webcam.py
```

Press `Q` to quit.

## Train (optional)

```bash
yolo detect train model=yolo26n.pt data=data.yaml imgsz=640 epochs=100
```
