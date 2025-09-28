import os
import sys
import subprocess
from pathlib import Path
from tkinter import messagebox, filedialog

# ---------- RUTAS / SETUP ----------
BASE_DIR = Path(__file__).resolve().parent
Y5_DIR = BASE_DIR / "yolov5"

def _pip_install_requirements():
    req = Y5_DIR / "requirements.txt"
    if not req.exists():
        return
    # instala requirements si falta algo típico
    try:
        import seaborn  # suele faltar
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req)])

def ensure_yolov5_ready():
    if not Y5_DIR.exists():
        subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5.git", str(Y5_DIR)], check=True)
    _pip_install_requirements()

    parent = str(BASE_DIR)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    # 👇 Agrega ESTO: que Python pueda resolver "utils" dentro de yolov5/
    y5_str = str(Y5_DIR)
    if y5_str not in sys.path:
        sys.path.insert(0, y5_str)

    assert (Y5_DIR / "models" / "common.py").exists(), "YOLOv5 repo no válido."


# Llama al setup antes de los imports de yolov5
ensure_yolov5_ready()

# ---------- IMPORTS YOLOV5 ----------
import cv2
import torch
import numpy as np
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import letterbox
from yolov5.utils.torch_utils import select_device
from yolov5.utils.general import non_max_suppression

WINDOW_WIDTH, WINDOW_HEIGHT = 960, 720

# ---------- UTILIDADES ----------
def find_best_model():
    """
    1) $MODEL_PATH (opcional, prioriza si existe)
    2) runs/train/*/weights/best.pt (el más reciente)
    3) runs/train/*/weights/last.pt (fallback)
    4) diálogo para elegir .pt
    """
    env_path = os.getenv("MODEL_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)

    candidates = []
    for name in ("best.pt", "last.pt"):
        for p in (Y5_DIR / "runs" / "train").glob(f"*/weights/{name}"):
            try:
                mtime = p.stat().st_mtime
            except Exception:
                mtime = 0
            candidates.append((mtime, p))
    if candidates:
        candidates.sort(reverse=True)  # más reciente primero
        return candidates[0][1]

    # diáologo como último recurso
    messagebox.showwarning("Modelo no encontrado", "No se halló ningún best.pt/last.pt en runs/train.\nSelecciona un archivo .pt.")
    f = filedialog.askopenfilename(
        title="Seleccionar modelo (.pt)",
        filetypes=[("PyTorch weights", "*.pt"), ("Todos", "*.*")]
    )
    if f:
        return Path(f)
    raise FileNotFoundError("No se seleccionó modelo .pt")

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = ((img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2)
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    coords[:, 0].clamp_(0, img0_shape[1]); coords[:, 2].clamp_(0, img0_shape[1])
    coords[:, 1].clamp_(0, img0_shape[0]); coords[:, 3].clamp_(0, img0_shape[0])
    return coords

def draw_label_with_bg(img, text, topleft, color_bg):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.6, 2
    text_size, _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = topleft
    cv2.rectangle(img, (x, y - text_size[1] - 4), (x + text_size[0] + 4, y), color_bg, -1)
    cv2.putText(img, text, (x + 2, y - 2), font, scale, (255, 255, 255), thickness)

# ---------- MODELO ----------
def load_model_auto():
    model_path = find_best_model()
    device = select_device('0' if torch.cuda.is_available() else 'cpu')
    model = DetectMultiBackend(str(model_path), device=device, dnn=False)
    model.model.float().eval()
    return model, model.stride, model.names, device

# ---------- INFERENCIA ----------
def run_detection(source, model, stride, names, device):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        messagebox.showerror("Error", f"No se pudo abrir la fuente: {source}")
        return

    paused, delay = False, 30
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            img = letterbox(frame, new_shape=640, stride=stride, auto=True)[0]
            img = img.transpose((2, 0, 1))[::-1]
            img = np.ascontiguousarray(img)
            img_tensor = torch.from_numpy(img).to(device).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0)

            with torch.no_grad():
                pred = model(img_tensor, augment=False)
                pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]

            danger_count = safe_count = 0
            if pred is not None and len(pred):
                pred[:, :4] = scale_coords(img_tensor.shape[2:], pred[:, :4], frame.shape).round()
                for *xyxy, conf, cls in pred:
                    label = names[int(cls)].lower()
                    conf = float(conf)
                    x1, y1, x2, y2 = map(int, xyxy)
                    if label == 'protegido' and conf > 0.8:
                        safe_count += 1
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        draw_label_with_bg(frame, "Protegido", (x1, y1), (0, 200, 0))
                    elif label == 'riesgo' and conf > 0.5:
                        danger_count += 1
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        draw_label_with_bg(frame, "Riesgo", (x1, y1), (0, 0, 200))

            total = danger_count + safe_count
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (270, 80), (255, 255, 255), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            info = [f"People in danger: {danger_count}", f"People protected: {safe_count}", f"Total people: {total}"]
            for i, text in enumerate(info):
                cv2.putText(frame, text, (20, 30 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            h, w = frame.shape[:2]
            aspect = w / h
            if aspect > WINDOW_WIDTH / WINDOW_HEIGHT:
                new_w = WINDOW_WIDTH; new_h = int(WINDOW_WIDTH / aspect)
            else:
                new_h = WINDOW_HEIGHT; new_w = int(WINDOW_HEIGHT * aspect)
            resized = cv2.resize(frame, (new_w, new_h))
            canvas = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
            y0 = (WINDOW_HEIGHT - new_h) // 2; x0 = (WINDOW_WIDTH - new_w) // 2
            canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
            cv2.imshow("Helmet Detection System - YOLOv5", canvas)

        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key in (ord('+'), ord('=')):
            delay = max(1, delay - 5)
        elif key in (ord('-'), ord('_')):
            delay += 5

    cap.release()
    cv2.destroyAllWindows()
