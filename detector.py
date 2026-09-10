"""Detector de objetos de escritorio via webcam (mouse x celular).

YOLO leve pre-treinado no COCO (default yolo11s.pt), baixado
automaticamente pelo pacote `ultralytics` na primeira execucao.
Toda deteccao acima do conf desenha bounding box + label na GUI.
"""

import argparse
import time

import cv2
from ultralytics import YOLO

from office_config import (
    DEFAULT_CONF,
    DEFAULT_IMGSZ,
    DEFAULT_MODEL,
    DEFAULT_SHOW_ALL,
    OTHER_COLOR,
    OTHER_TAG,
    TARGETS,
)


def parse_args():
    p = argparse.ArgumentParser(description="Detector de mouse/celular via webcam")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Ex: yolo11s.pt")
    p.add_argument("--conf", type=float, default=DEFAULT_CONF, help="Confianca minima (0-1)")
    p.add_argument("--camera", type=int, default=0, help="Indice da webcam")
    p.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="Tamanho de inferencia")
    p.add_argument("--show-all", dest="show_all", action="store_true",
                   help="Desenha tambem outras classes em cinza [OUTRO]")
    p.add_argument("--no-show-all", dest="show_all", action="store_false",
                   help="Desenha so mouse/celular")
    p.set_defaults(show_all=DEFAULT_SHOW_ALL)
    p.add_argument("--debug", action="store_true", help="Log de deteccoes no console")
    p.add_argument("--list-classes", action="store_true", help="Lista classes do modelo e sai")
    return p.parse_args()


def draw_box(frame, x1, y1, x2, y2, color, label):
    """Desenha SEMPRE bounding box + label. Nunca invisivel."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y_bg = max(y1 - th - 10, 0)
    cv2.rectangle(frame, (x1, y_bg), (x1 + tw, y1), color, -1)
    cv2.putText(frame, label, (x1, y1 - 5 if y1 - 5 > 0 else y1 + th + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def draw_detections(frame, boxes, names, show_all=True, debug=False):
    """Retorna (n_targets, n_others). Desenha box+label para cada deteccao."""
    if boxes is None or len(boxes) == 0:
        return 0, 0
    n_targets = 0
    n_others = 0
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        name = names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if name in TARGETS:
            n_targets += 1
            cfg = TARGETS[name]
            label = f"{name} {conf:.2f} [{cfg['tag']}]"
            draw_box(frame, x1, y1, x2, y2, cfg["color"], label)
        elif show_all:
            n_others += 1
            label = f"{name} {conf:.2f} [{OTHER_TAG}]"
            draw_box(frame, x1, y1, x2, y2, OTHER_COLOR, label)
        if debug:
            print(f"  - {name} conf={conf:.2f} xyxy=({x1},{y1},{x2},{y2})")
    return n_targets, n_others


def draw_hud(frame, model_name, conf, fps, n_targets, n_others):
    lines = [
        f"{model_name} conf={conf:.2f} FPS={fps:.1f} alvos={n_targets} outros={n_others}",
        "verde=mouse azul=celular cinza=outro | q=sair",
    ]
    y = 22
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (5, y - th - 6), (5 + tw + 6, y + 4), (0, 0, 0), -1)
        cv2.putText(frame, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        y += 24


def main():
    args = parse_args()
    # Na primeira vez, o .pt e baixado sozinho do GitHub releases Ultralytics.
    model = YOLO(args.model)

    if args.list_classes:
        print("Classes do modelo:")
        for i, n in model.names.items():
            mark = " <-- ALVO" if n in TARGETS else ""
            print(f"  {i}: {n}{mark}")
        return

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERRO] Webcam {args.camera} nao encontrada.")
        print("Dica: teste `yolo predict model=yolo11s.pt source=0 show=True conf=0.25`")
        return

    print(f"Modelo {args.model} conf={args.conf} show_all={args.show_all}")
    print("Pressione 'q' para sair.")
    prev = time.time()
    fps = 0.0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERRO] Falha ao ler frame da webcam.")
            break

        results = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)
        boxes = results[0].boxes
        if args.debug:
            total = 0 if boxes is None else len(boxes)
            print(f"[frame] {total} deteccoes raw:")
        n_targets, n_others = draw_detections(
            frame, boxes, model.names, show_all=args.show_all, debug=args.debug)

        now = time.time()
        dt = now - prev
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt
        prev = now
        draw_hud(frame, args.model, args.conf, fps, n_targets, n_others)

        if n_targets == 0 and n_others == 0:
            cv2.putText(frame, "Mostre um mouse ou celular",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Office Detector (q p/ sair)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
