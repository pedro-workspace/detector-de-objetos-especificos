"""Config de alvos do detector de escritorio (yolo11s, mais preciso)."""

# Alvos de interesse. Todo o resto e desenhado em cinza como [OUTRO]
# quando show-all estiver ativo, para provar a classificacao.
# Cores em BGR (padrao OpenCV).
TARGETS = {
    "mouse": {
        "color": (0, 255, 0),  # verde
        "tag": "ESCRITORIO",
    },
    "cell phone": {
        "color": (255, 0, 0),  # azul
        "tag": "ESCRITORIO",
    },
}

OTHER_COLOR = (128, 128, 128)  # cinza para demais classes COCO
OTHER_TAG = "OUTRO"

DEFAULT_MODEL = "yolo11s.pt"
DEFAULT_CONF = 0.25
DEFAULT_IMGSZ = 640
DEFAULT_SHOW_ALL = True
