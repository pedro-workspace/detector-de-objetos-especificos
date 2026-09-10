# Office Detector — mouse x celular (YOLO + webcam)

Toda detecção acima do `conf` desenha **bounding box + label** na GUI:
- `mouse 0.62 [ESCRITORIO]` → caixa **verde**
- `cell phone 0.71 [ESCRITORIO]` → caixa **azul**
- `remote 0.55 [OUTRO]`, etc. → caixa **cinza** (prova que classificou)

Rede: `yolo11s.pt` (Ultralytics, mais preciso que `n`), pré-treinada no COCO.
Não precisa de imagens de treino. O `.pt` baixa sozinho na 1ª execução.

## Instalação (Linux, Python 3.10+)

```bash
pip install -r requirements.txt
```

## Uso

```bash
python detector.py
python detector.py --model yolo11s.pt --conf 0.25 --camera 0 --imgsz 640 --show-all --debug
python detector.py --no-show-all          # só alvos (verde/azul)
python detector.py --list-classes         # confere nomes exatos das classes
```

Pressione `q` para sair. HUD mostra `modelo + conf + FPS + contadores`.

## Teste sem webcam (só CLI YOLO)

```bash
yolo predict model=yolo11s.pt source=0 show=True conf=0.25
yolo predict model=yolo11s.pt source='https://ultralytics.com/images/bus.jpg' save=True
```

## Arquivos

- `office_config.py` — TARGETS (verde/azul), OTHER cinza, defaults (s, 0.25, 640)
- `detector.py` — loop webcam + draw_box/draw_hud sempre visíveis + debug
- `requirements.txt` — `ultralytics`, `opencv-python`

## Dicas

- Boa luz + aproximar o objeto. `conf=0.25` pega mouse ocluso na mão.
- `cell phone` cobre smartphone. Mouse na mão pode sair como `remote` (cinza) — me avise que adiciono alias.
- `yolo11s` é ~2-3x mais lento que `n` no CPU; se travar, use `--imgsz 480`.
