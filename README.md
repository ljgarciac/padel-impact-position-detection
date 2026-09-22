# Padel Impact Position Detection

Proyecto propio para la Unidad 2 (Técnicas Avanzadas de Modelado en IA, Maestría en Inteligencia
Artificial). El objetivo final es **trackear correctamente los 5 puntos clave de cada jugador**
(**cabeza, hombro, codo, muñeca, raqueta**) a lo largo de un video real de pádel, con una identidad
persistente por jugador (no solo detección independiente por frame) — comparando distintos enfoques de
modelado para la regresión de esos puntos (CNN propia, transfer learning, YOLO-pose).

## Origen y etiquetado de los datos

Las imágenes provienen de **video de pádel grabado/recopilado por el autor** (frames extraídos de
distintos "puntos"/rallies, con nombres tipo `Punto_1_mp4-0000...jpg`, `Punto_1-3_mp4-...`, etc.).
Los 5 keypoints por jugador (cabeza, hombro, codo, muñeca, raqueta) fueron **anotados manualmente por
el autor** usando Roboflow, y exportados en formato **YOLO-pose**:

- `nc: 1`, `names: ['Person']`, `kpt_shape: [5, 3]` (5 keypoints × [x, y, visibilidad]).
- Hasta 4 jugadores anotados por frame.
- 335 imágenes en total, 1150 instancias de jugador (335 imágenes × hasta 4 jugadores).

No es un dataset público preexistente: la anotación de pose es trabajo original del autor, hecha para
este proyecto.

**El dataset vive en `data/Padel_an_2-6/`** (train/valid/test), descargado directamente desde Roboflow
(`notebooks/actividad_3_clasificacion_visual_cnn.ipynb`, Checkpoint 0.5). Un primer export del proyecto tenía un bug
de exportación de Roboflow (valores de keypoints desplazados entre ejes para hombro/codo/raqueta,
confirmado comparando ambos exports y revisando directamente en Roboflow que la anotación manual era
correcta); reexportando con `download("yolov5")` en vez del formato usado originalmente, el problema
desaparece por completo (verificado: 0 valores fuera de rango en las 1150×5×2 coordenadas). Por eso ya
no hay carpetas `train/`, `valid/`, `test/` en la raíz del repo — todo el dataset está en `data/`.

## De pose a clasificación (Actividad 3)

`notebooks/actividad_3_clasificacion_visual_cnn.ipynb` aborda dos tareas:

1. **Predicción de los 5 keypoints por jugador** (regresión de coordenadas), comparando tres modelos:
   una CNN propia entrenada desde cero, transfer learning (ResNet18 preentrenado + cabeza de
   regresión) y el detector YOLO-pose ya entrenado en `notebooks/actividad_3b_yolo_pose_deteccion.ipynb`. Es la
   base para trackear los 5 puntos en video (`notebooks/actividad_3c_demo_inferencia_video.ipynb`).
2. **Clasificación del tipo de keypoint** (cabeza/hombro/codo/muñeca/raqueta) a partir de un recorte
   pequeño centrado en cada punto anotado (tamaño proporcional al alto del jugador, ~5500 parches en
   total) — esto produce la salida de clasificación con matriz de confusión y métricas por clase que
   exige la rúbrica de la Actividad 3, usando la clase real de la anotación directamente (sin ninguna
   regla geométrica derivada). Resultado: **81% de exactitud en test** (vs. 21% del baseline
   mayoritario), con confusiones interpretables entre keypoints vecinos (hombro↔codo, por texturas de
   ropa similares; cabeza↔raqueta en algunos casos).

   *Nota histórica*: se probó primero clasificar una "posición de impacto" derivada geométricamente de
   los keypoints (altura de la raqueta respecto a la muñeca), pero con el dataset corregido esa
   variable resultó tener un rango real demasiado angosto para ser una clase robusta (~50% de exactitud
   frente al ~83% que daba, engañosamente, con el dataset con el bug de exportación). Se abandonó esa
   idea en favor de la clasificación de tipo de keypoint.

El split `train/valid/test` que trae el export de Roboflow mezcla frames del mismo rally entre
particiones (fuga de información), así que el notebook **reparticiona por rally** con una semilla fija
(reutilizada también por `notebooks/actividad_3b_yolo_pose_deteccion.ipynb`, que entrena sobre el split original
sin reagrupar, por decisión explícita de no reorganizar el dataset).

## Tracking de jugadores en video

`notebooks/actividad_3c_demo_inferencia_video.ipynb` y `run_inference_video.py` usan el detector YOLO-pose
entrenado en `notebooks/actividad_3b_yolo_pose_deteccion.ipynb` con **tracking de identidad persistente**
(ByteTrack, vía `model.track()` de ultralytics) en vez de detección independiente por frame: cada
jugador mantiene un ID estable mientras el tracker lo siga, con sus 5 keypoints dibujados cuadro a
cuadro. No forma parte de las métricas exigidas por la Actividad 3.

## Estructura del repo

```
data.yaml                            # config YOLO-pose original del proyecto (nc, kpt_shape, flip_idx)
data/Padel_an_2-6/                   # dataset (train/valid/test), descargado de Roboflow (ver Checkpoint 0.5)
pyproject.toml, uv.lock, .python-version  # dependencias y version de Python, gestionadas con uv
notebooks/
  actividad_3_clasificacion_visual_cnn.ipynb      # Actividad 3: descarga el dataset, regresión de 5 keypoints
                                      # (3 modelos comparados) + clasificación del tipo de keypoint (entregable evaluado)
  actividad_3b_yolo_pose_deteccion.ipynb       # entrena un detector YOLO-pose (ultralytics) sobre data/Padel_an_2-6
  actividad_3c_demo_inferencia_video.ipynb     # demo del framework de tracking sobre un video real
data/processed/                      # manifest_keypoints.csv, crops_kpts/ y artefactos del notebook principal
run_inference_video.py               # framework de ejecución: video -> YOLO-pose + tracking (ByteTrack) -> ID de jugador + 5 keypoints por frame -> video anotado
report/                              # reporte técnico en PDF (máx. 5 páginas) — pendiente de actualizar al nuevo pipeline
```

## Alcance de cada componente

- **`notebooks/actividad_3_clasificacion_visual_cnn.ipynb`**: es el entregable evaluado por la rúbrica de la
  Actividad 3 (preparación de datos, CNN, evaluación, interpretación visual, comunicación técnica).
- **`notebooks/actividad_3b_yolo_pose_deteccion.ipynb`**, **`notebooks/actividad_3c_demo_inferencia_video.ipynb`** y
  **`run_inference_video.py`**: trabajo adicional que sienta las bases del modelo final del proyecto
  (detección y tracking de los 5 keypoints por jugador en video real, frame por frame). No forma parte
  de las métricas exigidas por la Actividad 3.

## Entorno (uv) y GPU

El proyecto usa [uv](https://docs.astral.sh/uv/) para fijar la versión de Python (`.python-version`,
3.12) y las dependencias (`pyproject.toml` + `uv.lock`, reproducibles).

```powershell
# Instala uv (instalador oficial; deja uv.exe en C:\Users\<tu_usuario>\.local\bin y lo agrega al PATH)
irm https://astral.sh/uv/install.ps1 | iex

# IMPORTANTE: cierra y vuelve a abrir la terminal de PowerShell despues de instalar,
# para que recoja el PATH nuevo (las ventanas ya abiertas no se actualizan solas).

# Crea el entorno (.venv) con todas las dependencias fijadas en uv.lock
uv sync
```

> Si `uv` no se reconoce incluso en una terminal nueva, revisa que
> `C:\Users\<tu_usuario>\.local\bin` este en tu PATH de usuario, o usa la ruta completa:
> `& "$env:USERPROFILE\.local\bin\uv.exe" sync`.

`torch`/`torchvision` están configurados en `pyproject.toml` para instalarse desde el índice CUDA de
PyTorch (`https://download.pytorch.org/whl/cu132`), detectado para la GPU de esta máquina (NVIDIA
GeForce RTX 4070 Laptop, driver con soporte CUDA 13.4). Verificación rápida:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**En una máquina sin GPU NVIDIA compatible**, cambia la URL en `[[tool.uv.index]]` de
`pyproject.toml` a `https://download.pytorch.org/whl/cpu` (o corre
`uv pip install torch torchvision --torch-backend=cpu`) antes de `uv sync`.

Los notebooks corren dentro del entorno de uv:

```bash
uv run jupyter notebook notebooks/actividad_3_clasificacion_visual_cnn.ipynb
```

También se registró un kernel de Jupyter (`padel-gpu`) apuntando a `.venv`, por si abres los notebooks
desde otra instalación de Jupyter/VS Code y quieres seleccionar ese entorno manualmente.

`notebooks/actividad_3_clasificacion_visual_cnn.ipynb` detecta la GPU automáticamente
(`device = "cuda" if torch.cuda.is_available() else "cpu"`); `notebooks/actividad_3b_yolo_pose_deteccion.ipynb`
(ultralytics) también usa GPU automáticamente cuando está disponible.
