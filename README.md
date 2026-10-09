# Padel Impact Position Detection

Proyecto propio para las Unidades 2 y 3 de *Técnicas Avanzadas de Modelado en IA* (Maestría en
Inteligencia Artificial). El objetivo es **seguir los 5 puntos clave de cada jugador**
(**hombro, codo, muñeca, raqueta y cabeza**) a lo largo de un video real de pádel, manteniendo la
identidad de cada jugador entre frames, y comparar distintos enfoques de modelado: CNN propia,
*transfer learning* y YOLO-pose.

**Contenido**

1. [Instalación del entorno](#1-instalación-del-entorno)
2. [Archivos que no vienen en el repositorio](#2-archivos-que-no-vienen-en-el-repositorio)
3. [Ejecutar la app web en vivo](#3-ejecutar-la-app-web-en-vivo)
4. [Notebooks por actividad](#4-notebooks-por-actividad)
5. [Origen y etiquetado de los datos](#5-origen-y-etiquetado-de-los-datos)
6. [Estructura del repositorio](#6-estructura-del-repositorio)

---

## 1. Instalación del entorno

El proyecto usa [uv](https://docs.astral.sh/uv/) para fijar la versión de Python (3.12, en
`.python-version`) y las dependencias (`pyproject.toml`). Pasos en Windows (PowerShell):

```powershell
# 1. Instalar uv (deja uv.exe en C:\Users\<tu_usuario>\.local\bin y lo agrega al PATH)
irm https://astral.sh/uv/install.ps1 | iex

# 2. IMPORTANTE: cerrar y volver a abrir la terminal, para que tome el PATH nuevo.

# 3. Clonar el repo y entrar a la carpeta
git clone https://github.com/ljgarciac/padel-impact-position-detection.git
cd padel-impact-position-detection

# 4. Crear el entorno (.venv) e instalar todas las dependencias (incluye pytest y httpx para las pruebas)
uv sync
```

En macOS o Linux el instalador de uv es `curl -LsSf https://astral.sh/uv/install.sh | sh`, y el resto
de los pasos es igual.

> Si `uv` no se reconoce, incluso en una terminal nueva, revisa que `C:\Users\<tu_usuario>\.local\bin`
> esté en el PATH de usuario, o usa la ruta completa: `& "$env:USERPROFILE\.local\bin\uv.exe" sync`.

### GPU (CUDA) o CPU

`torch` y `torchvision` se instalan desde el índice CUDA de PyTorch
(`https://download.pytorch.org/whl/cu132`), elegido para la GPU del equipo de desarrollo (NVIDIA RTX
4070 Laptop, driver con CUDA 13.4). Para verificarlo:

```powershell
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

- **Con otra GPU NVIDIA:** si `cuda.is_available()` da `False`, prueba con un índice más antiguo
  (`cu128`, `cu126`) en la sección `[[tool.uv.index]]` de `pyproject.toml`, según lo que soporte tu
  driver (lo muestra `nvidia-smi`).
- **Sin GPU NVIDIA:** cambia esa URL a `https://download.pytorch.org/whl/cpu` antes de `uv sync`.
  Todo funciona en CPU, pero el entrenamiento y la app van varias veces más lentos.

### Comprobaciones rápidas

```powershell
uv run python -c "import cv2, mediapipe, ultralytics; print('cv2', cv2.__version__, '| mediapipe', mediapipe.__version__)"
uv run pytest -q
```

Las pruebas (`tests/`) no necesitan GPU, ni pesos, ni cámara; deben pasar todas.

### Notas sobre las dependencias

- **Una sola variante de OpenCV.** `opencv-python`, `opencv-python-headless` y
  `opencv-contrib-python` instalan el mismo paquete `cv2` y se pisan entre sí; `roboflow` arrastra la
  versión *headless* y `mediapipe` la versión *contrib*. Las dos se excluyen en
  `[tool.uv] override-dependencies`, para que quede solo `opencv-python` (con soporte de ventanas,
  necesario para `cv2.imshow`). No agregues otra variante de OpenCV.
- **`uv.lock` no se versiona.** La regla `*.lock` del `.gitignore` lo excluye, así que `uv sync`
  resuelve las versiones de nuevo en cada máquina. Para que todo el equipo use exactamente las mismas
  versiones, hay que quitar esa regla y subir `uv.lock`.

---

## 2. Archivos que no vienen en el repositorio

Las imágenes, los videos y los pesos entrenados están en `.gitignore`: pesan demasiado o se pueden
regenerar. En un clon nuevo hay que conseguirlos antes de usar la app o los notebooks.

| Qué | Ruta esperada | Cómo conseguirlo |
|---|---|---|
| Dataset (imágenes) | `data/Padel_an_2-6/{train,valid,test}/images/` | Se descarga de Roboflow: define `ROBOFLOW_API_KEY` y ejecuta el Checkpoint 0.5 de `notebooks/unidad_2/actividad_3_clasificacion_visual_cnn.ipynb` (o el Checkpoint 1 del notebook de *transfer learning*) |
| **Pesos de nuestro modelo** (YOLOv8n-pose afinado) | `runs/pose/padel_yolo_pose/weights/best.pt` | Entrénalos con `notebooks/unidad_2/actividad_3b_yolo_pose_deteccion.ipynb`, o copia `best.pt` de un compañero a esa ruta |
| Video demo | `data/processed/video_demo/Punto_1-4_mp4_demo.mp4` | Lo genera `notebooks/unidad_2/actividad_3c_demo_inferencia_video.ipynb`, o copia el archivo. Cualquier `.mp4` en esa carpeta aparece como fuente en la app |
| Modelo de MediaPipe Pose | `models/pose_landmarker_full.task` | Se descarga solo la primera vez que se usa (unos 9 MB) |
| Pesos YOLOv8n-pose genéricos | `yolov8n-pose.pt` | Ultralytics los descarga solo cuando hacen falta |

La ruta de los pesos se resuelve en `run_inference_video.load_weights_path()`. Primero lee
`data/processed/pose_model_artifact.json`; si esa ruta no existe en tu máquina, usa
`runs/pose/padel_yolo_pose/weights/best.pt`.

---

## 3. Ejecutar la app web en vivo

La app compara en vivo **nuestro modelo de 5 keypoints** con **MediaPipe Pose** sobre los mismos
frames, en dos pantallas lado a lado. Corresponde al despliegue de la Actividad 4 de la Unidad 3.

```powershell
uv run python -m webapp
```

Se abre el navegador en **http://127.0.0.1:8000**, y el servidor se cierra con `Ctrl+C`. Opciones:

```powershell
uv run python -m webapp --port 8001        # otro puerto
uv run python -m webapp --no-browser       # sin abrir el navegador
```

### Uso

1. **Fuente**: elige una de estas tres.
   - **Video:** cualquier `.mp4` de `data/processed/video_demo/`. Se entrega a ritmo de reloj, como si
     fuera una cámara; *Ritmo de la fuente* fija los FPS (0 = los nativos del video) y *Loops*, cuántas
     veces se repite (0 = hasta detener).
   - **Webcam de este PC:** con su *índice de cámara* (normalmente 0).
   - **iPhone por QR:** ver la sección siguiente.
2. **Modelos**: deja marcados los dos para comparar, o desmarca uno para medirlo aislado (juntos
   compiten por la CPU).
3. **Modo**:
   - **Concurrente:** hilos + colas.
   - **Síncrono:** un solo hilo.
4. **Puntos a mostrar**: *Brazo + raqueta* (por defecto), *Solo brazo* o *Todos*.
5. Pulsa **Ejecutar**. Para cortar antes, **Detener**: termina de forma ordenada y guarda los
   resultados parciales.

Bajo cada pantalla se actualizan en vivo:

- **Desempeño:** FPS, latencias de inferencia y de extremo a extremo (p50/p95), jitter, espera en
  colas, y frames procesados y descartados.
- **Máquina:** CPU, RAM, hilos y VRAM de ese proceso.
- **Predicción:** personas detectadas y confianza de los keypoints.

Al terminar aparece una tabla comparativa con los videos anotados. Todo se guarda en
`report/live_runs/<sesión>/`:

```
report/live_runs/<sesión>/
  yolo/       anotado.mp4  resultados.csv  summary.json
  mediapipe/  anotado.mp4  resultados.csv  summary.json
  recursos.csv                       # CPU, RAM y GPU por proceso a lo largo del tiempo
```

### Usar la cámara del iPhone

El iPhone funciona solo como cámara: transmite por la red local y los dos modelos corren en el PC.

1. Conecta el PC y el iPhone **al mismo Wi-Fi**.
2. En la app, elige **Fuente → iPhone por QR**, o abre directamente
   http://127.0.0.1:8000/#iphone. Aparecen un código QR y el enlace.
3. Escanea el QR con la cámara del iPhone y abre el enlace en **Safari**.
4. Safari mostrará *"Esta conexión no es privada"*: el certificado lo genera la app y no sale de tu
   red. Toca **Mostrar detalles → visitar este sitio web**.
5. En el iPhone pulsa **Empezar** y permite el acceso a la cámara (trasera o frontal; la resolución se
   elige en la misma página).
6. Cuando el PC muestre **Transmitiendo** y la vista previa, pulsa **Ejecutar**.

Si algo no funciona:

| Síntoma | Causa probable | Solución |
|---|---|---|
| El iPhone no abre el enlace | Windows bloquea la conexión entrante | La primera vez Windows pregunta si Python puede aceptar conexiones: permite **redes privadas**. Si ya lo rechazaste, habilítalo en *Firewall de Windows → Permitir una aplicación* |
| El iPhone no abre el enlace, aunque el firewall lo permite | La red aísla a los dispositivos entre sí (redes de invitados o de algunas universidades) | Usa otra red, o comparte internet desde el iPhone al PC |
| "Este enlace ya no sirve" | Se generó un enlace nuevo | Vuelve a escanear el QR (el botón *Nuevo enlace* invalida el anterior) |
| La sesión se detiene sola | El iPhone bloqueó la pantalla o se cambió de app | Deja Safari al frente con la pantalla encendida |
| La cámara no abre en Safari | Falta el permiso | *Ajustes → Safari → Cámara → Permitir* |

Detalles técnicos:

- La app solo escucha en `127.0.0.1`. Para el celular se abre un servidor HTTPS aparte (el *relay*)
  en la IP local del PC, entre los puertos 8443 y 8462, porque Safari solo permite usar la cámara en
  páginas seguras.
- El certificado autofirmado se guarda en `.tools/phone-cert/` y se regenera si cambia la IP.
- El enlace lleva un token aleatorio: solo puede transmitir quien escanea el QR.
- La latencia de extremo a extremo se mide desde que el frame **llega al PC**. El reloj del iPhone no
  está sincronizado con el del PC, así que la red queda fuera de esa cifra.

### Alcance del modelo propio

Nuestro modelo se entrenó con 268 imágenes de transmisión de pádel, en las que cada jugador ocupa entre
el **6 % y el 23 % del alto** del cuadro: de cuerpo entero, visto de lejos, sobre la cancha. Detecta
bien a partir de **un solo** jugador en ese tipo de escena (93 % de las escenas con un único jugador
visible). **No generaliza a primeros planos ni a escenas fuera de la cancha**, como una persona frente
a la webcam: en esos casos no detecta, o devuelve puntos con confianza alta pero en posiciones
incorrectas. MediaPipe sí funciona ahí, porque se entrenó con personas en contextos variados. Para usar
nuestro modelo en otras escenas hay que afinarlo con imágenes de ese tipo.

### Arquitectura

```
iPhone ──wss JPEG──► relay HTTPS (webapp/phone.py) ──┐
Webcam / video ──────────────────────────────────────┤
                                                      ▼
Navegador ◄── MJPEG (video en vivo, uno por modelo) + SSE (métricas)
   │
Servidor FastAPI en 127.0.0.1 (no carga torch ni mediapipe)
   ├─ SessionManager: una sesión a la vez; parada ordenada -> terminate -> kill
   └─ ResourceMonitor: CPU / RAM / VRAM por PID, medidos desde afuera del proceso
   │  multiprocessing "spawn"
   ├─ Captura (1 proceso) ──► ring de frames en SharedMemory (una sola lectura de la fuente)
   ├─ Worker YOLO (GPU)      ─┐ cada uno corre run_sync / run_threaded de realtime_pipeline.py
   └─ Worker MediaPipe (CPU) ─┘ y toma siempre el frame más nuevo (los saltados = descartados)
```

La arquitectura (subproceso `spawn`, progreso por SSE, parada escalonada, video H.264, interfaz Carbon)
se portó de la app local de `padel-vision-analytics-light` (rama `feat/local-app`), y la cámara del
celular, de su rama `live_camara_dual`.

### Benchmark por línea de comandos

Sin interfaz web, el mismo pipeline se mide desde la terminal:

```powershell
uv run python realtime_pipeline.py --mode compare --loops 15            # síncrono vs. concurrente
uv run python realtime_pipeline.py --mode threaded --detector mediapipe
uv run python realtime_pipeline.py --mode threaded --camera-index 0 --display   # ventana OpenCV, 'q' para salir
```

---

## 4. Notebooks por actividad

Los notebooks corren dentro del entorno de uv:

```powershell
uv run jupyter notebook
```

En VS Code, selecciona como kernel el intérprete de `.venv` (`.venv\Scripts\python.exe`). Todos los
notebooks ubican la raíz del repo solos (buscan `data.yaml`), así que funcionan desde su carpeta.

| Notebook | Actividad | Contenido |
|---|---|---|
| `notebooks/unidad_2/actividad_1_imagen_como_arreglo_numerico.ipynb` | U2 · A1 | La imagen como arreglo numérico |
| `notebooks/unidad_2/actividad_2_pipeline_procesamiento_clasico.ipynb` | U2 · A2 | Pipeline de procesamiento clásico |
| `notebooks/unidad_2/actividad_3_clasificacion_visual_cnn.ipynb` | U2 · A3 | Descarga del dataset; regresión de los 5 keypoints (CNN propia, ResNet18, YOLO-pose); clasificación del tipo de keypoint con una CNN desde cero (**87 %** en test) |
| `notebooks/unidad_2/actividad_3b_yolo_pose_deteccion.ipynb` | U2 · A3 (complemento) | Entrena el detector YOLO-pose (**genera `best.pt`**) |
| `notebooks/unidad_2/actividad_3c_demo_inferencia_video.ipynb` | U2 · A3 (complemento) | Tracking con ByteTrack sobre un video real (**genera el video demo**) |
| `notebooks/unidad_2/actividad_4_aumentacion_datos.ipynb` | U2 · A4 | Calidad de datos, aumentación y generalización |
| `notebooks/unidad_3/actividad_3_unidad_3_transfer_learning.ipynb` | U3 · A3 | *Transfer learning* y *fine-tuning* de ResNet18 para el tipo de keypoint, contra la CNN de la U2 como línea base: **96 %** en test frente a 86 % |
| `notebooks/unidad_3/actividad_4_unidad_3_pipeline_tiempo_real.ipynb` | U3 · A4 | Pipeline síncrono vs. concurrente: latencia, FPS, CPU y GPU; el despliegue es la app web (sección 3) |

Las figuras y tablas de los notebooks se guardan en `report/` (`figs/`, `figs_eficiencia/`,
`figs_transfer/`). El notebook de *transfer learning* tarda alrededor de 75 minutos completo en una RTX
4070, porque entrena cinco modelos.

**Partición sin fuga de información.** El split de Roboflow mezcla frames del mismo rally entre
particiones. Los notebooks reparticionan **por rally**, con semilla fija, y guardan el resultado en
`data/processed/image_splits.csv`; todos los modelos comparados usan esa misma partición.

---

## 5. Origen y etiquetado de los datos

Las imágenes son frames de **video de pádel grabado o recopilado por el autor**, extraídos de distintos
puntos (rallies), con nombres como `Punto_1-3_mp4-0000...jpg`. Los 5 keypoints de cada jugador fueron
**anotados manualmente por el autor** en Roboflow y exportados en formato **YOLO-pose**:

- `nc: 1`, `names: ['Person']`, `kpt_shape: [5, 3]` (5 keypoints × [x, y, visibilidad]).
- Orden real de los keypoints en las etiquetas: `0 = hombro, 1 = codo, 2 = muñeca, 3 = raqueta,
  4 = cabeza`, verificado visualmente.
- 335 imágenes, 1.150 jugadores (hasta 4 por frame), unos 35 rallies.

No es un dataset público: la anotación de pose es trabajo original, hecho para este proyecto.

El dataset vive en `data/Padel_an_2-6/` y se descarga con `download("yolov5")` (ver sección 2). El
export "nativo" de YOLO-pose de Roboflow tenía un bug: valores centinela `2.0` desplazados entre ejes en
hombro, codo y raqueta. Con el export `yolov5` el problema desaparece, y se verificó que no hay ningún
valor fuera de rango en las 1150×5×2 coordenadas.

**Historia de la tarea de clasificación.** Primero se intentó clasificar una "posición de impacto"
derivada geométricamente de los keypoints. Con el dataset corregido, esa variable resultó tener un
rango demasiado angosto (unos 50 % de exactitud) y se reemplazó por la clasificación del **tipo de
keypoint**, cuya clase viene directamente de la anotación.

---

## 6. Estructura del repositorio

```
data.yaml                     # config YOLO-pose original del proyecto (nc, kpt_shape, flip_idx)
pyproject.toml                # dependencias (uv) y configuracion de pytest
.python-version               # Python 3.12
data/
  Padel_an_2-6/               # dataset train/valid/test (imagenes no versionadas, ver seccion 2)
  processed/                  # image_splits.csv, manifest_keypoints.csv, crops_kpts/, video_demo/
notebooks/
  unidad_2/                   # actividades 1 a 4 de la Unidad 2
  unidad_3/                   # actividades 3 (transfer learning) y 4 (tiempo real) de la Unidad 3
run_inference_video.py        # video -> YOLO-pose + ByteTrack -> ID de jugador + 5 keypoints -> video anotado
realtime_pipeline.py          # pipeline sincrono vs. concurrente, detector intercambiable (CLI y app)
mediapipe_detector.py         # MediaPipe Pose con la misma interfaz de detector (referencia)
webapp/                       # app web en vivo (uv run python -m webapp)
  server.py                   #   rutas HTTP, SSE y MJPEG (solo 127.0.0.1)
  sessions.py                 #   sesiones: procesos de captura y de modelos, parada escalonada
  capture.py, shared_frames.py#   lectura unica de la fuente y ring de frames en memoria compartida
  worker.py, resources.py     #   un proceso por modelo; CPU/RAM/VRAM por PID
  phone.py                    #   relay HTTPS + QR para la camara del iPhone
  templates/, static/         #   interfaz (Carbon) y pagina del celular
tests/                        # pruebas de la app y del pipeline (uv run pytest)
report/                       # reportes PDF, figuras, benchmarks y live_runs/ de la app
runs/                         # pesos entrenados por ultralytics (no versionados)
models/                       # modelo de MediaPipe descargado (no versionado)
```
