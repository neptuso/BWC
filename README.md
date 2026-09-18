# Espejo Stickman Interactivo (YOLOv8 Pose + Pyglet)

Aplicación interactiva en Python que captura la transmisión de una cámara web en tiempo real, procesa la estimación de pose humana utilizando **YOLOv8 Pose** (Ultralytics) en un subproceso independiente y renderiza un muñeco de palitos (**Stickman**) en una ventana interactiva de **Pyglet**.

---

## 🚀 Características

- **Espejo Interactivo en Tiempo Real**: Mapeo horizontal (efecto espejo) de los movimientos frente a la webcam.
- **Renderizado de Stickman**:
  - Cabeza representada por un círculo limpio en el área facial.
  - Torso definido entre el punto medio de los hombros y el punto medio de las caderas.
  - Brazos y piernas dibujados con articulaciones continuas.
- **Procesamiento Multihilo (Multi-threaded)**: La inferencia de la IA corre en un hilo secundario con `threading.Thread`, evitando congelamientos o caídas de FPS en la interfaz de usuario de Pyglet.
- **Adaptación a Cuerpo Parcial**: Filtra keypoints por umbral de confianza (`conf > 0.5`), permitiendo renderizar extremidades o torso según el encuadre visible en la cámara.
- **Pantalla Blanca Limpia**: Mantiene el lienzo libre de artefactos o fallos cuando no hay personas presentes en el encuadre.

---

## 📋 Requisitos Previos

- **Python 3.10 o superior** (probado en Python 3.12).
- Cámara web integrada o USB conectada.

---

## 🛠️ Instalación y Uso

### 1. Clonar o acceder al repositorio
```bash
git clone <URL_DEL_REPOSITORIO>
cd bwc_body_webcam_copy/Ver1
```

### 2. Crear y activar el entorno virtual (`venv`)

**En Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**En Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Ejecutar la aplicación
```bash
python main.py
```

*Nota: En la primera ejecución, YOLOv8 descargará automáticamente el modelo preentrenado `yolov8n-pose.pt` si no existe localmente.*

---

## 📁 Estructura del Proyecto

```
Ver1/
├── venv/                 # Entorno virtual de Python
├── yolov8n-pose.pt       # Modelo de pose preentrenado de YOLOv8
├── coordinate_utils.py   # Transformaciones de coordenadas (OpenCV -> Pyglet + Espejo)
├── pose_detector.py      # Captura de cámara e inferencia YOLOv8 en hilo secundario
├── renderer.py           # Renderizador del Stickman con primitivas de Pyglet
├── main.py               # Ventana interactiva y bucle principal de la aplicación
├── requirements.txt      # Dependencias del proyecto
└── README.md             # Documentación del proyecto
```

---

## 📄 Licencia

Proyecto basado en una idea de @linkfydev y orquestado por @neptuso con herramientas opensource, para uso personal.
