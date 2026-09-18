# Espejo Stickman Interactivo Multi-Persona (YOLOv8 Pose + Pyglet)

Aplicación interactiva en Python que captura la transmisión de una cámara web en tiempo real, procesa la estimación de pose humana para **múltiples personas simultáneamente** utilizando **YOLOv8 Pose** (Ultralytics) en un subproceso independiente y renderiza muñecos de palitos (**Stickmans multicolor**) en una ventana interactiva de **Pyglet**.

---

## 🚀 Características

- **Soporte Multi-Persona Simultáneo**: Detecta y dibuja el Stickman de cada persona presente en la toma.
- **Paleta de Colores por Persona**: Asigna un color único a cada persona (Azul, Rojo, Verde, Púrpura, Naranja, etc.) para diferenciarlas fácilmente.
- **Espejo Interactivo en Tiempo Real**: Mapeo horizontal (efecto espejo) de los movimientos frente a la webcam.
- **Renderizado de Stickman**:
  - Cabeza representada por un círculo limpio en el área facial.
  - Torso definido entre el punto medio de los hombros y el punto medio de las caderas.
  - Brazos y piernas dibujados con articulaciones continuas.
- **Procesamiento Multihilo (Multi-threaded)**: La inferencia de la IA corre en un hilo secundario con `threading.Thread`, evitando congelamientos o caídas de FPS en la interfaz gráfica de Pyglet.
- **Pantalla Blanca Limpia**: Mantiene el lienzo libre de artefactos o fallos cuando no hay personas presentes en el encuadre.

---

## 📋 Requisitos Previos

- **Python 3.10 o superior** (probado en Python 3.12).
- Cámara web integrada o USB conectada.

---

## 🛠️ Instalación y Uso

### 1. Clonar el repositorio
```bash
git clone https://github.com/neptuso/BWC.git
cd BWC/Ver1
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

---

## 📄 Créditos y Licencia

Proyecto basado en una idea de @linkfydev y orquestado por @neptuso con herramientas opensource, para uso personal.
