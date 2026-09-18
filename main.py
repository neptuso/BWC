"""
Módulo Principal: main.py
Descripción: Aplicación interactiva con Pyglet que integra la captura de cámara en un hilo secundario,
             la inferencia de pose con YOLOv8, la transformación de coordenadas y el renderizado de Stickman en tiempo real.

Requisitos cumplidos:
- Ventana de Pyglet de 800x600 px con fondo blanco limpio.
- Captura de webcam con OpenCV en hilo secundario (sin congelar la UI).
- Inferencia con yolov8n-pose.pt (17 keypoints).
- Transformación de coordenadas (flip espejo + inversión de eje Y + escalado).
- Renderizado de Stickman clásico (cabeza circular, torso central, articulaciones y extremidades).
- Pantalla blanca limpia y libre de fallos cuando no hay personas en cámara.
"""

import sys
import pyglet
from pyglet.gl import glClearColor

from pose_detector import PoseDetectorThread
from coordinate_utils import process_keypoints
from renderer import PoseRenderer


# Dimensiones de la ventana de Pyglet
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
TARGET_FPS = 30.0
CONF_THRESHOLD = 0.5  # Umbral de confianza para renderizar puntos y conexiones


class PoseApp(pyglet.window.Window):
    """
    Ventana principal de la aplicación Pyglet.
    
    QUÉ HACE:
        Maneja el ciclo de vida de la ventana, la programación de eventos de actualización (clock schedule),
        la consulta de datos de pose del hilo secundario y la renderización del Stickman en el canvas.
        
    POR QUÉ:
        Heredar de pyglet.window.Window centraliza la gestión de eventos de Pyglet (on_draw, on_close)
        y mantiene una estructura de código limpia y modular.
    """
    def __init__(self):
        super().__init__(
            width=WINDOW_WIDTH,
            height=WINDOW_HEIGHT,
            caption="Espejo Stickman Interactivo - Pyglet + YOLOv8 Pose",
            resizable=False
        )
        
        # Fondo Blanco Explícito
        glClearColor(1.0, 1.0, 1.0, 1.0)
        
        # Rectángulo de fondo blanco de respaldo
        self.bg_rect = pyglet.shapes.Rectangle(
            0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, color=(255, 255, 255)
        )
        
        # Inicializar hilo de procesamiento de pose
        print("[INFO] Iniciando hilo de captura de webcam e inferencia YOLOv8...")
        self.pose_thread = PoseDetectorThread(camera_index=0, model_path="yolov8n-pose.pt")
        self.pose_thread.start()
        
        # Instanciar el renderizador de Stickman clásico
        self.renderer = PoseRenderer(
            color=(20, 20, 20),      # Color negro/gris oscuro para el Stickman
            joint_radius=6,          # Radio de articulaciones
            head_radius=22,          # Radio del círculo de la cabeza
            line_thickness=6         # Grosor de las líneas del cuerpo
        )
        
        # Etiqueta de estado
        self.status_label = pyglet.text.Label(
            "Esperando detección de cuerpo...",
            font_name="Arial",
            font_size=11,
            x=15, y=WINDOW_HEIGHT - 25,
            color=(120, 120, 120, 255)
        )
        
        self.transformed_keypoints = {}
        self.has_detection = False
        
        # Programar el bucle de actualización a 30 FPS
        pyglet.clock.schedule_interval(self.update, 1.0 / TARGET_FPS)

    def update(self, dt):
        """
        Bucle de actualización periódico llamado por pyglet.clock.
        
        QUÉ HACE:
            Recupera los datos de keypoints más recientes capturados por el hilo de cámara,
            y aplica las transformaciones de coordenadas de OpenCV a Pyglet.
        """
        keypoints, cam_w, cam_h, has_det = self.pose_thread.get_latest_data()
        self.has_detection = has_det
        
        if has_det and keypoints:
            self.transformed_keypoints = process_keypoints(
                keypoints_list=keypoints,
                cam_w=cam_w,
                cam_h=cam_h,
                win_w=WINDOW_WIDTH,
                win_h=WINDOW_HEIGHT,
                conf_threshold=CONF_THRESHOLD
            )
            self.status_label.text = "Stickman activo (Pose detectada)"
        else:
            self.transformed_keypoints = {}
            self.status_label.text = "Buscando persona en la webcam..."

    def on_draw(self):
        """
        Evento de renderizado de OpenGL de Pyglet.
        
        QUÉ HACE:
            Limpia el canvas, dibuja el fondo blanco y renderiza el Stickman si existe una persona en encuadre.
            Si no hay nadie en cámara, mantiene la pantalla blanca limpia sin romper la ejecución.
        """
        self.clear()
        
        # Renderizar fondo blanco
        self.bg_rect.draw()
        
        # Renderizar Stickman (si existen keypoints transformados válidos)
        if self.transformed_keypoints:
            self.renderer.update_and_draw(self.transformed_keypoints)
            
        # Renderizar etiqueta informativa
        self.status_label.draw()

    def on_close(self):
        """
        Manejador del cierre de ventana. Liberación limpia de recursos.
        """
        print("[INFO] Cerrando ventana y liberando recursos de la webcam...")
        self.pose_thread.stop()
        super().on_close()


def main():
    """Punto de entrada principal."""
    app = PoseApp()
    pyglet.app.run()


if __name__ == "__main__":
    main()
