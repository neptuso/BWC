"""
Módulo Principal: main.py
Descripción: Aplicación interactiva en Pyglet con soporte Multi-Persona en tiempo real,
             integrando captura de cámara en subproceso, inferencia YOLOv8 Pose y renderizado multicolor de Stickman.
"""

import sys
import pyglet
from pyglet.gl import glClearColor

from pose_detector import PoseDetectorThread
from coordinate_utils import process_multi_person_keypoints
from renderer import PoseRenderer


# Dimensiones de la ventana de Pyglet
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
TARGET_FPS = 30.0
CONF_THRESHOLD = 0.5  # Umbral de confianza para renderizar puntos y conexiones


class PoseApp(pyglet.window.Window):
    """
    Ventana principal de la aplicación Pyglet Multi-persona.
    """
    def __init__(self):
        super().__init__(
            width=WINDOW_WIDTH,
            height=WINDOW_HEIGHT,
            caption="Espejo Stickman Interactivo Multi-Persona - Pyglet + YOLOv8 Pose",
            resizable=False
        )
        
        # Fondo Blanco
        glClearColor(1.0, 1.0, 1.0, 1.0)
        
        # Rectángulo de fondo blanco de respaldo
        self.bg_rect = pyglet.shapes.Rectangle(
            0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, color=(255, 255, 255)
        )
        
        # Inicializar hilo de procesamiento de pose
        print("[INFO] Iniciando hilo de captura de webcam e inferencia YOLOv8 multi-persona...")
        self.pose_thread = PoseDetectorThread(camera_index=0, model_path="yolov8n-pose.pt")
        self.pose_thread.start()
        
        # Instanciar el renderizador multi-persona
        self.renderer = PoseRenderer(
            joint_radius=6,          # Radio de articulaciones
            head_radius=22,          # Radio del círculo de la cabeza
            line_thickness=6         # Grosor de las líneas del cuerpo
        )
        
        # Etiqueta de estado de la interfaz
        self.status_label = pyglet.text.Label(
            "Esperando detección de cuerpo...",
            font_name="Arial",
            font_size=11,
            x=15, y=WINDOW_HEIGHT - 25,
            color=(120, 120, 120, 255)
        )
        
        self.transformed_persons = []
        self.has_detection = False
        
        # Programar el bucle de actualización a 30 FPS
        pyglet.clock.schedule_interval(self.update, 1.0 / TARGET_FPS)

    def update(self, dt):
        """
        Bucle de actualización periódico llamado por pyglet.clock.
        """
        persons_kps_list, cam_w, cam_h, has_det = self.pose_thread.get_latest_data()
        self.has_detection = has_det
        
        if has_det and persons_kps_list:
            # QUÉ: Transformar las coordenadas de TODAS las personas detectadas
            self.transformed_persons = process_multi_person_keypoints(
                persons_keypoints_list=persons_kps_list,
                cam_w=cam_w,
                cam_h=cam_h,
                win_w=WINDOW_WIDTH,
                win_h=WINDOW_HEIGHT,
                conf_threshold=CONF_THRESHOLD
            )
            count = len(self.transformed_persons)
            self.status_label.text = f"Stickman activo: {count} persona(s) detectada(s)"
        else:
            self.transformed_persons = []
            self.status_label.text = "Buscando personas en la webcam..."

    def on_draw(self):
        """
        Evento de renderizado de OpenGL de Pyglet.
        """
        self.clear()
        
        # Renderizar fondo blanco
        self.bg_rect.draw()
        
        # Renderizar Stickmans de todas las personas en pantalla
        if self.transformed_persons:
            self.renderer.update_and_draw(self.transformed_persons)
            
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
