"""
Módulo: pose_detector.py
Descripción: Captura frames de la webcam en un hilo separado y procesa la estimación
             de pose utilizando YOLOv8 pose de Ultralytics para evitar congelamientos en la UI de Pyglet.
"""

import threading
import cv2
from ultralytics import YOLO


class PoseDetectorThread(threading.Thread):
    """
    Hilo dedicado a la captura de cámara web y la inferencia del modelo YOLO pose.
    
    QUÉ HACE:
        Lee continuamente frames de OpenCV cv2.VideoCapture, los procesa con yolov8n-pose.pt
        y almacena el último conjunto de 17 keypoints extraídos en una estructura segura para subprocesos.
        
    POR QUÉ:
        La inferencia de IA en CPU/GPU puede demorar entre 15ms y 50ms por frame. Ejecutarla en el hilo principal
        de Pyglet provocaría fluctuaciones en los FPS y congelamientos en la renderización de la ventana.
    """
    def __init__(self, camera_index=0, model_path="yolov8n-pose.pt"):
        super().__init__()
        self.daemon = True  # Permite que el hilo finalice si el programa principal se cierra
        self.camera_index = camera_index
        self.model_path = model_path
        
        self._lock = threading.Lock()
        self._running = False
        
        # Variables compartidas de estado
        self.latest_keypoints = []  # Lista de tuplas: [(x, y, confidence), ...] para los 17 puntos
        self.frame_width = 640       # Valor por defecto (se actualiza con la cámara)
        self.frame_height = 480      # Valor por defecto (se actualiza con la cámara)
        self.has_detection = False
        
        self.model = None
        self.cap = None

    def run(self):
        """
        Bucle principal del hilo de procesamiento.
        Carga el modelo e inicia el ciclo de lectura de cámara.
        """
        # Cargar modelo YOLO pose
        # QUÉ: Cargar yolov8n-pose.pt
        # POR QUÉ: Es la variante nano de YOLOv8 pose, optimizada para ejecución rápida en tiempo real.
        self.model = YOLO(self.model_path)
        
        # Inicializar cámara OpenCV
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"[ERROR] No se pudo abrir la cámara index {self.camera_index}")
            return
            
        # Obtener dimensiones reales de la captura
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        
        self._running = True
        
        while self._running:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                continue
                
            # Realizar inferencia de pose (verbose=False para evitar saturar la consola)
            results = self.model(frame, verbose=False)
            
            keypoints_list = []
            detection_found = False
            
            if results and len(results) > 0:
                result = results[0]
                # Verificar si se detectaron personas en el frame
                if result.keypoints is not None and len(result.keypoints) > 0:
                    # Tomamos la primera persona detectada con mayor confianza
                    xy_data = result.keypoints.xy[0].cpu().numpy()
                    conf_data = result.keypoints.conf[0].cpu().numpy() if result.keypoints.conf is not None else None
                    
                    for i in range(len(xy_data)):
                        x, y = xy_data[i]
                        c = conf_data[i] if conf_data is not None else 1.0
                        keypoints_list.append((float(x), float(y), float(c)))
                        
                    detection_found = True

            # Actualizar datos compartidos utilizando el lock de seguridad
            with self._lock:
                self.latest_keypoints = keypoints_list
                self.has_detection = detection_found

        # Liberar recursos al detener
        if self.cap:
            self.cap.release()

    def get_latest_data(self):
        """
        Retorna una copia de los datos más recientes de detección.
        
        Returns:
            tuple: (keypoints_list, frame_width, frame_height, has_detection)
        """
        with self._lock:
            return list(self.latest_keypoints), self.frame_width, self.frame_height, self.has_detection

    def stop(self):
        """Detiene el hilo de procesamiento de forma limpia."""
        self._running = False
