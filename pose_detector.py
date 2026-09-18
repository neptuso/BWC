"""
Módulo: pose_detector.py
Descripción: Captura frames de la webcam en un hilo separado y procesa la estimación
             de pose multi-persona utilizando YOLOv8 pose de Ultralytics.
"""

import threading
import cv2
from ultralytics import YOLO


class PoseDetectorThread(threading.Thread):
    """
    Hilo dedicado a la captura de cámara web y la inferencia del modelo YOLO pose para múltiples personas.
    
    QUÉ HACE:
        Lee continuamente frames de OpenCV cv2.VideoCapture, ejecuta YOLOv8 pose en tiempo real
        y almacena una lista con los 17 keypoints de CADA persona detectada en el encuadre.
        
    POR QUÉ:
        Permite detectar N personas en simultáneo sin bloquear la interfaz gráfica de Pyglet.
    """
    def __init__(self, camera_index=0, model_path="yolov8n-pose.pt"):
        super().__init__()
        self.daemon = True
        self.camera_index = camera_index
        self.model_path = model_path
        
        self._lock = threading.Lock()
        self._running = False
        
        # Estructura compartida de personas: [[(x,y,conf)_1..17], [(x,y,conf)_1..17], ...]
        self.latest_persons_keypoints = []
        self.frame_width = 640
        self.frame_height = 480
        self.has_detection = False
        
        self.model = None
        self.cap = None

    def run(self):
        """
        Bucle principal del hilo de procesamiento.
        """
        self.model = YOLO(self.model_path)
        
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"[ERROR] No se pudo abrir la cámara index {self.camera_index}")
            return
            
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        
        self._running = True
        
        while self._running:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                continue
                
            results = self.model(frame, verbose=False)
            
            persons_list = []
            detection_found = False
            
            if results and len(results) > 0:
                result = results[0]
                # QUÉ: Extraer keypoints de TODAS las personas detectadas en la toma
                # POR QUÉ: result.keypoints.xy tiene dimensiones (num_personas, 17, 2)
                if result.keypoints is not None and len(result.keypoints) > 0:
                    xy_all = result.keypoints.xy.cpu().numpy()
                    conf_all = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None
                    
                    num_persons = len(xy_all)
                    for p in range(num_persons):
                        person_kps = []
                        xy_person = xy_all[p]
                        conf_person = conf_all[p] if conf_all is not None else None
                        
                        for i in range(len(xy_person)):
                            x, y = xy_person[i]
                            c = float(conf_person[i]) if conf_person is not None else 1.0
                            person_kps.append((float(x), float(y), c))
                            
                        persons_list.append(person_kps)
                        
                    detection_found = True

            with self._lock:
                self.latest_persons_keypoints = persons_list
                self.has_detection = detection_found

        if self.cap:
            self.cap.release()

    def get_latest_data(self):
        """
        Retorna los datos de detección multi-persona más recientes.
        
        Returns:
            tuple: (persons_keypoints_list, frame_width, frame_height, has_detection)
        """
        with self._lock:
            return list(self.latest_persons_keypoints), self.frame_width, self.frame_height, self.has_detection

    def stop(self):
        """Detiene el hilo de procesamiento."""
        self._running = False
