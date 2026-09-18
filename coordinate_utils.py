"""
Módulo: coordinate_utils.py
Descripción: Proporciona funciones para transformar coordenadas desde el sistema de OpenCV
             (origen arriba-izquierda) al sistema de Pyglet (origen abajo-izquierda) con efecto espejo horizontal.
"""

def transform_cv_to_pyglet(x_cv, y_cv, cam_w, cam_h, win_w, win_h):
    """
    Transforma un punto único (x, y) de OpenCV a las coordenadas de lienzo de Pyglet.
    
    QUÉ HACE:
        1. Aplica efecto espejo horizontal: x_flipped = cam_w - x_cv.
        2. Invierte el eje Y para alinear el origen abajo-izquierda: y_inverted = cam_h - y_cv.
        3. Escala proporcionalmente las coordenadas al tamaño de la ventana de Pyglet (win_w, win_h).
        
    POR QUÉ:
        OpenCV posiciona el (0,0) en la esquina superior izquierda (matriz de imagen), mientras que Pyglet
        utiliza el sistema cartesiano estándar donde (0,0) está en la esquina inferior izquierda.
        El efecto espejo es imprescindible para una interacción natural del usuario frente a la webcam.
    """
    if cam_w <= 0 or cam_h <= 0:
        return 0.0, 0.0

    # 1. Flip horizontal
    x_flipped = cam_w - x_cv
    
    # 2. Inversión de eje Y (Top-Left a Bottom-Left)
    y_inverted = cam_h - y_cv
    
    # 3. Escalado a la ventana de Pyglet
    x_py = (x_flipped / float(cam_w)) * float(win_w)
    y_py = (y_inverted / float(cam_h)) * float(win_h)
    
    return x_py, y_py


def process_keypoints(keypoints_list, cam_w, cam_h, win_w, win_h, conf_threshold=0.5):
    """
    Procesa la lista completa de 17 keypoints extraídos por YOLOv8.
    
    QUÉ HACE:
        Mapea cada tupla (x, y, confidence) aplicando la conversión de coordenadas
        y retorna una lista estructurada donde cada punto indica su validez según el umbral de confianza.
        
    POR QUÉ:
        Permite que el renderizador maneje cuerpos completos o parciales de forma homogénea,
        marcando como no válidos (valid=False) los puntos con baja confianza o fuera de encuadre.
        
    Returns:
        dict: Diccionario mapeado {index: {"x": float, "y": float, "conf": float, "valid": bool}}
    """
    transformed = {}
    
    for i, (x, y, conf) in enumerate(keypoints_list):
        is_valid = conf >= conf_threshold
        if is_valid:
            x_py, y_py = transform_cv_to_pyglet(x, y, cam_w, cam_h, win_w, win_h)
        else:
            x_py, y_py = 0.0, 0.0
            
        transformed[i] = {
            "x": x_py,
            "y": y_py,
            "conf": conf,
            "valid": is_valid
        }
        
    return transformed
