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
    """
    if cam_w <= 0 or cam_h <= 0:
        return 0.0, 0.0

    x_flipped = cam_w - x_cv
    y_inverted = cam_h - y_cv
    
    x_py = (x_flipped / float(cam_w)) * float(win_w)
    y_py = (y_inverted / float(cam_h)) * float(win_h)
    
    return x_py, y_py


def process_keypoints(keypoints_list, cam_w, cam_h, win_w, win_h, conf_threshold=0.5):
    """
    Procesa la lista de 17 keypoints para una única persona.
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


def process_multi_person_keypoints(persons_keypoints_list, cam_w, cam_h, win_w, win_h, conf_threshold=0.5):
    """
    Procesa múltiples personas detectadas en la toma.
    
    QUÉ HACE:
        Mapea cada lista de 17 keypoints individual mediante process_keypoints.
        
    POR QUÉ:
        Permite transformar de forma limpia y homogénea N conjuntos de keypoints para su posterior renderizado.
        
    Returns:
        list: Lista de diccionarios [transformed_person_1, transformed_person_2, ...]
    """
    all_transformed = []
    for person_kps in persons_keypoints_list:
        transformed_person = process_keypoints(
            person_kps, cam_w, cam_h, win_w, win_h, conf_threshold
        )
        all_transformed.append(transformed_person)
    return all_transformed
