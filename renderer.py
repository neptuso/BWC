"""
Módulo: renderer.py
Descripción: Se encarga de construir y renderizar un Stickman (muñeco de palitos) clásico sobre
             el canvas de Pyglet utilizando primitivas gráficas (pyglet.shapes).
"""

import pyglet
from pyglet import shapes


class PoseRenderer:
    """
    Renderizador de Stickman adaptable para Pyglet 2.x.
    
    QUÉ HACE:
        Construye una representación de Stickman clásico dibujando:
        1. Cabeza: Círculo principal sobre el centro del rostro (nariz/ojos).
        2. Torso: Línea recta que conecta el punto medio de los hombros con el punto medio de las caderas.
        3. Extremidades: Líneas continuas para brazos (Hombro -> Codo -> Muñeca) y piernas (Cadera -> Rodilla -> Tobillo).
        4. Articulaciones: Círculos en los puntos de inflexión.
        
    POR QUÉ:
        Proporciona un visualizador tipo 'Stickman Espejo' limpio y expresivo sin depender de texturas pesadas,
        evaluando el estado 'valid' (confianza > 0.5) de cada keypoint para soportar encuadres parciales del cuerpo.
    """
    def __init__(self, color=(20, 20, 20), joint_radius=6, head_radius=22, line_thickness=6):
        """
        Inicializa la configuración visual del Stickman.
        
        Args:
            color (tuple): Color RGB principal para el Stickman (por defecto negro/gris oscuro).
            joint_radius (int): Radio de los círculos de las articulaciones.
            head_radius (int): Radio del círculo de la cabeza.
            line_thickness (int): Grosor de las líneas del cuerpo en Pyglet 2.x.
        """
        self.color = color
        self.joint_radius = joint_radius
        self.head_radius = head_radius
        self.line_thickness = line_thickness
        
        self.batch = pyglet.graphics.Batch()
        self.shapes_list = []  # Evita que el Recolector de Basura (GC) elimine las primitivas antes de dibujar

    def update_and_draw(self, transformed_keypoints):
        """
        Actualiza las figuras primitivas en el batch y las renderiza en pantalla.
        
        Args:
            transformed_keypoints (dict): Diccionario {index: {"x": float, "y": float, "conf": float, "valid": bool}}
        """
        # Limpiar referencias de figuras anteriores
        self.shapes_list.clear()
        self.batch = pyglet.graphics.Batch()
        
        if not transformed_keypoints:
            return

        # Helper interno para verificar validez de un punto por índice COCO
        def is_valid(idx):
            kp = transformed_keypoints.get(idx)
            return kp is not None and kp.get("valid", False)

        def get_pt(idx):
            kp = transformed_keypoints[idx]
            return kp["x"], kp["y"]

        def add_line(x1, y1, x2, y2):
            """Dibuja una línea entre dos puntos asegurando compatibilidad con Pyglet 2.x (thickness)."""
            line = shapes.Line(
                x1, y1, x2, y2,
                thickness=self.line_thickness,
                color=self.color,
                batch=self.batch
            )
            self.shapes_list.append(line)

        def add_circle(x, y, radius):
            """Dibuja un círculo relleno en las coordenadas dadas."""
            circle = shapes.Circle(
                x, y,
                radius=radius,
                color=self.color,
                batch=self.batch
            )
            self.shapes_list.append(circle)

        # -------------------------------------------------------------------------
        # 1. CABEZA DE STICKMAN
        # -------------------------------------------------------------------------
        # QUÉ HACE: Calcula la posición central del rostro (promedio de puntos faciales válidos o nariz).
        # POR QUÉ: Recrea la cabeza redondeada clásica del Stickman de forma consistente.
        facial_pts = [i for i in range(5) if is_valid(i)]
        if facial_pts:
            avg_x = sum(get_pt(i)[0] for i in facial_pts) / len(facial_pts)
            avg_y = sum(get_pt(i)[1] for i in facial_pts) / len(facial_pts)
            add_circle(avg_x, avg_y, self.head_radius)

        # -------------------------------------------------------------------------
        # 2. TORSO Y HOMBROS/CADERAS
        # -------------------------------------------------------------------------
        # QUÉ HACE: Calcula el punto medio entre ambos hombros y el punto medio entre ambas caderas.
        # POR QUÉ: El torso de un Stickman se dibuja como una columna vertical central única.
        shoulder_mid = None
        hip_mid = None

        if is_valid(5) and is_valid(6):  # Hombro Izquierdo (5) y Derecho (6)
            x5, y5 = get_pt(5)
            x6, y6 = get_pt(6)
            shoulder_mid = ((x5 + x6) / 2.0, (y5 + y6) / 2.0)
            add_line(x5, y5, x6, y6)  # Clavícula/Línea de hombros
        elif is_valid(5):
            shoulder_mid = get_pt(5)
        elif is_valid(6):
            shoulder_mid = get_pt(6)

        if is_valid(11) and is_valid(12):  # Cadera Izquierda (11) y Derecha (12)
            x11, y11 = get_pt(11)
            x12, y12 = get_pt(12)
            hip_mid = ((x11 + x12) / 2.0, (y11 + y12) / 2.0)
            add_line(x11, y11, x12, y12)  # Línea de caderas
        elif is_valid(11):
            hip_mid = get_pt(11)
        elif is_valid(12):
            hip_mid = get_pt(12)

        if shoulder_mid and hip_mid:
            add_line(shoulder_mid[0], shoulder_mid[1], hip_mid[0], hip_mid[1])

        # -------------------------------------------------------------------------
        # 3. EXTREMIDADES (BRAZOS Y PIERNAS)
        # -------------------------------------------------------------------------
        # Pares directos de conexiones de huesos
        limb_connections = [
            # Brazo Izquierdo: Hombro -> Codo -> Muñeca
            (5, 7), (7, 9),
            # Brazo Derecho: Hombro -> Codo -> Muñeca
            (6, 8), (8, 10),
            # Pierna Izquierda: Cadera -> Rodilla -> Tobillo
            (11, 13), (13, 15),
            # Pierna Derecha: Cadera -> Rodilla -> Tobillo
            (12, 14), (14, 16)
        ]

        for p1_idx, p2_idx in limb_connections:
            if is_valid(p1_idx) and is_valid(p2_idx):
                x1, y1 = get_pt(p1_idx)
                x2, y2 = get_pt(p2_idx)
                add_line(x1, y1, x2, y2)

        # -------------------------------------------------------------------------
        # 4. ARTICULACIONES (CIRCULOS EN PUNTOS CLAVE DEL CUERPO)
        # -------------------------------------------------------------------------
        # Excluimos los puntos del rostro (0..4) ya reemplazados por el círculo de la cabeza
        for i in range(5, 17):
            if is_valid(i):
                x, y = get_pt(i)
                add_circle(x, y, self.joint_radius)

        # Dibujar todo el lote acumulado en el lienzo OpenGL de Pyglet
        self.batch.draw()
