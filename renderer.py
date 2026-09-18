"""
Módulo: renderer.py
Descripción: Se encarga de construir y renderizar múltiples Stickmans de colores sobre
             el canvas de Pyglet utilizando primitivas gráficas (pyglet.shapes).
"""

import pyglet
from pyglet import shapes

# Paleta de colores distintivos para cada persona detectada
PERSON_COLORS = [
    (30, 144, 255),   # Persona 1: Azul brillante
    (220, 20, 60),    # Persona 2: Rojo Carmesí
    (46, 139, 87),    # Persona 3: Verde Mar
    (153, 50, 204),   # Persona 4: Púrpura Violeta
    (255, 140, 0),    # Persona 5: Naranja Intenso
    (0, 180, 216),    # Persona 6: Teal Cían
]


class PoseRenderer:
    """
    Renderizador de Stickman Multi-persona adaptable para Pyglet 2.x.
    
    QUÉ HACE:
        Construye y dibuja múltiples Stickmans en pantalla, asignando a cada persona
        un color único de la paleta PERSON_COLORS.
        
    POR QUÉ:
        Permite diferenciar visualmente a varios usuarios interactuando simultáneamente
        frente a la webcam sin saturar la UI.
    """
    def __init__(self, joint_radius=6, head_radius=22, line_thickness=6):
        """
        Inicializa la configuración visual del Stickman.
        """
        self.joint_radius = joint_radius
        self.head_radius = head_radius
        self.line_thickness = line_thickness
        
        self.batch = pyglet.graphics.Batch()
        self.shapes_list = []  # Mantiene referencias vivas para el GC de Pyglet

    def update_and_draw(self, persons_keypoints_list):
        """
        Actualiza las figuras primitivas en el batch para todas las personas y las dibuja.
        
        Args:
            persons_keypoints_list (list): Lista de diccionarios con keypoints transformados por persona.
        """
        self.shapes_list.clear()
        self.batch = pyglet.graphics.Batch()
        
        if not persons_keypoints_list:
            return

        # Dibujar cada persona con su respectivo color
        for person_idx, transformed_keypoints in enumerate(persons_keypoints_list):
            color = PERSON_COLORS[person_idx % len(PERSON_COLORS)]
            self._draw_single_stickman(transformed_keypoints, color)

        # Dibujar todo el lote acumulado en el lienzo de Pyglet
        self.batch.draw()

    def _draw_single_stickman(self, transformed_keypoints, color):
        """
        Dibuja un único Stickman con el color especificado.
        """
        def is_valid(idx):
            kp = transformed_keypoints.get(idx)
            return kp is not None and kp.get("valid", False)

        def get_pt(idx):
            kp = transformed_keypoints[idx]
            return kp["x"], kp["y"]

        def add_line(x1, y1, x2, y2):
            line = shapes.Line(
                x1, y1, x2, y2,
                thickness=self.line_thickness,
                color=color,
                batch=self.batch
            )
            self.shapes_list.append(line)

        def add_circle(x, y, radius):
            circle = shapes.Circle(
                x, y,
                radius=radius,
                color=color,
                batch=self.batch
            )
            self.shapes_list.append(circle)

        # 1. CABEZA
        facial_pts = [i for i in range(5) if is_valid(i)]
        if facial_pts:
            avg_x = sum(get_pt(i)[0] for i in facial_pts) / len(facial_pts)
            avg_y = sum(get_pt(i)[1] for i in facial_pts) / len(facial_pts)
            add_circle(avg_x, avg_y, self.head_radius)

        # 2. TORSO Y HOMBROS/CADERAS
        shoulder_mid = None
        hip_mid = None

        if is_valid(5) and is_valid(6):
            x5, y5 = get_pt(5)
            x6, y6 = get_pt(6)
            shoulder_mid = ((x5 + x6) / 2.0, (y5 + y6) / 2.0)
            add_line(x5, y5, x6, y6)
        elif is_valid(5):
            shoulder_mid = get_pt(5)
        elif is_valid(6):
            shoulder_mid = get_pt(6)

        if is_valid(11) and is_valid(12):
            x11, y11 = get_pt(11)
            x12, y12 = get_pt(12)
            hip_mid = ((x11 + x12) / 2.0, (y11 + y12) / 2.0)
            add_line(x11, y11, x12, y12)
        elif is_valid(11):
            hip_mid = get_pt(11)
        elif is_valid(12):
            hip_mid = get_pt(12)

        if shoulder_mid and hip_mid:
            add_line(shoulder_mid[0], shoulder_mid[1], hip_mid[0], hip_mid[1])

        # 3. EXTREMIDADES (BRAZOS Y PIERNAS)
        limb_connections = [
            (5, 7), (7, 9),     # Brazo Izquierdo
            (6, 8), (8, 10),    # Brazo Derecho
            (11, 13), (13, 15), # Pierna Izquierda
            (12, 14), (14, 16)  # Pierna Derecha
        ]

        for p1_idx, p2_idx in limb_connections:
            if is_valid(p1_idx) and is_valid(p2_idx):
                x1, y1 = get_pt(p1_idx)
                x2, y2 = get_pt(p2_idx)
                add_line(x1, y1, x2, y2)

        # 4. ARTICULACIONES
        for i in range(5, 17):
            if is_valid(i):
                x, y = get_pt(i)
                add_circle(x, y, self.joint_radius)
