import numpy as np
# ÁNGULO DE CADERA

SITTING_MIN_ANGLE = 60.0
SITTING_MAX_ANGLE = 130.0

# ÁNGULO DE RODILLA

CROUCHING_MAX_KNEE_ANGLE = 125.0

# DIFERENCIA ENTRE LOS DOS LADOS

MAX_SIDE_ANGLE_DIFFERENCE = 45.0

# REACHING
# Distancia relativa para determinar si una mano está
# claramente por encima del hombro.
REACHING_HEIGHT_MARGIN = 0.10

NOSE = 0

L_SHOULDER = 5
R_SHOULDER = 6

L_ELBOW = 7
R_ELBOW = 8

L_WRIST = 9
R_WRIST = 10

L_HIP = 11
R_HIP = 12

L_KNEE = 13
R_KNEE = 14

L_ANKLE = 15
R_ANKLE = 16

# FUNCIONES BÁSICAS

def valid_keypoint(point: np.ndarray) -> bool:
#    Comprueba si un keypoint contiene coordenadas válidas.

    if point is None:
        return False

    point = np.asarray(point)

    if point.shape != (2,):
        return False

    if not np.all(np.isfinite(point)):
        return False
    # YOLO produce [0, 0] cuando el punto no
    # fue localizado correctamente.
    if np.allclose(point, 0):
        return False

    return True

# CÁLCULO DE ÁNGULOS
def calculate_angle(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray
) -> float | None:
    
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    c = np.asarray(c, dtype=np.float32)

    if not (
        valid_keypoint(a)
        and valid_keypoint(b)
        and valid_keypoint(c)
    ):
        return None

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return None

    cosine_angle = np.dot(ba, bc) / (
        norm_ba * norm_bc
    )

    cosine_angle = np.clip(
        cosine_angle,
        -1.0,
        1.0
    )

    angle = np.degrees(
        np.arccos(cosine_angle)
    )

    return float(angle)

# ÁNGULO DE CADERA

def calculate_hip_angles(keypoints):

    l_shoulder = keypoints[L_SHOULDER]
    r_shoulder = keypoints[R_SHOULDER]

    l_hip = keypoints[L_HIP]
    r_hip = keypoints[R_HIP]

    l_knee = keypoints[L_KNEE]
    r_knee = keypoints[R_KNEE]

    angle_left = None
    angle_right = None

    if (
        valid_keypoint(l_shoulder)
        and valid_keypoint(l_hip)
        and valid_keypoint(l_knee)
    ):
        angle_left = calculate_angle(
            l_shoulder,
            l_hip,
            l_knee
        )

    if (
        valid_keypoint(r_shoulder)
        and valid_keypoint(r_hip)
        and valid_keypoint(r_knee)
    ):
        angle_right = calculate_angle(
            r_shoulder,
            r_hip,
            r_knee
        )

    return angle_left, angle_right

# ÁNGULO DE RODILLA

def calculate_knee_angles(keypoints):
    l_hip = keypoints[L_HIP]
    l_knee = keypoints[L_KNEE]
    l_ankle = keypoints[L_ANKLE]

    r_hip = keypoints[R_HIP]
    r_knee = keypoints[R_KNEE]
    r_ankle = keypoints[R_ANKLE]

    angle_left = None
    angle_right = None

    if (
        valid_keypoint(l_hip)
        and valid_keypoint(l_knee)
        and valid_keypoint(l_ankle)
    ):
        angle_left = calculate_angle(
            l_hip,
            l_knee,
            l_ankle
        )

    if (
        valid_keypoint(r_hip)
        and valid_keypoint(r_knee)
        and valid_keypoint(r_ankle)
    ):
        angle_right = calculate_angle(
            r_hip,
            r_knee,
            r_ankle
        )

    return angle_left, angle_right

# ÁNGULO ROBUSTO DE CADERA

def get_robust_hip_angle(keypoints):
    #Obtiene un único ángulo de cadera utilizando ambos lados
    #cuando están disponibles.
    
    left, right = calculate_hip_angles(keypoints)

    if left is not None and right is not None:

        difference = abs(left - right)

        if difference <= MAX_SIDE_ANGLE_DIFFERENCE:

            return (left + right) / 2.0

        # Cuando existe una diferencia grande entre ambos
        # lados, utilizamos el menor ángulo como referencia.
        return min(left, right)

    if left is not None:
        return left

    if right is not None:
        return right

    return None

# DETECCIÓN DE SENTADO

def is_sitting(hip_angle):
 #    Determina si el ángulo de cadera es compatible
#    con una postura sentada.
 
    if hip_angle is None:
        return False

    return (
        SITTING_MIN_ANGLE
        <= hip_angle
        <= SITTING_MAX_ANGLE
    )

# DETECCIÓN DE AGACHADO

def is_crouching(
    hip_angle,
    knee_angles
):
    """
    Detecta una postura agachada.

    Utiliza simultáneamente:

    - ángulo de cadera
    - flexión de rodillas

    Esto evita confundir fácilmente una persona sentada
    con una persona agachada.
    """

    if hip_angle is None:
        return False

    left_knee, right_knee = knee_angles

    valid_knees = [
        angle
        for angle in (left_knee, right_knee)
        if angle is not None
    ]

    if not valid_knees:
        return False

    average_knee_angle = np.mean(valid_knees)

    # Una persona agachada normalmente presenta
    # una flexión significativa de rodillas.
    if average_knee_angle <= CROUCHING_MAX_KNEE_ANGLE:

        # Evitamos clasificar como agachado cuando
        # claramente corresponde a sentado.
        if not is_sitting(hip_angle):
            return True

    return False

# DETECCIÓN DE REACHING

def is_reaching(keypoints):
    """
    Se considera reaching cuando una muñeca está claramente
    por encima del hombro correspondiente.
    """

    l_shoulder = keypoints[L_SHOULDER]
    r_shoulder = keypoints[R_SHOULDER]

    l_wrist = keypoints[L_WRIST]
    r_wrist = keypoints[R_WRIST]

    # Altura corporal aproximada.
    valid_body_points = [
        p for p in [
            l_shoulder,
            r_shoulder,
            keypoints[L_HIP],
            keypoints[R_HIP]
        ]
        if valid_keypoint(p)
    ]

    if len(valid_body_points) < 2:
        return False

    body_height = max(
        p[1] for p in valid_body_points
    ) - min(
        p[1] for p in valid_body_points
    )

    # Evitamos divisiones con valores pequeños.
    if body_height < 20:
        return False
    # Brazo izquierdo

    if (
        valid_keypoint(l_shoulder)
        and valid_keypoint(l_wrist)
    ):

        vertical_difference = (
            l_shoulder[1] - l_wrist[1]
        )

        if vertical_difference > (
            body_height * REACHING_HEIGHT_MARGIN
        ):
            return True
    # Brazo derecho

    if (
        valid_keypoint(r_shoulder)
        and valid_keypoint(r_wrist)
    ):

        vertical_difference = (
            r_shoulder[1] - r_wrist[1]
        )

        if vertical_difference > (
            body_height * REACHING_HEIGHT_MARGIN
        ):
            return True

    return False

# CLASIFICACIÓN PRINCIPAL

def classify_pose(
    keypoints: np.ndarray
) -> tuple[str, float | None]:
    """
    Clasifica la postura de una persona.

    Categorías actuales:

        Sitting
        Crouching
        Reaching
        Standing / Other

    Retorna:

        (postura, ángulo_de_cadera)
    """
    # VALIDACIÓN

    if keypoints is None:
        return "Unknown", None

    keypoints = np.asarray(
        keypoints,
        dtype=np.float32
    )

    if keypoints.ndim != 2:
        return "Unknown", None

    if keypoints.shape[0] < 17:
        return "Unknown", None

    if keypoints.shape[1] < 2:
        return "Unknown", None
    # ÁNGULO DE CADERA

    hip_angle = get_robust_hip_angle(
        keypoints
    )
    # ÁNGULOS DE RODILLA

    knee_angles = calculate_knee_angles(
        keypoints
    )

    # 1. SITTING

    if is_sitting(hip_angle):
        return "Sitting", hip_angle

    # 2. CROUCHING

    if is_crouching(
        hip_angle,
        knee_angles
    ):
        return "Crouching", hip_angle

    # 3. REACHING

    if is_reaching(keypoints):
        return "Reaching", hip_angle

    # 4. STANDING / OTHER

    return "Standing / Other", hip_angle

