from __future__ import annotations
"""
Detecção facial usando o modelo YuNet via OpenCV.

Implementa detecção em múltiplas rotações para cobrir diferentes orientações de cabeça,
relevante em contextos clínicos onde o paciente pode estar em postura não-frontal.
"""

import cv2
import numpy as np


def criar_detector_yunet(
    caminho_modelo: str,
    tamanho_entrada: tuple[int, int],
    limiar_confianca: float,
) -> cv2.FaceDetectorYN:
    """Instancia o detector YuNet com os parâmetros fornecidos."""
    largura, altura = tamanho_entrada
    return cv2.FaceDetectorYN.create(
        model=caminho_modelo,
        config="",
        input_size=(largura, altura),
        score_threshold=limiar_confianca,
        nms_threshold=0.30,
        top_k=5000,
    )


def _rotacionar_imagem(imagem, angulo: int):
    if angulo == 0:
        return imagem
    if angulo == 90:
        return cv2.rotate(imagem, cv2.ROTATE_90_CLOCKWISE)
    if angulo == 180:
        return cv2.rotate(imagem, cv2.ROTATE_180)
    if angulo == 270:
        return cv2.rotate(imagem, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError("Ângulo deve ser 0, 90, 180 ou 270")


def _mapear_caixa_para_original(
    x: int, y: int, w: int, h: int,
    angulo: int,
    largura_orig: int,
    altura_orig: int,
) -> tuple[int, int, int, int]:
    """
    Converte coordenadas da bounding box da imagem rotacionada
    de volta ao sistema de coordenadas da imagem original.
    """
    if angulo == 0:
        return x, y, w, h
    if angulo == 180:
        return largura_orig - (x + w), altura_orig - (y + h), w, h
    if angulo == 90:
        return y, altura_orig - (x + w), h, w
    if angulo == 270:
        return largura_orig - (y + h), x, h, w
    raise ValueError("Ângulo deve ser 0, 90, 180 ou 270")


def _suprimir_nao_maximos(
    caixas: list[tuple[int, int, int, int]],
    limiar_sobreposicao: float = 0.30,
) -> list[tuple[int, int, int, int]]:
    """
    Non-Maximum Suppression: elimina caixas redundantes que detectam a mesma face.
    Ordena por coordenada Y inferior e descarta sobreposições acima do limiar.
    """
    if not caixas:
        return []

    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in caixas], dtype=np.float32)
    x1, y1, x2, y2 = rects[:, 0], rects[:, 1], rects[:, 2], rects[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    indices = np.argsort(y2)

    selecionados = []
    while len(indices) > 0:
        ultimo = indices[-1]
        selecionados.append(ultimo)

        xx1 = np.maximum(x1[ultimo], x1[indices[:-1]])
        yy1 = np.maximum(y1[ultimo], y1[indices[:-1]])
        xx2 = np.minimum(x2[ultimo], x2[indices[:-1]])
        yy2 = np.minimum(y2[ultimo], y2[indices[:-1]])

        w_inter = np.maximum(0, xx2 - xx1 + 1)
        h_inter = np.maximum(0, yy2 - yy1 + 1)
        sobreposicao = (w_inter * h_inter) / areas[indices[:-1]]

        indices = indices[np.where(sobreposicao <= limiar_sobreposicao)[0]]

    return [
        (int(rects[i][0]), int(rects[i][1]),
         int(rects[i][2] - rects[i][0]),
         int(rects[i][3] - rects[i][1]))
        for i in selecionados
    ]


def _detectar_com_yunet(
    detector: cv2.FaceDetectorYN,
    quadro_bgr,
) -> list[tuple[int, int, int, int, float]]:
    """Executa o YuNet no quadro e retorna lista de (x, y, w, h, score)."""
    altura, largura = quadro_bgr.shape[:2]
    detector.setInputSize((largura, altura))
    _, faces = detector.detect(quadro_bgr)

    if faces is None:
        return []

    return [
        (int(f[0]), int(f[1]), int(f[2]), int(f[3]), float(f[-1]))
        for f in faces
        if int(f[2]) > 0 and int(f[3]) > 0
    ]


def detectar_faces(
    quadro_bgr,
    detector: cv2.FaceDetectorYN,
    tamanho_minimo: tuple[int, int] = (50, 50),
    proporcao_min: float = 0.70,
    proporcao_max: float = 1.45,
    sobreposicao_nms: float = 0.30,
) -> list[tuple[int, int, int, int]]:
    """
    Detecta faces no quadro usando o YuNet em 4 rotações (0°, 90°, 180°, 270°).

    A detecção multi-rotação compensa a ausência de invariância rotacional do YuNet,
    garantindo cobertura quando o paciente está em posição não-frontal.
    Aplica NMS ao final para eliminar detecções sobrepostas.

    Retorna lista de (x, y, w, h) nas coordenadas do quadro recebido.
    """
    altura_orig, largura_orig = quadro_bgr.shape[:2]
    melhor_conjunto: list[tuple] = []

    for angulo in (0, 90, 180, 270):
        rotacionado = _rotacionar_imagem(quadro_bgr, angulo)
        deteccoes_brutas = _detectar_com_yunet(detector, rotacionado)

        mapeadas = []
        for (x, y, w, h, _score) in deteccoes_brutas:
            if w < tamanho_minimo[0] or h < tamanho_minimo[1]:
                continue
            if not (proporcao_min <= w / float(h) <= proporcao_max):
                continue

            xo, yo, wo, ho = _mapear_caixa_para_original(x, y, w, h, angulo, largura_orig, altura_orig)
            xo = max(0, min(xo, largura_orig - 1))
            yo = max(0, min(yo, altura_orig - 1))
            wo = max(1, min(wo, largura_orig - xo))
            ho = max(1, min(ho, altura_orig - yo))
            mapeadas.append((xo, yo, wo, ho))

        # Mantém o conjunto com maior número de detecções entre as 4 rotações
        if len(mapeadas) > len(melhor_conjunto):
            melhor_conjunto = mapeadas

    return _suprimir_nao_maximos(melhor_conjunto, sobreposicao_nms)


def escalar_faces_para_original(
    faces: list[tuple[int, int, int, int]],
    fator_inverso: float,
    largura_orig: int,
    altura_orig: int,
) -> list[tuple[int, int, int, int]]:
    """
    Converte coordenadas detectadas no quadro redimensionado para o quadro original.
    Necessário porque a detecção é feita em resolução reduzida por desempenho,
    mas as anotações são desenhadas na resolução completa.
    """
    resultado = []
    for (x, y, w, h) in faces:
        xo = max(0, min(int(x * fator_inverso), largura_orig - 1))
        yo = max(0, min(int(y * fator_inverso), altura_orig - 1))
        wo = max(1, min(int(w * fator_inverso), largura_orig - xo))
        ho = max(1, min(int(h * fator_inverso), altura_orig - yo))
        resultado.append((xo, yo, wo, ho))
    return resultado
