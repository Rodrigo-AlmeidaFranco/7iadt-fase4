"""
Análise de movimento e detecção de alertas clínicos comportamentais.

Utiliza diferença absoluta entre quadros consecutivos como proxy para intensidade de movimento.
A classificação de estado motor é feita via z-score em janela deslizante, tornando
o limiar adaptativo ao nível basal de atividade do paciente ao longo do vídeo.
"""

import cv2
import numpy as np
from typing import NamedTuple


# Rótulos de estado motor em terminologia clínica
ESTADO_REPOUSO = "repouso"
ESTADO_MOVIMENTO_VOLUNTARIO = "movimento_voluntario"
ESTADO_AGITACAO_PSICOMOTORA = "agitacao_psicomotora"


class ResultadoMovimento(NamedTuple):
    estado_motor: str
    alerta_clinico: bool
    z_movimento: float
    indice_movimento: float


def calcular_indice_movimento(quadro_cinza_anterior, quadro_cinza_atual) -> float:
    """
    Calcula o índice de movimento como média das diferenças absolutas entre
    quadros consecutivos em escala de cinza.

    Valores elevados indicam maior deslocamento físico ou mudança brusca de iluminação.
    """
    diferenca = cv2.absdiff(quadro_cinza_anterior, quadro_cinza_atual)
    return float(np.mean(diferenca))


def classificar_estado_motor(
    indice_movimento: float,
    historico: list[float],
    limiar_z: float = 3.0,
) -> ResultadoMovimento:
    """
    Classifica o estado motor com base no z-score do índice de movimento
    em relação ao histórico recente (janela deslizante).

    A abordagem adaptativa via z-score é preferível a limiares fixos porque o nível
    basal de movimento varia entre pacientes e entre trechos do mesmo vídeo clínico.

    Estados possíveis:
      - repouso              : movimento abaixo ou igual à média histórica
      - movimento_voluntario : acima da média, dentro dos 3σ
      - agitacao_psicomotora : acima de 3σ — dispara alerta clínico
    """
    if len(historico) < 2:
        return ResultadoMovimento(ESTADO_REPOUSO, False, 0.0, indice_movimento)

    media = float(np.mean(historico))
    desvio = float(np.std(historico))

    # Evita divisão por zero em vídeos com movimento absolutamente constante
    z = 0.0 if desvio <= 1e-6 else (indice_movimento - media) / desvio

    if z > limiar_z:
        return ResultadoMovimento(ESTADO_AGITACAO_PSICOMOTORA, True, z, indice_movimento)

    if indice_movimento > media:
        return ResultadoMovimento(ESTADO_MOVIMENTO_VOLUNTARIO, False, z, indice_movimento)

    return ResultadoMovimento(ESTADO_REPOUSO, False, z, indice_movimento)
