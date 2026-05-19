"""
Reconhecimento de expressões faciais para análise clínica comportamental.

Utiliza a biblioteca FER com MTCNN para alinhamento facial antes da classificação.
As emoções são mapeadas para terminologia clínica em português para padronização dos laudos.
"""

from fer import FER


# Tradução dos rótulos do FER (inglês) para terminologia clínica em português
MAPA_EMOCOES_CLINICO: dict[str, str] = {
    "angry":   "raiva",
    "disgust": "desgosto",
    "fear":    "medo",
    "happy":   "alegria",
    "sad":     "tristeza",
    "surprise": "surpresa",
    "neutral": "neutro",
}


def criar_analisador_emocoes() -> FER:
    """
    Instancia o modelo FER com MTCNN ativado.
    MTCNN realiza alinhamento facial antes da classificação, aumentando a acurácia
    em faces parcialmente giradas — relevante em contextos clínicos variados.
    """
    return FER(mtcnn=True)


def analisar_expressao(modelo_fer: FER, roi_facial) -> str:
    """
    Classifica a expressão facial dominante na ROI fornecida.

    Retorna o rótulo clínico em português.
    Retorna 'indeterminado' quando o modelo não atinge confiança suficiente
    ou quando a ROI é inválida (muito pequena, corrompida ou fora do plano frontal).
    """
    if roi_facial is None or roi_facial.size == 0:
        return "indeterminado"

    try:
        resultado = modelo_fer.detect_emotions(roi_facial)
        if not resultado:
            return "indeterminado"

        emocao_ingles = max(resultado[0]["emotions"], key=resultado[0]["emotions"].get)
        return MAPA_EMOCOES_CLINICO.get(emocao_ingles, "indeterminado")
    except Exception:
        # FER pode lançar exceções em ROIs muito pequenas ou com baixo contraste
        return "indeterminado"
