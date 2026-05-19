"""
Parâmetros de configuração do pipeline de análise clínica comportamental.
Centraliza hiperparâmetros para facilitar ajuste sem modificar a lógica principal.
"""

from dataclasses import dataclass


@dataclass
class ConfiguracaoPipeline:

    # --- Entradas ---
    caminho_video: str = "assets/ConsultaMedicaViroseIA.mp4"
    caminho_modelo_yunet: str = "assets/face_detection_yunet.onnx"

    # --- Saídas (nomenclatura clínica) ---
    caminho_video_anotado: str = "outputs/video_clinico_anotado.mp4"
    caminho_laudo_txt: str = "outputs/laudo_clinico.txt"
    caminho_laudo_json: str = "outputs/laudo_clinico.json"
    caminho_registro_eventos: str = "outputs/registro_eventos_clinicos.csv"

    # --- Detecção facial (YuNet) ---
    # Confiança mínima para aceitar uma detecção como face válida
    limiar_confianca_yunet: float = 0.75
    # Fator de redução antes da detecção — equilibra velocidade e precisão
    escala_redimensionamento: float = 0.75
    # Faces menores que 45×45 px tendem a gerar falsos positivos em contexto clínico
    tamanho_minimo_face: tuple[int, int] = (45, 45)
    # Sobreposição máxima tolerada antes de suprimir caixas duplicadas (NMS)
    sobreposicao_nms: float = 0.30

    # --- Análise de expressões faciais (FER) ---
    # Analisar emoção a cada N quadros reduz custo computacional sem perda clínica relevante
    intervalo_analise_emocao: int = 10

    # --- Análise de movimento e alertas clínicos ---
    # Janela deslizante para média/desvio do movimento (90 quadros ≈ 3 s a 30 FPS)
    janela_movimento: int = 90
    # Z-score acima de 3,0 indica desvio estatisticamente significativo (regra dos 3σ)
    limiar_z_alerta: float = 3.0
