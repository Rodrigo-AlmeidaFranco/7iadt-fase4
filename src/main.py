"""
Pipeline de análise clínica comportamental por vídeo.

Orquestra detecção facial, reconhecimento de expressões e análise de estado motor
para geração de laudo clínico estruturado em múltiplos formatos.
"""
from __future__ import annotations

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suprime logs informativos do TensorFlow

import warnings
warnings.filterwarnings("ignore")

import cv2
from collections import deque

from config import ConfiguracaoPipeline
from detector_facial import (
    criar_detector_yunet,
    detectar_faces,
    escalar_faces_para_original,
)
from analisador_emocoes import criar_analisador_emocoes, analisar_expressao
from analisador_movimento import calcular_indice_movimento, classificar_estado_motor
from gerador_laudo import (
    garantir_diretorio,
    exportar_registro_eventos,
    gerar_laudo_json,
    gerar_laudo_txt,
)

from tqdm import tqdm


def _anotar_quadro(
    quadro,
    numero_quadro: int,
    faces: list[tuple[int, int, int, int]],
    expressoes: list[str],
    estado_motor: str,
    alerta_clinico: bool,
) -> None:
    """Desenha anotações clínicas diretamente no quadro (operação in-place)."""

    for idx, (x, y, w, h) in enumerate(faces, start=1):
        expressao = expressoes[idx - 1] if idx - 1 < len(expressoes) else "indeterminado"

        cv2.rectangle(quadro, (x, y), (x + w, y + h), (0, 220, 0), 2)
        cv2.putText(
            quadro,
            f"Indivíduo {idx} | {expressao}",
            (x, max(0, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        quadro,
        f"Quadro: {numero_quadro} | Rostos: {len(faces)} | Estado: {estado_motor}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if alerta_clinico:
        cv2.putText(
            quadro,
            "ALERTA CLÍNICO",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )


def executar_pipeline(config: ConfiguracaoPipeline | None = None) -> None:
    """
    Executa o pipeline completo de análise clínica sobre o vídeo configurado.
    Aceita uma instância de ConfiguracaoPipeline ou usa os valores padrão.
    """
    if config is None:
        config = ConfiguracaoPipeline()

    if not os.path.exists(config.caminho_video):
        raise FileNotFoundError(f"Vídeo não encontrado: {config.caminho_video}")
    if not os.path.exists(config.caminho_modelo_yunet):
        raise FileNotFoundError(f"Modelo YuNet não encontrado: {config.caminho_modelo_yunet}")

    for caminho in [
        config.caminho_video_anotado,
        config.caminho_laudo_txt,
        config.caminho_laudo_json,
        config.caminho_registro_eventos,
    ]:
        garantir_diretorio(caminho)

    # --- Inicialização do vídeo ---
    captura = cv2.VideoCapture(config.caminho_video)
    if not captura.isOpened():
        raise RuntimeError("Não foi possível abrir o vídeo de entrada.")

    fps = captura.get(cv2.CAP_PROP_FPS) or 30.0
    largura = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_quadros_estimado = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))

    gravador = cv2.VideoWriter(
        config.caminho_video_anotado,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (largura, altura),
    )

    # --- Inicialização dos modelos ---
    detector = criar_detector_yunet(
        config.caminho_modelo_yunet,
        (largura, altura),
        config.limiar_confianca_yunet,
    )
    modelo_fer = criar_analisador_emocoes()

    # --- Acumuladores ---
    total_quadros = 0
    total_alertas = 0
    contagem_expressoes: dict[str, int] = {}
    contagem_estados: dict[str, int] = {
        "repouso": 0,
        "movimento_voluntario": 0,
        "agitacao_psicomotora": 0,
    }
    historico_movimento: deque[float] = deque(maxlen=config.janela_movimento)
    quadro_cinza_anterior = None
    cache_expressao: dict[int, str] = {}
    eventos_clinicos: list[dict] = []

    # Fator para mapear coordenadas do frame reduzido ao frame original
    fator_inverso = 1.0 / config.escala_redimensionamento

    progresso = tqdm(
        total=total_quadros_estimado if total_quadros_estimado > 0 else None,
        desc="Análise clínica em andamento",
    )

    try:
        while True:
            ok, quadro = captura.read()
            if not ok:
                break

            total_quadros += 1

            # --- Detecção facial ---
            # Reduz resolução para acelerar a inferência do YuNet
            quadro_reduzido = cv2.resize(
                quadro, None,
                fx=config.escala_redimensionamento,
                fy=config.escala_redimensionamento,
                interpolation=cv2.INTER_AREA,
            ) if config.escala_redimensionamento != 1.0 else quadro

            faces_reduzidas = detectar_faces(
                quadro_reduzido,
                detector,
                tamanho_minimo=config.tamanho_minimo_face,
                sobreposicao_nms=config.sobreposicao_nms,
            )
            faces = escalar_faces_para_original(faces_reduzidas, fator_inverso, largura, altura)

            # --- Análise de expressões faciais (amostrada a cada N quadros) ---
            analisar_este_quadro = (total_quadros % config.intervalo_analise_emocao == 0)
            expressoes_quadro: list[str] = []

            for idx, (x, y, w, h) in enumerate(faces, start=1):
                if analisar_este_quadro:
                    roi = quadro[y:y + h, x:x + w]
                    expressao = analisar_expressao(modelo_fer, roi)
                    cache_expressao[idx] = expressao
                    contagem_expressoes[expressao] = contagem_expressoes.get(expressao, 0) + 1
                else:
                    expressao = cache_expressao.get(idx, "indeterminado")

                expressoes_quadro.append(expressao)

            # --- Análise de estado motor ---
            quadro_cinza = cv2.cvtColor(quadro, cv2.COLOR_BGR2GRAY)

            if quadro_cinza_anterior is not None:
                indice = calcular_indice_movimento(quadro_cinza_anterior, quadro_cinza)
                historico_movimento.append(indice)  # deque descarta automaticamente entradas antigas
                resultado = classificar_estado_motor(indice, historico_movimento, config.limiar_z_alerta)
                contagem_estados[resultado.estado_motor] += 1

                if resultado.alerta_clinico:
                    total_alertas += 1
            else:
                # Primeiro quadro não tem referência anterior — registra como repouso
                resultado = None
                contagem_estados["repouso"] += 1

            quadro_cinza_anterior = quadro_cinza

            estado = resultado.estado_motor if resultado else "repouso"
            alerta = resultado.alerta_clinico if resultado else False
            indice_mov = resultado.indice_movimento if resultado else 0.0
            z_mov = resultado.z_movimento if resultado else 0.0

            # --- Anotação visual e registro ---
            _anotar_quadro(quadro, total_quadros, faces, expressoes_quadro, estado, alerta)

            eventos_clinicos.append({
                "quadro": total_quadros,
                "tempo_s": round(total_quadros / fps, 3),
                "rostos_detectados": len(faces),
                "expressoes_faciais": "|".join(expressoes_quadro) if expressoes_quadro else "",
                "estado_motor": estado,
                "indice_movimento": round(indice_mov, 6),
                "z_movimento": round(z_mov, 4),
                "alerta_clinico": int(alerta),
            })

            gravador.write(quadro)
            progresso.update(1)

    finally:
        progresso.close()
        captura.release()
        gravador.release()

    # --- Geração dos laudos ---
    duracao_s = total_quadros / fps if fps else 0.0
    expressao_predominante = (
        max(contagem_expressoes, key=contagem_expressoes.get)
        if contagem_expressoes else "indeterminado"
    )
    estado_predominante = (
        max(contagem_estados, key=contagem_estados.get)
        if contagem_estados else "indeterminado"
    )

    caminhos_saida = {
        "video_clinico_anotado": config.caminho_video_anotado,
        "registro_eventos_clinicos": config.caminho_registro_eventos,
        "laudo_clinico_txt": config.caminho_laudo_txt,
        "laudo_clinico_json": config.caminho_laudo_json,
    }

    gerar_laudo_json(
        total_quadros, duracao_s, fps, total_alertas,
        expressao_predominante, estado_predominante,
        contagem_expressoes, contagem_estados,
        caminhos_saida, config.caminho_laudo_json,
    )

    gerar_laudo_txt(
        total_quadros, duracao_s, total_alertas,
        expressao_predominante, estado_predominante,
        contagem_estados, contagem_expressoes,
        config.caminho_laudo_txt,
    )

    exportar_registro_eventos(eventos_clinicos, config.caminho_registro_eventos)

    print(f"\nAnálise clínica concluída — {total_quadros} quadros processados.")
    print(f"  Vídeo anotado              : {config.caminho_video_anotado}")
    print(f"  Laudo clínico (JSON)       : {config.caminho_laudo_json}")
    print(f"  Laudo clínico (TXT)        : {config.caminho_laudo_txt}")
    print(f"  Registro de eventos (CSV)  : {config.caminho_registro_eventos}")


if __name__ == "__main__":
    executar_pipeline()
