"""
Geração de laudos clínicos e exportação de dados analíticos.

Produz saídas em múltiplos formatos para integração com sistemas
de prontuário eletrônico, BI clínico e análise posterior em Python.
"""

import json
import os

import pandas as pd


def garantir_diretorio(caminho: str) -> None:
    """Cria o diretório pai do caminho fornecido, caso não exista."""
    diretorio = os.path.dirname(caminho)
    if diretorio:
        os.makedirs(diretorio, exist_ok=True)


def exportar_registro_eventos(eventos: list[dict], caminho_csv: str) -> None:
    """Exporta o registro quadro-a-quadro dos eventos clínicos para CSV."""
    pd.DataFrame(eventos).to_csv(caminho_csv, index=False, encoding="utf-8")


def gerar_laudo_json(
    total_quadros: int,
    duracao_s: float,
    fps: float,
    total_alertas: int,
    expressao_predominante: str,
    estado_motor_predominante: str,
    contagem_expressoes: dict,
    contagem_estados: dict,
    caminhos_saida: dict,
    caminho_json: str,
) -> None:
    """Serializa e salva o laudo clínico estruturado em JSON."""
    laudo = {
        "total_quadros_analisados": total_quadros,
        "duracao_segundos": round(duracao_s, 3),
        "taxa_quadros_fps": fps,
        "alertas_clinicos_detectados": total_alertas,
        "expressao_facial_predominante": expressao_predominante,
        "estado_motor_predominante": estado_motor_predominante,
        "distribuicao_expressoes_faciais": contagem_expressoes,
        "distribuicao_estados_motores": contagem_estados,
        "arquivos_gerados": caminhos_saida,
    }

    with open(caminho_json, "w", encoding="utf-8") as arquivo:
        json.dump(laudo, arquivo, ensure_ascii=False, indent=2)


def gerar_laudo_txt(
    total_quadros: int,
    duracao_s: float,
    total_alertas: int,
    expressao_predominante: str,
    estado_motor_predominante: str,
    contagem_estados: dict,
    contagem_expressoes: dict,
    caminho_txt: str,
) -> None:
    """
    Gera o laudo clínico em formato texto legível.
    Estruturado para incorporação direta em prontuários ou relatórios acadêmicos.
    """
    linhas = [
        "LAUDO CLÍNICO — ANÁLISE COMPORTAMENTAL POR VÍDEO",
        "=" * 52,
        f"Duração analisada              : {duracao_s:.2f} s ({total_quadros} quadros)",
        f"Alertas clínicos detectados    : {total_alertas}",
        f"Expressão facial predominante  : {expressao_predominante}",
        f"Estado motor predominante      : {estado_motor_predominante}",
        "",
        "Distribuição de estados motores:",
    ]

    for estado, quantidade in sorted(contagem_estados.items(), key=lambda x: -x[1]):
        linhas.append(f"  {estado}: {quantidade} quadros")

    linhas += ["", "Distribuição de expressões faciais (amostradas):"]
    for expressao, quantidade in sorted(contagem_expressoes.items(), key=lambda x: -x[1]):
        linhas.append(f"  {expressao}: {quantidade}")

    with open(caminho_txt, "w", encoding="utf-8") as arquivo:
        arquivo.write("\n".join(linhas))
