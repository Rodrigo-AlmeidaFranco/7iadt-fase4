# Pipeline de Análise Clínica Comportamental em Vídeo

**FIAP — 7IADT · Tech Challenge — Fase 4**

| | |
|---|---|
| RM368747 | Hermes Vieira Júnior |
| RM368409 | Raphael Rodrigues Pereira |
| RM367428 | Rodrigo Franco |

**Repositório:** https://github.com/Rodrigo-AlmeidaFranco/7iadt-fase4

**Maio de 2026**

---

## 1. Introdução

Este documento apresenta o relatório técnico do projeto desenvolvido para o Tech Challenge da Fase 4 da pós-graduação em Inteligência Artificial da FIAP (7IADT). O projeto consiste em um pipeline de Inteligência Artificial voltado à análise clínica comportamental automatizada em vídeo, integrando técnicas de Visão Computacional, reconhecimento de expressões faciais e análise estatística de movimento.

A solução processa vídeos quadro a quadro, detecta rostos, classifica expressões faciais em terminologia clínica em português e avalia o estado motor dos indivíduos filmados, gerando laudos analíticos estruturados em múltiplos formatos. Toda a execução ocorre localmente, sem dependência de APIs externas.

### 1.1 Objetivos do Projeto

- Processar vídeos clínicos automaticamente
- Detectar rostos em tempo real com cobertura multi-rotação
- Analisar expressões faciais e classificá-las com terminologia clínica em PT-BR
- Identificar estados motores e eventos de agitação psicomotora
- Gerar laudos clínicos estruturados em JSON, TXT e CSV

### 1.2 Arquitetura de Módulos

O pipeline foi estruturado em módulos com responsabilidade única, aplicando o princípio de Separação de Responsabilidades (SoC):

| Módulo | Responsabilidade |
|---|---|
| `config.py` | Parâmetros e limiares clínicos centralizados |
| `detector_facial.py` | YuNet, detecção multi-rotação, NMS |
| `analisador_emocoes.py` | FER, MTCNN, mapa clínico PT-BR |
| `analisador_movimento.py` | Índice de movimento, z-score, estado motor |
| `gerador_laudo.py` | Exportação de laudos JSON, TXT e CSV |
| `main.py` | Orquestrador do pipeline |

---

## 2. Descrição do Fluxo Multimodal

O pipeline integra três modalidades de dados processadas sequencialmente em cada quadro do vídeo: dados visuais estáticos (detecção facial), dados de expressão facial (emoções) e dados temporais de movimento (fluxo óptico entre quadros consecutivos). A fusão das três modalidades em uma anotação clínica por quadro caracteriza a abordagem multimodal do sistema.

### 2.1 Visão Geral do Pipeline

```
ENTRADA: Vídeo Clínico  (37 MB · 30 FPS · 110,87 s · 3.326 quadros)
 |
 +-- MODALIDADE 1: Detecção Facial
 |     detector_facial.py
 |     YuNet ONNX  ·  4 rotações  ·  NMS (IoU <= 0,30)
 |
 +-- MODALIDADE 2: Expressões Faciais
 |     analisador_emocoes.py
 |     FER + MTCNN  ·  mapa clínico PT-BR  ·  cache N=10 quadros
 |
 +-- MODALIDADE 3: Estado Motor
       analisador_movimento.py
       cv2.absdiff  ·  z-score  ·  janela deslizante 90 quadros
             |
             +-- repouso
             +-- movimento_voluntario
             +-- agitacao_psicomotora  -->  ALERTA CLÍNICO (z > 3σ)
 |
SAÍDAS CLÍNICAS:
 video_clinico_anotado.mp4
 registro_eventos_clinicos.csv
 laudo_clinico.json  |  laudo_clinico.txt
```

### 2.2 Etapas Detalhadas

**Etapa 1 — Inicialização**

O módulo `config.py` centraliza todos os parâmetros em um `@dataclass` (`ConfiguracaoPipeline`). Os modelos YuNet e FER são instanciados uma única vez antes do loop de quadros, evitando overhead de carregamento em cada frame.

**Etapa 2 — Detecção Facial (`detector_facial.py`)**

O quadro é reduzido a 75% da resolução original para acelerar a inferência. O YuNet é executado em 4 rotações (0°, 90°, 180°, 270°), com cada detecção remapeada para o sistema de coordenadas original via `_mapear_caixa_para_original`. Todas as detecções das 4 rotações são combinadas e submetidas ao Non-Maximum Suppression (NMS, limiar IoU = 0,30), que elimina caixas duplicadas internamente ao módulo. As coordenadas são então escalonadas de volta à resolução original.

**Etapa 3 — Análise de Expressões Faciais (`analisador_emocoes.py`)**

A cada 10 quadros, a ROI de cada face detectada é extraída e processada pela FER com MTCNN. O resultado em inglês é traduzido para terminologia clínica em português via `MAPA_EMOCOES_CLINICO`. Nos 9 quadros intermediários, o último resultado é reutilizado via cache por índice de face. Quando o modelo não atinge confiança suficiente, o rótulo `indeterminado` é registrado.

**Etapa 4 — Análise de Estado Motor (`analisador_movimento.py`)**

O índice de movimento é calculado como a média das diferenças absolutas entre quadros consecutivos em escala de cinza (`cv2.absdiff`). Uma janela deslizante de 90 quadros (~3 segundos a 30 FPS) acumula o histórico. O z-score do quadro atual é calculado em relação à média e desvio padrão da janela, resultando na classificação do estado motor. A abordagem adaptativa via z-score ajusta-se automaticamente ao nível basal de atividade do paciente, sem necessidade de calibração manual.

**Etapa 5 — Fusão, Anotação e Exportação (`main.py` + `gerador_laudo.py`)**

O `main.py` funde os três resultados em anotações visuais sobre o quadro original e em um registro estruturado por quadro. O módulo `gerador_laudo.py` consolida todos os eventos ao final, exportando o laudo clínico em JSON, TXT e CSV.

---

## 3. Modelos Aplicados em Cada Tipo de Dado

A tabela a seguir sintetiza os modelos e técnicas aplicados a cada modalidade do pipeline:

| Modalidade | Modelo / Técnica | Saída |
|---|---|---|
| Detecção Facial | YuNet 2023 (ONNX) via OpenCV | Bounding boxes (x, y, w, h) |
| Expressão Facial | FER 22.4.0 + MTCNN (TensorFlow) | Rótulo clínico em PT-BR |
| Estado Motor | MAD + Z-score (janela deslizante) | Estado + alerta clínico |

### 3.1 Dados Visuais — Detecção Facial (YuNet)

| Atributo | Valor |
|---|---|
| Modelo | `face_detection_yunet.onnx` |
| Framework | OpenCV FaceDetectorYN |
| Formato | ONNX — execução local, sem dependência de nuvem |
| Tamanho do modelo | 227 KB |
| Confiança mínima | 0,75 |
| Tamanho mínimo de face | 45×45 pixels |
| Módulo | `detector_facial.py → detectar_faces()` |

O YuNet é uma CNN leve projetada para detecção facial em tempo real. A execução em 4 rotações compensa a ausência de invariância rotacional nativa, garantindo cobertura em cenários clínicos onde o paciente pode estar em posturas variadas. O NMS é aplicado internamente ao módulo `detector_facial.py`, encapsulando a complexidade do pipeline de detecção.

### 3.2 Dados de Expressão Facial — FER + MTCNN

| Atributo | Valor |
|---|---|
| Biblioteca | FER 22.4.0 |
| Pré-processador | MTCNN (`mtcnn=True`) — alinhamento facial |
| Classes (inglês) | happy, sad, angry, neutral, surprise, disgust, fear |
| Classes clínicas | alegria, tristeza, raiva, neutro, surpresa, desgosto, medo, indeterminado |
| Frequência | 1 análise a cada 10 quadros |
| Módulo | `analisador_emocoes.py → analisar_expressao()` |

O MTCNN realiza alinhamento facial antes da classificação, aumentando a acurácia em faces parcialmente giradas. O subamostramento de 10 quadros mitiga a latência do MTCNN sem perda clínica relevante. A tradução para português via `MAPA_EMOCOES_CLINICO` padroniza os rótulos para integração com sistemas de prontuário eletrônico.

### 3.3 Dados Temporais — Estado Motor e Alertas Clínicos

| Atributo | Valor |
|---|---|
| Técnica | Diferença absoluta entre quadros (`cv2.absdiff`) |
| Métrica | Mean Absolute Difference (MAD) em escala de cinza |
| Estatística | Z-score com janela deslizante de 90 quadros (~3 s a 30 FPS) |
| Limiar de alerta | z > 3,0 (regra dos 3 desvios padrão) |
| Estados motores | repouso / movimento_voluntario / agitacao_psicomotora |
| Módulo | `analisador_movimento.py → classificar_estado_motor()` |

A abordagem via z-score com janela deslizante é adaptativa ao contexto: o limiar de alerta ajusta-se automaticamente ao nível basal de atividade do paciente no trecho em análise, evitando falsos positivos em vídeos com alta atividade basal e falsos negativos em vídeos de repouso prolongado. O resultado é encapsulado em um `NamedTuple` tipado (`ResultadoMovimento`).

---

## 4. Resultados Obtidos

O pipeline foi executado sobre o vídeo de entrada incluído no repositório (https://github.com/Rodrigo-AlmeidaFranco/7iadt-fase4), com duração de 110,87 segundos a 30 FPS. Os resultados consolidados são apresentados a seguir.

### 4.1 Sumário Geral do Processamento

| Métrica | Valor |
|---|---|
| Quadros processados | 3.326 |
| Duração analisada | 110,87 segundos |
| Taxa de quadros | 30 FPS |
| Alertas clínicos detectados | 27 eventos |
| Taxa de alerta | 0,81% dos quadros |
| Expressão predominante | indeterminado (41 amostras) |
| Estado motor predominante | repouso — 1.897 quadros (57,0%) |

### 4.2 Distribuição de Estados Motores

| Estado Motor | Quadros | % |
|---|---|---|
| repouso | 1.897 | 57,0% |
| movimento_voluntario | 1.402 | 42,2% |
| agitacao_psicomotora | 27 | 0,8% |

O estado de repouso predominou em 57% do vídeo. Movimentos voluntários compõem 42,2% dos quadros. Os 27 eventos de agitação psicomotora (0,8%) correspondem aos alertas clínicos detectados pelo limiar z > 3,0.

### 4.3 Distribuição de Expressões Faciais

Amostradas a cada 10 quadros — 333 amostras totais:

| Expressão (PT-BR) | Amostras | % do Total |
|---|---|---|
| indeterminado | 41 | 12,3% |
| neutro | 38 | 11,4% |
| alegria | 34 | 10,2% |
| surpresa | 29 | 8,7% |
| raiva | 22 | 6,6% |
| tristeza | 17 | 5,1% |
| desgosto | 6 | 1,8% |
| medo | 5 | 1,5% |

O rótulo `indeterminado` — o mais frequente com 12,3% — indica quadros onde o modelo FER não atingiu confiança suficiente. Isso ocorre tipicamente em faces de perfil ou parcialmente ocluídas. Representa ausência de dado clínico válido, não uma condição patológica.

### 4.4 Estrutura dos Arquivos de Saída

**`registro_eventos_clinicos.csv`**

| Coluna | Descrição |
|---|---|
| `quadro` | Número sequencial do quadro |
| `tempo_s` | Timestamp em segundos |
| `rostos_detectados` | Quantidade de faces no quadro |
| `expressoes_faciais` | Expressões separadas por `\|` (ex.: `alegria\|neutro`) |
| `estado_motor` | `repouso`, `movimento_voluntario` ou `agitacao_psicomotora` |
| `indice_movimento` | Mean Absolute Difference entre quadros consecutivos |
| `z_movimento` | Z-score em relação à janela deslizante |
| `alerta_clinico` | `1` quando z > 3 desvios padrão, `0` caso contrário |

**`laudo_clinico.json` — estrutura**

```json
{
  "total_quadros_analisados": 3326,
  "duracao_segundos": 110.87,
  "taxa_quadros_fps": 30.0,
  "alertas_clinicos_detectados": 27,
  "expressao_facial_predominante": "indeterminado",
  "estado_motor_predominante": "repouso",
  "distribuicao_expressoes_faciais": { "indeterminado": 41, "neutro": 38, "..." : "..." },
  "distribuicao_estados_motores": { "repouso": 1897, "movimento_voluntario": 1402, "..." : "..." },
  "arquivos_gerados": {
    "video_clinico_anotado": "outputs/video_clinico_anotado.mp4",
    "registro_eventos_clinicos": "outputs/registro_eventos_clinicos.csv",
    "laudo_clinico_json": "outputs/laudo_clinico.json",
    "laudo_clinico_txt": "outputs/laudo_clinico.txt"
  }
}
```

---

## 5. Exemplos de Alertas Clínicos Detectados

Os 27 alertas clínicos (z > 3,0) correspondem a picos do índice de movimento que excedem 3 desvios padrão da média local da janela deslizante. A análise dos padrões identificou três categorias distintas de evento:

| Tipo | Característica | Provável Origem |
|---|---|---|
| Agitação Psicomotora | 2–4 quadros consecutivos com z > 3 desvios, face detectada | Movimento brusco do paciente |
| Perturbação Ambiental | Spike isolado sem face detectada no quadro de alerta | Variação de iluminação ou câmera |
| Corte de Cena | Spike isolado de 1 quadro, retorno imediato ao basal | Edição do vídeo |

### 5.1 Tipo 1 — Agitação Psicomotora

Sequência de 2 a 4 quadros consecutivos com z > 3 desvios padrão, com rosto detectado nos quadros adjacentes. A elevação do índice é gradual e seguida de decaimento. Expressões como `raiva` ou `surpresa` são frequentemente registradas no cache dos quadros vizinhos. Este é o padrão clinicamente mais relevante, indicando movimento físico brusco do indivíduo filmado.

### 5.2 Tipo 2 — Perturbação Ambiental

Spike isolado de z > 3 desvios padrão sem face detectada no quadro de alerta (`rostos_detectados = 0`). Indica variação brusca de iluminação ou deslocamento de câmera, não de sujeito. Em análise clínica supervisionada, esses eventos devem ser filtrados ou anotados como artefatos ambientais.

### 5.3 Tipo 3 — Corte de Cena

Spike isolado de exatamente 1 quadro com retorno imediato ao nível basal no quadro seguinte. Corresponde a cortes de edição no vídeo. Característica: o índice de movimento do quadro pós-alerta é tipicamente abaixo da média histórica.

### 5.4 Exemplo de Registro de Alerta no CSV

| quadro | tempo_s | estado_motor | indice_mov. | z_movimento | alerta_clinico |
|---|---|---|---|---|---|
| 345 | 11,50 | movimento_voluntario | 12,40 | 1,87 | 0 |
| 346 | 11,53 | movimento_voluntario | 18,90 | 2,54 | 0 |
| 347 | 11,57 | agitacao_psicomotora | 34,92 | 3,61 | 1 |
| 348 | 11,60 | agitacao_psicomotora | 31,15 | 3,28 | 1 |
| 349 | 11,63 | movimento_voluntario | 11,20 | 1,62 | 0 |
