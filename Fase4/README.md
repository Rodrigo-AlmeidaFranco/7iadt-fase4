# Análise Clínica Comportamental por Vídeo

Pipeline de visão computacional que processa vídeos de consultas médicas para gerar laudos clínicos estruturados com detecção facial, reconhecimento de expressões e análise de estado motor.

## Funcionalidades

- Detecção facial com YuNet em 4 rotações (0°, 90°, 180°, 270°)
- Reconhecimento de 7 expressões faciais via FER + MTCNN
- Classificação de estado motor por z-score adaptativo em janela deslizante
- Geração de laudo em TXT, JSON e CSV quadro-a-quadro
- Vídeo anotado com indicadores clínicos

## Estrutura

```
Fase4/
├── assets/
│   ├── ConsultaMedicaViroseIA.mp4      # vídeo de entrada
│   └── face_detection_yunet.onnx       # modelo YuNet
├── outputs/                            # gerado automaticamente
├── src/
│   ├── config.py                       # hiperparâmetros centralizados
│   ├── detector_facial.py
│   ├── analisador_emocoes.py
│   ├── analisador_movimento.py
│   ├── gerador_laudo.py
│   └── main.py
└── requirements.txt
```

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install tensorflow            # instalar antes do fer
pip install -r requirements.txt
```

## Uso

O script deve ser executado a partir do diretório `Fase4/` (os caminhos em `config.py` são relativos):

```bash
cd Fase4
python src/main.py
```

Para personalizar parâmetros (limiares, janelas, caminhos), edite `src/config.py` ou instancie `ConfiguracaoPipeline` com valores customizados e passe para `executar_pipeline(config)`.

## Saídas

| Arquivo | Descrição |
|---|---|
| `video_clinico_anotado.mp4` | Vídeo com bounding boxes, expressões e alertas |
| `laudo_clinico.json` | Resumo estruturado para integração com sistemas |
| `laudo_clinico.txt` | Laudo legível para prontuário |
| `registro_eventos_clinicos.csv` | Dados quadro-a-quadro para análise posterior |

## Dependências principais

| Pacote | Função |
|---|---|
| `opencv-python` | Detecção facial (YuNet) e processamento de vídeo |
| `fer==22.4.0` | Reconhecimento de expressões faciais |
| `tensorflow` | Backend do FER |
| `numpy` / `pandas` | Análise numérica e exportação CSV |
| `tqdm` | Barra de progresso |
