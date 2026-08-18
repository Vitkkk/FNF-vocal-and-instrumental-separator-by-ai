# FNF Vocal and Instrumental Separator by AI

Projeto experimental para separar **vocais de Friday Night Funkin'** da instrumental usando modelos treinados especificamente em músicas de FNF.

## Objetivo

Separadores genéricos de voz normalmente são treinados para música convencional e tendem a confundir chromatics, vozes altamente processadas, synth-like vocals, screams, bitcrush, formant shifting e outros elementos comuns em FNF com a instrumental.

A proposta deste projeto é construir um dataset supervisionado com pares/trios reais de:

- `mix` — música completa;
- `instrumental` — instrumental associada;
- `vocals` — stem vocal verdadeiro quando disponível.

O modelo deve aprender diretamente a relação:

`mix -> vocals + instrumental`

sem depender da hipótese de que `mix - instrumental = vocals`, porque renders, masterização, compressão, encoding e versões diferentes podem impedir cancelamento matemático perfeito.

## Primeiro gabarito

O primeiro caso documentado é **Maniac (His Apocalypse Mix)**. Os detalhes e medições ficam em `ground_truth/maniac_his_apocalypse_mix/README.md`.

> Os arquivos de áudio usados como referência não são redistribuídos neste repositório. Apenas metadados, medições e scripts de preparação devem ser versionados.

## Preparação do dataset

Requisitos:

- Python 3.10+;
- `ffmpeg` disponível no `PATH`;
- dependências de `requirements.txt`.

Instalação:

```bash
pip install -r requirements.txt
```

Exemplo com mix, instrumental e vocal target:

```bash
python scripts/prepare_dataset.py \
  --mix song.mp3 \
  --instrumental inst.mp3 \
  --vocals vocals.mp3 \
  --dataset-id gt_001_maniac_his_apocalypse_mix \
  --output datasets/gt_001_maniac_his_apocalypse_mix
```

Por padrão o preparador:

1. converte tudo para WAV float32 estéreo em 44.1 kHz;
2. estima o offset de cada referência em relação ao mix;
3. encontra a região temporal comum;
4. corta segmentos de 8 segundos sem overlap;
5. cria splits determinísticos de treino, validação e teste;
6. gera `manifest.jsonl` e `metadata.json` com hashes, offsets e caminhos dos segmentos.

O alinhamento é apenas uma medida temporal. O script **não** assume que `mix == instrumental + vocals` e não altera os targets para forçar essa igualdade.
