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
