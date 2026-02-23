---
title: ReviewCheckIA
emoji: 🎯
colorFrom: green
colorTo: green
sdk: docker
pinned: false
---

# 🇧🇷 Análise de Sentimento — Reviews Mercado Livre

Projeto de **NLP (Processamento de Linguagem Natural)** para classificação de sentimento em reviews de produtos do Mercado Livre usando fine-tuning do **BERTimbau** (BERT pré-treinado em Português).

## 📊 Objetivo

Classificar automaticamente reviews de consumidores em **3 categorias**:

| Sentimento | Rating | Label |
|-----------|--------|-------|
| 🔴 Negativo | ⭐ 1-2 | 0 |
| 🟡 Neutro | ⭐ 3 | 1 |
| 🟢 Positivo | ⭐ 4-5 | 2 |

## 🧠 Arquitetura

- **Modelo base**: `neuralmind/bert-base-portuguese-cased` (BERTimbau)
- **Fine-tuning**: Classification head com 3 classes
- **Balanceamento**: Class weights para lidar com desbalanceamento (~85% positivo)
## Como Executar

### 1. Instalar dependências

Para rodar em **CPU**:
```bash
pip install -r requirements.txt
```

Para rodar em **GPU AMD (ROCm)**:
```bash
# Reinstale o PyTorch build ROCm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
pip install -r requirements.txt
```

> [!NOTE]
> Para a **RX 6600**, é necessário setar o workaround no `~/.bashrc`:
> `export HSA_OVERRIDE_GFX_VERSION=10.3.0`

### 2. Pipeline completo
```bash
python3 main.py
```

Isso executa automaticamente todos os steps em sequência:
1. 📦 Pré-processamento
2. 📊 Análise Exploratória
3. 🔧 Preparação do Dataset
4. 🤖 Treinamento
5. 📊 Avaliação
6. 🔍 Insights de Mercado

### Opções
```bash
# Teste rápido (500 amostras, 1 epoch — ideal pra validar o pipeline)
python3 main.py --fast

# Pular treinamento (só EDA + insights)
python3 main.py --skip-train

# Definir número de epochs
python3 main.py --epochs 5

# Rodar apenas um step específico
python3 main.py --only 02_eda
```

### 3. Interface Web Interativa (Visual)
O projeto inclui um servidor FastAPI e um frontend moderno para testar o modelo em tempo real direto no navegador.

```bash
python3 app.py
```
Acesse [http://localhost:8000](http://localhost:8000) no seu navegador.

### 4. Inferência via Terminal (Texto Livre)
Agora você pode testar o modelo com qualquer frase:

```bash
# Rodar exemplos pré-definidos
python3 src/07_inference.py

# Analisar sua própria frase
python3 src/07_inference.py --text "O produto é fantástico, amei!"
```


## 📈 Análises Geradas

### EDA (`results/eda/`)
- Distribuição de ratings e sentimentos
- Comprimento de texto por sentimento
- Top palavras por sentimento
- Produtos mais avaliados

### Insights de Mercado (`results/insights/`)
- Consumidores insatisfeitos escrevem mais?
- Palavras exclusivas de reviews negativas
- Evolução temporal do sentimento
- Ranking de satisfação por produto

## 🛠 Tecnologias

- **PyTorch** + **HuggingFace Transformers**
- **BERTimbau** (BERT Português)
- **scikit-learn** (métricas, splits, class weights)
- **matplotlib** + **seaborn** (visualizações)
- **pandas** (manipulação de dados)
