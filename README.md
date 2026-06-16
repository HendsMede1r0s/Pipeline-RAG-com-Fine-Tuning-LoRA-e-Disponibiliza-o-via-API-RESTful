# 🤖 Pipeline RAG com Fine-Tuning LoRA e Disponibilização via API RESTful

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-00C7B7?logo=fastapi)](https://fastapi.tiangolo.com/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Transformers-FFD21E)](https://huggingface.co/)
[![LoRA](https://img.shields.io/badge/PEFT-LoRA-FF6B35)](https://huggingface.co/docs/peft/en/developer_guides/lora)

---

## 📋 Visão Geral

Este projeto implementa um **pipeline completo** de:

1. **Geração de Dataset Instruction Tuning** — Extração de conhecimento de documentos PDF e geração automatizada de pares `instruction-input-output` no formato JSONL.
2. **Fine-Tuning com LoRA** — Ajuste eficiente de 4 modelos de linguagem (causais e seq2seq) utilizando a técnica **Low-Rank Adaptation (LoRA)** via PEFT.
3. **API RESTful com FastAPI** — Servidor com interface web para interação em tempo real com todos os modelos finos carregados em memória.
4. **Avaliação Comparativa** — Notebook dedicado para avaliação de métricas (ROUGE, BLEU, Perplexidade, entre outras) e comparação entre os modelos.

---

## 🧠 Modelos Fine-Tunados

| Modelo | Arquitetura | Parâmetros | ID no Sistema |
|--------|-------------|-----------|---------------|
| **BART Base** (`facebook/bart-base`) | Seq2Seq (Encoder-Decoder) | ~140M | `seq2seq-bart` |
| **Blenderbot 400M** (`facebook/blenderbot-400M-distill`) | Seq2Seq (Encoder-Decoder) | ~400M | `seq2seq-blenderbot` |
| **Pythia 70M** (`EleutherAI/pythia-70m`) | Causal (Decoder-only) | ~70M | `causal-pythia` |
| **Qwen 2.5 0.5B** (`Qwen/Qwen2.5-0.5B`) | Causal (Decoder-only) | ~500M | `causal-qwen` |

### Por que LoRA?

A técnica **Low-Rank Adaptation (LoRA)** permite o fine-tuning de modelos com dezenas a centenas de milhões de parâmetros utilizando **pouca memória GPU**, congelando os pesos originais e treinando apenas matrizes de adaptação de baixo posto. Isso mantém o custo computacional acessível mesmo para hardware modesto.

---

## 📂 Estrutura do Projeto

```
📦 Pipeline-RAG-com-Fine-Tuning-LoRA-e-Disponibiliza-o-via-API-RESTful
├── main.py                    # Servidor FastAPI com os 4 modelos carregados
├── dataset.jsonl              # Dataset instruction tuning (107 exemplos)
├── RelatGestSaud.pdf          # Documento-fonte para geração do dataset
│
├── RAG_llm.ipynb              # Notebook: geração de dataset instruction a partir de PDF
├── lora_bart.ipynb            # Notebook: fine-tuning LoRA do BART (seq2seq)
├── lora_blenderbolt.ipynb     # Notebook: fine-tuning LoRA do Blenderbot (seq2seq)
├── lora_pythia.ipynb          # Notebook: fine-tuning LoRA do Pythia (causal)
├── lora_qwen.ipynb            # Notebook: fine-tuning LoRA do Qwen 2.5 (causal)
├── avaliacao_modelo_finetuned.ipynb  # Notebook: avaliação comparativa de métricas
│
├── modelos_finais/            # Adapters LoRA salvos após fine-tuning
│   ├── lora_seq2seq_model_1/  # BART fine-tunado
│   ├── lora_seq2seq_model_2/  # Blenderbot fine-tunado
│   ├── lora_causal_model_1/   # Pythia fine-tunado
│   └── lora_causal_model_2/   # Qwen 2.5 fine-tunado
│
├── Resultados/                # Checkpoints e estados dos treinos
│   ├── resultados_bart/
│   ├── resultados_blenderbot/
│   ├── resultados_pythia/
│   └── resultados_qwen/
│
├── static/
│   └── index.html             # Interface web (frontend) da API
│
├── avaliacao_metricas_BART.png        # Gráficos de avaliação
├── avaliacao_metricas_Blenderbot.png
├── avaliacao_metricas_Pythia.png
├── avaliacao_metricas_Qwen.png
├── resultados_detalhados_modelos.csv  # Tabela comparativa de métricas
│
└── .gitignore
```

---

## 🚀 Como Executar

### 1. Requisitos

- Python 3.10+
- pip / conda
- GPU com CUDA (recomendado, mas funcional em CPU)

### 2. Instalação

```bash
# Clone o repositório
git clone https://github.com/HendsMede1r0s/Pipeline-RAG-com-Fine-Tuning-LoRA-e-Disponibiliza-o-via-API-RESTful.git
cd Pipeline-RAG-com-Fine-Tuning-LoRA-e-Disponibiliza-o-via-API-RESTful

# Instale as dependências
pip install -r requirements.txt
```

> **Nota:** Não há arquivo `requirements.txt` no repositório. Instale manualmente os pacotes necessários:

```bash
pip install torch transformers accelerate peft datasets fastapi uvicorn pydantic pypdf rouge-score evaluate bert-score
```

### 3. Executar a API

```bash
python main.py
```

O servidor será iniciado em **http://localhost:8000**. A interface web estará disponível acessando esta URL no navegador.

### 4. Endpoints da API

| Método | Rota | Descrição |
|--------|------|-----------|
| `GET` | `/` | Página web do chat |
| `GET` | `/modelos` | Lista modelos disponíveis |
| `POST` | `/chat` | Envia mensagem para um modelo |

#### Exemplo de requisição POST `/chat`

```json
{
  "modelo": "causal-qwen",
  "mensagem": "Explique o que é o SUS",
  "max_tokens": 150,
  "temperatura": 0.7
}
```

#### Resposta

```json
{
  "resposta": "O SUS (Sistema Único de Saúde) é o sistema público de saúde do Brasil...",
  "modelo": "causal-qwen",
  "tokens_gerados": 67
}
```

---

## 🔬 Fine-Tuning (LoRA)

Cada notebook de fine-tuning segue a mesma estrutura:

1. **Carregamento do modelo base** (pré-treinado do HuggingFace Hub)
2. **Aplicação da configuração LoRA** usando o PEFT (rank `r=16`, alpha `lora_alpha=32`)
3. **Tokenização do dataset** (`dataset.jsonl`) no formato instruction-tuning
4. **Treinamento** com `Trainer` do Transformers
5. **Salvamento** do adapter LoRA em `modelos_finais/`

### Dataset Instruction Tuning

O dataset foi gerado a partir do documento **Relatório de Gestão do Ministério da Saúde 2021** (`RelatGestSaud.pdf`), contendo 107 exemplos no formato:

```jsonl
{"instruction": "Pergunta sobre o documento", "input": "Trecho do documento", "output": "Resposta esperada"}
```

---

## 📊 Avaliação

A avaliação comparativa foi realizada no notebook `avaliacao_modelo_finetuned.ipynb`, considerando as seguintes métricas:

- **ROUGE-1 / ROUGE-2 / ROUGE-L** — Similaridade de n-gramas com o texto de referência
- **BLEU** — Precisão de n-gramas (tradução automática)
- **Perplexidade (PPL)** — Mede o quão "surpreso" o modelo fica com o texto
- **BERTScore** — Similaridade semântica via embeddings BERT

### Resultados

| Modelo | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | PPL | BERTScore |
|--------|---------|---------|---------|------|-----|-----------|
| BART | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Blenderbot | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Pythia | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Qwen 2.5 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

> 📈 Gráficos comparativos disponíveis nos arquivos `avaliacao_metricas_*.png`.

---

## 🖥️ Interface Web

O frontend (em `static/index.html`) oferece:

- **Seleção de modelo** entre os 4 disponíveis
- **Controle de temperatura** (criatividade das respostas)
- **Controle de tokens máximos** (tamanho da resposta)
- **Contador de tokens** gerados por resposta
- Indicador visual de modelo ativo

---

## 📚 Notebooks

| Notebook | Descrição |
|----------|-----------|
| `RAG_llm.ipynb` | Geração de dataset instruction a partir de PDF, usando modelos seq2seq e causais para criação automática de pares pergunta-resposta |
| `lora_bart.ipynb` | Fine-tuning LoRA do BART Base |
| `lora_blenderbolt.ipynb` | Fine-tuning LoRA do Blenderbot 400M |
| `lora_pythia.ipynb` | Fine-tuning LoRA do Pythia 70M |
| `lora_qwen.ipynb` | Fine-tuning LoRA do Qwen 2.5 0.5B |
| `avaliacao_modelo_finetuned.ipynb` | Avaliação quantitativa e comparativa dos modelos |

---

## 🛠️ Tecnologias Utilizadas

- **[PyTorch](https://pytorch.org/)** — Framework de deep learning
- **[HuggingFace Transformers](https://huggingface.co/docs/transformers)** — Modelos pré-treinados e tokenizadores
- **[PEFT](https://huggingface.co/docs/peft)** — Parameter-Efficient Fine-Tuning (LoRA)
- **[Datasets](https://huggingface.co/docs/datasets)** — Carregamento e processamento de datasets
- **[FastAPI](https://fastapi.tiangolo.com/)** — Framework web para API RESTful
- **[Uvicorn](https://www.uvicorn.org/)** — Servidor ASGI
- **[Pydantic](https://docs.pydantic.dev/)** — Validação de schemas
- **[PyPDF2 / pypdf](https://pypi.org/project/pypdf/)** — Extração de texto de PDFs
- **[Evaluate](https://huggingface.co/docs/evaluate)** — Métricas de avaliação (ROUGE, BLEU, BERTScore)

---

## 👨‍💻 Autor

**Henderson Medeiros**

[![GitHub](https://img.shields.io/badge/GitHub-HendsMede1r0s-181717?logo=github)](https://github.com/HendsMede1r0s)

---