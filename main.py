# =============================================================================
# LABORATÓRIO: Clone do ChatGPT com FastAPI + Modelos HuggingFace (LoRA)
# =============================================================================
# Este arquivo é o coração da aplicação. Ele:
#   1. Carrega os modelos de linguagem (base e fine-tunado com LoRA)
#   2. Expõe uma API REST via FastAPI
#   3. Serve o front-end estático (HTML/CSS/JS)
#   4. Processa mensagens do usuário e retorna respostas geradas pelos modelos
# =============================================================================

# --- Importações padrão do Python ---
import os
import logging
from typing import Optional

# --- Importações do FastAPI ---
# FastAPI: framework moderno para criação de APIs em Python
# StaticFiles: serve arquivos estáticos (HTML, CSS, JS)
# HTMLResponse: retorna respostas HTTP em HTML
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# --- Pydantic: validação de dados ---
# BaseModel: classe base para definir o "shape" dos dados que a API recebe/envia
from pydantic import BaseModel

# --- HuggingFace Transformers ---
# AutoModelForCausalLM : carrega qualquer modelo de geração de texto automaticamente
# AutoTokenizer        : carrega o tokenizador correspondente ao modelo
# pipeline             : abstração de alto nível para tarefas de NLP
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# --- PyTorch ---
# Biblioteca de deep learning; usada para inferência nos modelos
import torch

# =============================================================================
# CONFIGURAÇÃO DE LOGGING
# =============================================================================
# Configura o sistema de logs para exibir mensagens informativas no terminal
# Isso ajuda a acompanhar o que está acontecendo durante a execução

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# =============================================================================
# INSTÂNCIA DA APLICAÇÃO FASTAPI
# =============================================================================
# Criamos o objeto principal da aplicação.
# O título e a versão aparecem na documentação automática em /docs

app = FastAPI(
    title="ChatGPT Clone - Laboratório LLM",
    description="API para interagir com modelos de linguagem (base e fine-tunado com LoRA)",
    version="1.0.0",
)

# --- Middleware CORS ---
# CORS (Cross-Origin Resource Sharing) permite que o navegador faça requisições
# de uma origem diferente da API. Em laboratório, permitimos tudo ("*").
# Em produção, restrinja para domínios específicos!
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Permite qualquer origem
    allow_methods=["*"],   # Permite qualquer método HTTP (GET, POST, etc.)
    allow_headers=["*"],   # Permite qualquer cabeçalho
)

# =============================================================================
# DICIONÁRIO GLOBAL DE MODELOS
# =============================================================================
# Armazena os modelos e tokenizadores já carregados em memória.
# Usar um dicionário evita recarregar o modelo a cada requisição (muito lento!).
# Chave   → nome amigável do modelo (string)
# Valor   → dicionário com "model", "tokenizer" e "pipeline"

MODELS: dict = {}

# =============================================================================
# CARREGAMENTO DOS MODELOS
# =============================================================================

def carregar_modelo_base() -> dict:
    """
    Carrega o modelo base DistilGPT-2 diretamente do HuggingFace Hub.

    DistilGPT-2 é uma versão compacta do GPT-2, ideal para laboratório
    porque é rápido e leve, sem precisar de GPU potente.

    Retorna um dicionário com 'model', 'tokenizer' e 'pipeline'.
    """
    logger.info("Carregando modelo BASE (distilgpt2) do HuggingFace Hub...")

    # AutoTokenizer.from_pretrained():
    #   Baixa e carrega o tokenizador do modelo.
    #   O tokenizador converte texto → tokens (números) e tokens → texto.
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

    # Garantir que o token de padding esteja definido.
    # Modelos GPT não têm pad_token por padrão; usamos o eos_token como substituto.
    # Sem isso, batches de textos com tamanhos diferentes causariam erros.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("  → pad_token definido como eos_token")

    # AutoModelForCausalLM.from_pretrained():
    #   Baixa e carrega os pesos do modelo de linguagem causal (geração de texto).
    #   "Causal" significa que o modelo só olha para tokens anteriores (autoregressive).
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")

    # Coloca o modelo em modo de inferência (desativa dropout, batch norm em treino, etc.)
    # Isso economiza memória e acelera a geração.
    model.eval()

    # pipeline() é uma abstração que junta modelo + tokenizador + lógica de geração.
    # "text-generation" indica que queremos gerar continuações de texto.
    # device=-1 força o uso de CPU (use device=0 para GPU, se disponível).
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1   # -1 = CPU | 0 = primeira GPU
    )

    logger.info("  ✓ Modelo BASE carregado com sucesso!")
    return {"model": model, "tokenizer": tokenizer, "pipeline": pipe}


def carregar_modelo_finetuned() -> dict:
    """
    Carrega o modelo fine-tunado com LoRA (Low-Rank Adaptation).

    LoRA é uma técnica de fine-tuning eficiente que adiciona pequenas matrizes
    de adaptação aos pesos originais do modelo, sem modificá-los diretamente.
    Resultado: modelo especializado, muito menor que um fine-tune completo.

    Caminhos locais esperados:
      - ./lora_finetuned_model/   → adaptadores LoRA + config do modelo
      - ./distilgpt2_tokenizer/   → tokenizador salvo localmente

    Se os arquivos locais não existirem, faz fallback para o modelo base.
    """
    model_path = "./lora_finetuned_model"
    tokenizer_path = "./distilgpt2_tokenizer"

    # Verifica se os arquivos do modelo fine-tunado existem localmente
    if not os.path.exists(model_path):
        logger.warning(
            f"Diretório '{model_path}' não encontrado. "
            "Usando modelo base como fallback para o 'fine-tunado'."
        )
        # Fallback: usa o modelo base para não quebrar a aplicação
        return carregar_modelo_base()

    logger.info(f"Carregando modelo FINE-TUNADO (LoRA) de '{model_path}'...")

    # Carrega o tokenizador do caminho local
    # Se o tokenizador local não existir, tenta o caminho do modelo
    tok_path = tokenizer_path if os.path.exists(tokenizer_path) else model_path
    tokenizer = AutoTokenizer.from_pretrained(tok_path)

    # Mesma garantia do pad_token para o modelo fine-tunado
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("  → pad_token definido como eos_token (fine-tunado)")

    # Carrega o modelo com os adaptadores LoRA.
    # O from_pretrained é inteligente: se detectar configs LoRA, carrega corretamente.
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1
    )

    logger.info("  ✓ Modelo FINE-TUNADO carregado com sucesso!")
    return {"model": model, "tokenizer": tokenizer, "pipeline": pipe}


# =============================================================================
# EVENTO DE INICIALIZAÇÃO DA APLICAÇÃO
# =============================================================================
# Este bloco é executado UMA VEZ quando o servidor FastAPI sobe.
# É o lugar ideal para carregar recursos pesados (modelos, conexões de banco, etc.)

@app.on_event("startup")
async def startup_event():
    """
    Carrega todos os modelos na inicialização do servidor.
    Assim, a primeira requisição não precisa esperar o carregamento.
    """
    global MODELS
    logger.info("=" * 60)
    logger.info("  INICIANDO SERVIDOR - Carregando modelos de linguagem...")
    logger.info("=" * 60)

    # Carrega e registra o modelo base
    MODELS["distilgpt2-base"] = carregar_modelo_base()

    # Carrega e registra o modelo fine-tunado
    MODELS["distilgpt2-lora"] = carregar_modelo_finetuned()

    logger.info("=" * 60)
    logger.info(f"  ✓ {len(MODELS)} modelo(s) disponível(is): {list(MODELS.keys())}")
    logger.info("=" * 60)


# =============================================================================
# MODELOS PYDANTIC (Schemas de Request/Response)
# =============================================================================
# Pydantic valida automaticamente os dados recebidos pela API.
# Se o JSON não bater com o schema, FastAPI retorna 422 Unprocessable Entity.

class ChatRequest(BaseModel):
    """
    Schema da requisição de chat.

    Campos:
      - modelo   : nome do modelo a usar (deve existir em MODELS)
      - mensagem : texto do usuário
      - max_tokens: máximo de tokens a gerar na resposta (padrão: 150)
      - temperatura: controla aleatoriedade (0.0 = determinístico, 1.0 = criativo)
    """
    modelo: str
    mensagem: str
    max_tokens: Optional[int] = 150
    temperatura: Optional[float] = 0.7


class ChatResponse(BaseModel):
    """
    Schema da resposta de chat.

    Campos:
      - resposta : texto gerado pelo modelo
      - modelo   : qual modelo foi usado
      - tokens_gerados: quantidade de tokens na resposta
    """
    resposta: str
    modelo: str
    tokens_gerados: int


# =============================================================================
# ENDPOINTS DA API
# =============================================================================

@app.get("/modelos", response_class=JSONResponse)
async def listar_modelos():
    """
    GET /modelos

    Retorna a lista de modelos disponíveis no servidor.
    O front-end usa este endpoint para popular o dropdown de seleção.

    Exemplo de resposta:
    {
        "modelos": [
            {"id": "distilgpt2-base", "nome": "DistilGPT-2 Base"},
            {"id": "distilgpt2-lora", "nome": "DistilGPT-2 Fine-tunado (LoRA)"}
        ]
    }
    """
    modelos_info = {
        "distilgpt2-base": {
            "id": "distilgpt2-base",
            "nome": "DistilGPT-2 Base",
            "descricao": "Modelo base pré-treinado, sem fine-tuning."
        },
        "distilgpt2-lora": {
            "id": "distilgpt2-lora",
            "nome": "DistilGPT-2 Fine-tunado (LoRA)",
            "descricao": "Modelo adaptado com LoRA para domínio específico."
        },
    }

    # Filtra apenas os modelos que foram carregados com sucesso
    disponiveis = [
        info for key, info in modelos_info.items()
        if key in MODELS
    ]

    return {"modelos": disponiveis}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    POST /chat

    Endpoint principal: recebe a mensagem do usuário, gera uma resposta
    usando o modelo selecionado e retorna o texto gerado.

    Corpo da requisição (JSON):
    {
        "modelo": "distilgpt2-base",
        "mensagem": "What is artificial intelligence?",
        "max_tokens": 150,
        "temperatura": 0.7
    }
    """
    # --- Validação: modelo existe? ---
    if request.modelo not in MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"Modelo '{request.modelo}' não encontrado. "
                   f"Disponíveis: {list(MODELS.keys())}"
        )

    # --- Validação: mensagem não vazia ---
    if not request.mensagem.strip():
        raise HTTPException(
            status_code=400,
            detail="A mensagem não pode ser vazia."
        )

    logger.info(f"[CHAT] Modelo='{request.modelo}' | Mensagem='{request.mensagem[:50]}...'")

    # Recupera o pipeline do modelo selecionado
    pipe = MODELS[request.modelo]["pipeline"]

    try:
        # ---------------------------------------------------------------
        # GERAÇÃO DE TEXTO
        # ---------------------------------------------------------------
        # pipe() chama o modelo para gerar texto a partir do prompt.
        #
        # Parâmetros importantes:
        #   max_new_tokens : número máximo de NOVOS tokens (não inclui o prompt)
        #   temperature    : controla aleatoriedade
        #                    0.1 → quase determinístico
        #                    1.0 → muito aleatório/criativo
        #   do_sample      : True = amostragem estocástica | False = greedy decoding
        #   top_p          : nucleus sampling (considera tokens que somam p% de prob.)
        #   pad_token_id   : evita erro de padding no final da geração
        #   num_return_sequences: quantas alternativas gerar (usamos 1)

        tokenizer = MODELS[request.modelo]["tokenizer"]

        resultado = pipe(
            request.mensagem,
            max_new_tokens=request.max_tokens,
            temperature=request.temperatura,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
        )

        # O pipeline retorna uma lista de dicionários.
        # resultado[0]["generated_text"] contém o texto COMPLETO (prompt + geração).
        texto_completo = resultado[0]["generated_text"]

        # Removemos o prompt original para devolver apenas a resposta gerada.
        # Isso imita o comportamento de um chatbot que não repete a pergunta.
        resposta = texto_completo[len(request.mensagem):].strip()

        # Se a resposta ficou vazia (modelo só repetiu o prompt), retorna aviso
        if not resposta:
            resposta = "[O modelo não gerou texto adicional. Tente aumentar max_tokens.]"

        # Conta quantos tokens foram gerados (aproximação via tokenizador)
        tokens_gerados = len(tokenizer.encode(resposta))

        logger.info(f"  ✓ Resposta gerada: {tokens_gerados} tokens")

        # Retorna a resposta estruturada conforme ChatResponse
        return ChatResponse(
            resposta=resposta,
            modelo=request.modelo,
            tokens_gerados=tokens_gerados
        )

    except Exception as e:
        # Captura qualquer erro durante a geração e retorna HTTP 500
        logger.error(f"Erro na geração: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao gerar resposta: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """
    GET /health

    Endpoint de verificação de saúde do servidor.
    Retorna quais modelos estão carregados e prontos para uso.
    Útil para monitoramento e debugging em laboratório.
    """
    return {
        "status": "ok",
        "modelos_carregados": list(MODELS.keys()),
        "quantidade": len(MODELS)
    }


# =============================================================================
# SERVIR O FRONT-END (HTML/CSS/JS)
# =============================================================================
# Monta o diretório "static" para servir arquivos estáticos.
# O index.html será acessível em http://localhost:8000/

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    """
    GET /

    Serve a página principal do chat.
    Lê o arquivo HTML do diretório static/ e retorna seu conteúdo.
    """
    html_path = os.path.join("static", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


# =============================================================================
# PONTO DE ENTRADA (execução direta)
# =============================================================================
# Este bloco só executa quando rodamos `python main.py` diretamente.
# Em produção, usa-se `uvicorn main:app` para melhor controle.

if __name__ == "__main__":
    import uvicorn

    # uvicorn: servidor ASGI de alta performance para aplicações FastAPI/Starlette
    # host="0.0.0.0"  → aceita conexões de qualquer IP (necessário em laboratório)
    # port=8000       → porta padrão da aplicação
    # reload=True     → reinicia automaticamente ao salvar alterações no código
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Desative em produção!
    )
