# =============================================================================
# LABORATÓRIO: Clone do ChatGPT com FastAPI + Modelos HuggingFace (LoRA)
# =============================================================================

import os
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
import torch

# --- LOGGING ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- CAMINHOS ---
PATH_BART = "./lora_seq2seq_model_1"
PATH_BLENDERBOT = "./lora_seq2seq_model_2"
PATH_PYTHIA = "./lora_causal_model_1"
PATH_QWEN = "./lora_causal_model_2"

HF_BART = "facebook/bart-base"
HF_BLENDERBOT = "facebook/blenderbot-400M-distill"
HF_PYTHIA = "EleutherAI/pythia-70m"
HF_QWEN = "Qwen/Qwen2.5-0.5B"

app = FastAPI(title="ChatGPT Clone - Laboratório LLM")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODELS: dict = {}

# =============================================================================
# CARREGAMENTO NATIVO (SEM PIPELINE)
# =============================================================================

def carregar_seq2seq_finetuned(path_lora: str, repo_base_id: str) -> dict:
    logger.info(f"A carregar Seq2Seq: Local={path_lora} | Base={repo_base_id}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(repo_base_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            repo_base_id, 
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        
        if os.path.exists(path_lora):
            model = PeftModel.from_pretrained(base_model, path_lora)
        else:
            model = base_model
            
        model.eval()
        return {"model": model, "tokenizer": tokenizer}
    except Exception as e:
        logger.error(f"Erro no Seq2Seq: {e}")
        return {"model": None, "tokenizer": None}

def carregar_causal_finetuned(path_lora: str, repo_base_id: str) -> dict:
    logger.info(f"A carregar Causal: Local={path_lora} | Base={repo_base_id}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(repo_base_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            repo_base_id, 
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        
        if os.path.exists(path_lora):
            model = PeftModel.from_pretrained(base_model, path_lora)
        else:
            model = base_model
            
        model.eval()
        return {"model": model, "tokenizer": tokenizer}
    except Exception as e:
        logger.error(f"Erro no Causal: {e}")
        return {"model": None, "tokenizer": None}

@app.on_event("startup")
async def startup_event():
    global MODELS
    logger.info("INICIANDO SERVIDOR - A carregar os 4 Modelos de Linguagem...")
    MODELS["seq2seq-bart"] = carregar_seq2seq_finetuned(PATH_BART, HF_BART)
    MODELS["seq2seq-blenderbot"] = carregar_seq2seq_finetuned(PATH_BLENDERBOT, HF_BLENDERBOT)
    MODELS["causal-pythia"] = carregar_causal_finetuned(PATH_PYTHIA, HF_PYTHIA)
    MODELS["causal-qwen"] = carregar_causal_finetuned(PATH_QWEN, HF_QWEN)
    logger.info("Modelos carregados com sucesso!")

# =============================================================================
# SCHEMAS DA API
# =============================================================================
class ChatRequest(BaseModel):
    modelo: str
    mensagem: str
    max_tokens: Optional[int] = 150
    temperatura: Optional[float] = 0.7

class ChatResponse(BaseModel):
    resposta: str
    modelo: str
    tokens_gerados: int

# =============================================================================
# ENDPOINTS
# =============================================================================
@app.get("/modelos", response_class=JSONResponse)
async def listar_modelos():
    modelos_info = {
        "seq2seq-bart": {"id": "seq2seq-bart", "nome": "BART Base (Seq2Seq)", "descricao": "Modelo Seq2Seq fine-tunado"},
        "seq2seq-blenderbot": {"id": "seq2seq-blenderbot", "nome": "Blenderbot 400M (Seq2Seq)", "descricao": "Modelo Seq2Seq otimizado"},
        "causal-pythia": {"id": "causal-pythia", "nome": "Pythia 70M (Causal)", "descricao": "Modelo Causal leve"},
        "causal-qwen": {"id": "causal-qwen", "nome": "Qwen 2.5 0.5B (Causal)", "descricao": "Modelo Causal de última geração"},
    }
    disponiveis = [info for key, info in modelos_info.items() if key in MODELS and MODELS[key]["model"] is not None]
    return {"modelos": disponiveis}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    POST /chat
    Recebe a mensagem do usuário, encaminha para o modelo selecionado e retorna a resposta.
    """
    if request.modelo not in MODELS or MODELS[request.modelo]["model"] is None:
        raise HTTPException(
            status_code=404, 
            detail=f"Modelo '{request.modelo}' não está disponível ou falhou ao carregar."
        )
    
    try:
        model = MODELS[request.modelo]["model"]
        tokenizer = MODELS[request.modelo]["tokenizer"]
        device = model.device

        # Formatação idêntica ao padrão de instrução (Alpaca/LoRA template)
        prompt_formatado = (
            "Abaixo está uma instrução que descreve uma tarefa. "
            "Escreva uma resposta que complete adequadamente o pedido.\n\n"
            f"### Instrução:\n{request.mensagem}\n\n"
            "### Resposta:\n"
        )

        inputs = tokenizer(prompt_formatado, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=max(0.1, request.temperatura), # Evita temperaturas zeradas que quebram a amostragem
                do_sample=True if request.temperatura > 0.1 else False,
                top_k=40,
                top_p=0.85,
                repetition_penalty=1.4,  # Penalidade um pouco maior para esmagar repetições
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )

        texto_gerado = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # --- CORTE CIRÚRGICO DE TEXTO ---
        # Se o modelo gerou a tag de resposta, isolamos apenas o que vem depois dela
        if "### Resposta:" in texto_gerado:
            resposta = texto_gerado.split("### Resposta:")[-1].strip()
        # Se ele tentou reinventar o prompt e escreveu '### Instrução:' de novo, cortamos antes disso
        elif "### Instrução:" in texto_gerado:
            resposta = texto_gerado.split("### Instrução:")[0].replace(prompt_formatado, "").strip()
        else:
            resposta = texto_gerado.replace(prompt_formatado, "").strip()

        # Limpeza de qualquer sobra que repita a pergunta do usuário
        if resposta.startswith(request.mensagem):
            resposta = resposta[len(request.mensagem):].strip()

        # Se o modelo gerou instruções repetidas ou lixo colado, pegamos apenas a primeira linha útil
        if "\n###" in resposta:
            resposta = resposta.split("\n###")[0].strip()

        # Proteção final para respostas em branco ou ecos teimosos
        if not rstrip_garbage(resposta) or resposta.lower() == request.mensagem.lower():
            resposta = "[O modelo gerou fragmentos irrelevantes. Reduza a temperatura na barra lateral ou selecione o modelo Qwen.]"

        tokens_gerados = len(tokenizer.encode(resposta))
        logger.info(f"  ✓ Resposta filtrada: {tokens_gerados} tokens")

        return ChatResponse(
            resposta=resposta,
            modelo=request.modelo,
            tokens_gerados=tokens_gerados
        )
        
    except Exception as e:
        logger.error(f"Erro durante a geração de resposta: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno no modelo: {str(e)}")

def rstrip_garbage(text: str) -> str:
    # Função auxiliar para validar se a resposta não é apenas caracteres soltos
    clean = text.replace("[", "").replace("]", "").strip()
    return clean if len(clean) > 3 else ""
# =============================================================================
# FRONT-END E INICIALIZAÇÃO
# =============================================================================
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = os.path.join("static", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)