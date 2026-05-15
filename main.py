import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agente_atendimento")

app = FastAPI(title="Agente Atendimento", version="1.0.0")

EXPECTED_TOKEN = (os.environ.get("LARAVEL_AGENT_TOKEN") or "").strip()


class ProcessarBody(BaseModel):
    laravel_api_base: str = Field(..., min_length=8, description="Ex.: https://seu-backend/api")
    conversation_id: int = Field(..., ge=1)
    whatsapp_message_id: int = Field(..., ge=1)
    mensagem_texto: str = Field(..., min_length=1, max_length=8000)
    empresa_id: int = Field(..., ge=1)
    corporacao_id: int | None = None
    openai_api_key: str | None = None
    openai_model: str | None = None
    openai_base_url: str | None = None
    instrucoes_atendimento: str | None = Field(default=None, max_length=50000)


def _checar_token(authorization: str | None) -> None:
    if not EXPECTED_TOKEN:
        raise HTTPException(status_code=503, detail="Serviço sem LARAVEL_AGENT_TOKEN configurado.")
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Authorization Bearer obrigatório.")
    sent = authorization.split(" ", 1)[1].strip()
    if sent != EXPECTED_TOKEN:
        raise HTTPException(status_code=401, detail="Token inválido.")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/processar")
def processar(
    body: ProcessarBody,
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> dict[str, object]:
    _checar_token(authorization)

    from agent_runner import run_agent_turn

    api_key_body = (body.openai_api_key or "").strip()
    api_key_env = (os.environ.get("OPENAI_API_KEY") or "").strip()
    api_key = api_key_body or api_key_env
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Chave OpenAI não configurada: defina no painel (empresa) ou OPENAI_API_KEY no ambiente do serviço Python.",
        )

    logger.info(
        "processar_recebido conversation_id=%s whatsapp_message_id=%s empresa_id=%s",
        body.conversation_id,
        body.whatsapp_message_id,
        body.empresa_id,
    )

    out = run_agent_turn(
        laravel_api_base=body.laravel_api_base,
        token=EXPECTED_TOKEN,
        conversation_id=body.conversation_id,
        user_message=body.mensagem_texto,
        extra_system_instructions=(body.instrucoes_atendimento or "").strip(),
        openai_api_key=api_key,
        openai_model=(body.openai_model or "").strip() or None,
        openai_base_url=(body.openai_base_url or "").strip() or None,
    )
    return {"ok": True, "agent": out}
