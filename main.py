import logging
import os
import time
from urllib.parse import urlparse

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agente_atendimento")

app = FastAPI(title="Agente Atendimento", version="1.0.0")

EXPECTED_TOKEN = (os.environ.get("LARAVEL_AGENT_TOKEN") or "").strip().strip("'\"").strip()


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


def _laravel_base_resumo(url: str) -> str:
    """Host da API Laravel para log (sem credenciais)."""
    try:
        p = urlparse(url)
        if p.scheme and p.netloc:
            return f"{p.scheme}://{p.netloc}{p.path[:48]}"
    except Exception:
        pass
    return (url or "")[:120]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/processar")
def processar(
    request: Request,
    body: ProcessarBody,
    authorization: str | None = Header(default=None, alias="Authorization"),
    x_correlation_id: str | None = Header(default=None, alias="X-Correlation-Id"),
) -> dict[str, object]:
    cid = (x_correlation_id or "").strip() or "-"
    t0 = time.perf_counter()
    client_host = request.client.host if request.client else "?"

    logger.info(
        "[agente] processar_requisicao_recebida correlation_id=%s client=%s path=%s",
        cid,
        client_host,
        request.url.path,
    )
    logger.info(
        "[agente] processar_http_meta correlation_id=%s method=%s user_agent=%s content_length=%s",
        cid,
        request.method,
        (request.headers.get("user-agent") or "")[:160],
        request.headers.get("content-length"),
    )

    try:
        _checar_token(authorization)
    except HTTPException as e:
        logger.warning(
            "[agente] processar_token_recusado correlation_id=%s status=%s detail=%s",
            cid,
            e.status_code,
            e.detail,
        )
        raise

    logger.info("[agente] processar_token_ok correlation_id=%s", cid)

    from agent_runner import run_agent_turn

    api_key_body = (body.openai_api_key or "").strip().strip("'\"").strip()
    api_key_env = (os.environ.get("OPENAI_API_KEY") or "").strip().strip("'\"").strip()
    api_key = api_key_body or api_key_env
    if not api_key:
        logger.error("[agente] processar_sem_openai_key correlation_id=%s", cid)
        raise HTTPException(
            status_code=503,
            detail="Chave OpenAI não configurada: defina no painel (empresa) ou OPENAI_API_KEY no ambiente do serviço Python.",
        )

    openai_origem = "body" if api_key_body else "env"
    instr_len = len((body.instrucoes_atendimento or "").strip())
    texto_len = len((body.mensagem_texto or "").strip())

    logger.info(
        "[agente] processar_corpo_ok correlation_id=%s conversation_id=%s whatsapp_message_id=%s "
        "empresa_id=%s corporacao_id=%s laravel_api_base=%s openai_origem=%s instrucoes_chars=%s mensagem_chars=%s",
        cid,
        body.conversation_id,
        body.whatsapp_message_id,
        body.empresa_id,
        body.corporacao_id,
        _laravel_base_resumo(body.laravel_api_base),
        openai_origem,
        instr_len,
        texto_len,
    )

    try:
        out = run_agent_turn(
            laravel_api_base=body.laravel_api_base,
            token=EXPECTED_TOKEN,
            conversation_id=body.conversation_id,
            user_message=body.mensagem_texto,
            extra_system_instructions=(body.instrucoes_atendimento or "").strip(),
            openai_api_key=api_key,
            openai_model=(body.openai_model or "").strip() or None,
            openai_base_url=(body.openai_base_url or "").strip() or None,
            correlation_id=cid,
        )
    except Exception:
        logger.exception("[agente] processar_run_agent_falhou correlation_id=%s", cid)
        raise

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    out_output = ""
    if isinstance(out, dict):
        out_output = str(out.get("output") or "")
    preview = (out_output[:6000] + "…") if len(out_output) > 6000 else out_output
    logger.info(
        "[agente] processar_concluido_ok correlation_id=%s conversation_id=%s duracao_ms=%s "
        "openai_output_chars=%s openai_output_texto=%s",
        cid,
        body.conversation_id,
        elapsed_ms,
        len(out_output),
        preview or "(vazio)",
    )
    return {"ok": True, "agent": out}
