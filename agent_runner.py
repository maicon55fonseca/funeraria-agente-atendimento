"""Orquestra o LLM (OpenAI via LangChain) com ferramentas HTTP para a API Laravel."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


def _post_tool(http: httpx.Client, path: str, body: dict[str, Any]) -> str:
    r = http.post(path, json=body)
    try:
        data = r.json()
    except Exception:
        data = {"raw": r.text[:500]}
    out = {"http_status": r.status_code, "body": data}
    return json.dumps(out, ensure_ascii=False)


def run_agent_turn(
    *,
    laravel_api_base: str,
    token: str,
    conversation_id: int,
    user_message: str,
    extra_system_instructions: str,
    openai_api_key: str,
    openai_model: str | None = None,
    openai_base_url: str | None = None,
) -> dict[str, Any]:
    base = laravel_api_base.rstrip("/")
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    model_name = (openai_model or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini").strip()
    base_url = (openai_base_url or os.environ.get("OPENAI_BASE_URL") or "").strip() or None

    with httpx.Client(base_url=base, headers=headers, timeout=120.0) as http:

        def tool_contexto() -> str:
            return _post_tool(
                http,
                "/agente-atendimento/tools/contexto",
                {"conversation_id": conversation_id},
            )

        def tool_enviar_texto(texto: str) -> str:
            return _post_tool(
                http,
                "/agente-atendimento/tools/enviar-texto",
                {"conversation_id": conversation_id, "texto": texto},
            )

        def tool_enviar_boleto(parcela_id: int) -> str:
            return _post_tool(
                http,
                "/agente-atendimento/tools/enviar-boleto",
                {"conversation_id": conversation_id, "parcela_id": int(parcela_id)},
            )

        def tool_enviar_recibo(parcela_id: int | None = None) -> str:
            body: dict[str, Any] = {"conversation_id": conversation_id}
            if parcela_id is not None:
                body["parcela_id"] = int(parcela_id)
            return _post_tool(http, "/agente-atendimento/tools/enviar-recibo", body)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "buscar_contexto_cliente",
                    "description": (
                        "Carrega dados do cliente, parcelas e instruções configuradas no painel. "
                        "Sempre chame primeiro em novas interações."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "motivo": {
                                "type": "string",
                                "description": "Opcional. Por que você está consultando o contexto.",
                            },
                        },
                        "required": [],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "enviar_mensagem_texto_ao_cliente",
                    "description": "Envia uma mensagem de texto ao cliente no WhatsApp.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "texto": {"type": "string", "description": "Texto a enviar ao cliente."},
                        },
                        "required": ["texto"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "enviar_link_boleto_parcela",
                    "description": "Envia o link de boleto para pagamento da parcela.",
                    "parameters": {
                        "type": "object",
                        "properties": {"parcela_id": {"type": "integer"}},
                        "required": ["parcela_id"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "enviar_recibo_parcela",
                    "description": "Envia recibo/comprovante. Opcional parcela_id (parcela já paga).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "parcela_id": {
                                "type": "integer",
                                "description": "Opcional. ID da parcela já paga para filtrar o recibo.",
                            },
                        },
                        "required": [],
                        "additionalProperties": False,
                    },
                },
            },
        ]

        tool_dispatch: dict[str, Callable[..., str]] = {
            "buscar_contexto_cliente": lambda **_: tool_contexto(),
            "enviar_mensagem_texto_ao_cliente": lambda texto, **_: tool_enviar_texto(str(texto)),
            "enviar_link_boleto_parcela": lambda parcela_id, **_: tool_enviar_boleto(int(parcela_id)),
            "enviar_recibo_parcela": lambda parcela_id=None, **_: tool_enviar_recibo(
                int(parcela_id) if parcela_id is not None else None
            ),
        }

        llm_kwargs: dict[str, Any] = {"model": model_name, "temperature": 0.2, "api_key": openai_api_key}
        if base_url:
            llm_kwargs["base_url"] = base_url
        llm = ChatOpenAI(**llm_kwargs)
        llm_tools = llm.bind_tools(tools)

        system = (
            "Você é o assistente virtual de atendimento no WhatsApp. "
            "Seja cordial, objetivo e em português do Brasil. "
            "Use a ferramenta buscar_contexto_cliente para obter dados antes de responder sobre contrato ou parcelas. "
            "Se o cliente pedir boleto ou link para pagamento, use enviar_link_boleto_parcela com o id correto da parcela "
            "(priorize vencida ou a mais próxima do vencimento entre as em aberto). "
            "Se pedir recibo ou comprovante, use enviar_recibo_parcela. "
            "Não invente valores ou links; use apenas o retorno das ferramentas.\n\n"
            f"Instruções adicionais da empresa:\n{extra_system_instructions or '(nenhuma)'}"
        )

        messages: list[Any] = [
            SystemMessage(content=system),
            HumanMessage(content=user_message),
        ]

        max_turns = 10
        last_text = ""

        for turn in range(max_turns):
            ai: AIMessage = llm_tools.invoke(messages)
            messages.append(ai)
            calls = getattr(ai, "tool_calls", None) or []
            if not calls:
                last_text = str(ai.content or "")
                break

            for call in calls:
                name = str(call.get("name", ""))
                args = dict(call.get("args") or {})
                tid = str(call.get("id") or f"call_{turn}_{name}")
                fn = tool_dispatch.get(name)
                if fn is None:
                    payload = json.dumps({"erro": f"ferramenta_desconhecida:{name}"}, ensure_ascii=False)
                else:
                    try:
                        payload = fn(**args)
                    except Exception as exc:  # noqa: BLE001
                        payload = json.dumps({"erro": str(exc)}, ensure_ascii=False)
                messages.append(ToolMessage(content=payload, tool_call_id=tid))

        logger.info("agente_turnos=%s conversation_id=%s", turn + 1, conversation_id)

        return {"ok": True, "output": last_text}
