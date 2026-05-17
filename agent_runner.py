"""Orquestra o LLM (OpenAI via LangChain) com ferramentas HTTP para a API Laravel."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Callable

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


def _safe_log_text(s: str, max_len: int = 4000) -> str:
    """Encurta e normaliza texto para log (sem vazar conteúdo enorme)."""
    t = (s or "").replace("\r\n", "\n").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 3] + "..."


def _log_tool_args_para_diagnostico(cid: str, turn: int, name: str, args: dict[str, Any]) -> None:
    """Diagnóstico: o que o modelo pediu às tools (especialmente texto ao cliente)."""
    if name == "enviar_mensagem_texto_ao_cliente":
        preview = _safe_log_text(str(args.get("texto", "")), 3500)
        logger.info(
            "[agente] openai_tool_args_enviar_texto correlation_id=%s turn=%s chars=%s texto=%s",
            cid,
            turn,
            len(str(args.get("texto", "") or "")),
            preview,
        )
    elif name == "buscar_contexto_cliente":
        logger.info(
            "[agente] openai_tool_args_contexto correlation_id=%s turn=%s motivo=%s",
            cid,
            turn,
            _safe_log_text(str(args.get("motivo", "")), 800),
        )
    else:
        logger.info(
            "[agente] openai_tool_args correlation_id=%s turn=%s tool=%s args=%s",
            cid,
            turn,
            name,
            _safe_log_text(json.dumps(args, ensure_ascii=False), 2000),
        )


def _post_tool(
    http: httpx.Client,
    path: str,
    body: dict[str, Any],
    *,
    correlation_id: str = "-",
) -> str:
    cid = correlation_id or "-"
    t_req = time.perf_counter()
    logger.info("[agente] tool_laravel_inicio correlation_id=%s path=%s", cid, path)
    try:
        r = http.post(path, json=body)
    except httpx.ConnectError as e:
        logger.error(
            "[agente] tool_laravel_connect_error correlation_id=%s path=%s erro=%s",
            cid,
            path,
            e,
        )
        return json.dumps({"erro": f"connect:{e!s}", "http_status": None}, ensure_ascii=False)
    except httpx.TimeoutException as e:
        logger.error(
            "[agente] tool_laravel_timeout correlation_id=%s path=%s erro=%s",
            cid,
            path,
            e,
        )
        return json.dumps({"erro": f"timeout:{e!s}", "http_status": None}, ensure_ascii=False)
    except Exception as e:
        logger.exception("[agente] tool_laravel_erro correlation_id=%s path=%s", cid, path)
        return json.dumps({"erro": str(e), "tipo": type(e).__name__}, ensure_ascii=False)

    elapsed_ms = round((time.perf_counter() - t_req) * 1000, 2)
    try:
        data = r.json()
    except Exception:
        data = {"raw": r.text[:500]}
    if r.status_code >= 400:
        logger.warning(
            "[agente] tool_laravel_resposta_erro correlation_id=%s path=%s http_status=%s ms=%s body_keys=%s",
            cid,
            path,
            r.status_code,
            elapsed_ms,
            list(data.keys())[:25] if isinstance(data, dict) else None,
        )
    else:
        body_keys = list(data.keys())[:25] if isinstance(data, dict) else None
        logger.info(
            "[agente] tool_laravel_resposta_ok correlation_id=%s path=%s http_status=%s ms=%s body_keys=%s",
            cid,
            path,
            r.status_code,
            elapsed_ms,
            body_keys,
        )
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
    correlation_id: str | None = None,
) -> dict[str, Any]:
    cid = (correlation_id or "").strip() or "-"
    base = laravel_api_base.rstrip("/")
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    model_name = (openai_model or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini").strip()
    base_url = (openai_base_url or os.environ.get("OPENAI_BASE_URL") or "").strip() or None

    logger.info(
        "[agente] run_agent_turn_inicio correlation_id=%s conversation_id=%s "
        "laravel_api_base=%s model=%s",
        cid,
        conversation_id,
        base[:180],
        model_name,
    )

    with httpx.Client(base_url=base, headers=headers, timeout=120.0) as http:

        def tool_contexto() -> str:
            return _post_tool(
                http,
                "/agente-atendimento/tools/contexto",
                {"conversation_id": conversation_id},
                correlation_id=cid,
            )

        def tool_enviar_texto(texto: str) -> str:
            return _post_tool(
                http,
                "/agente-atendimento/tools/enviar-texto",
                {"conversation_id": conversation_id, "texto": texto},
                correlation_id=cid,
            )

        def tool_enviar_boleto(parcela_id: int) -> str:
            return _post_tool(
                http,
                "/agente-atendimento/tools/enviar-boleto",
                {"conversation_id": conversation_id, "parcela_id": int(parcela_id)},
                correlation_id=cid,
            )

        def tool_enviar_recibo(parcela_id: int | None = None) -> str:
            body: dict[str, Any] = {"conversation_id": conversation_id}
            if parcela_id is not None:
                body["parcela_id"] = int(parcela_id)
            return _post_tool(
                http,
                "/agente-atendimento/tools/enviar-recibo",
                body,
                correlation_id=cid,
            )

        def tool_finalizar_conversa(motivo: str | None = None) -> str:
            body: dict[str, Any] = {"conversation_id": conversation_id}
            if motivo is not None and str(motivo).strip() != "":
                body["motivo"] = str(motivo).strip()[:500]
            return _post_tool(
                http,
                "/agente-atendimento/tools/finalizar-conversa",
                body,
                correlation_id=cid,
            )

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "buscar_contexto_cliente",
                    "description": (
                        "Carrega dados do cliente, parcelas, histórico recente de mensagens e regras de intervenção humana. "
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
                    "description": (
                        "Envia ao cliente os dados de pagamento da parcela: link do boleto (se houver), "
                        "linha digitável, código de barras e PIX copia e cola quando disponíveis na integração (Progem v2 / Asaas). "
                        "Use após buscar_contexto_cliente e quando o cliente pedir boleto, linha, código de barras ou PIX."
                    ),
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
            {
                "type": "function",
                "function": {
                    "name": "finalizar_conversa_painel",
                    "description": (
                        "Marca a conversa como FINALIZADA no painel interno (status), sem enviar WhatsApp. "
                        "Use só depois de cumprimentar/encerrar com o cliente via enviar_mensagem_texto_ao_cliente "
                        "(ex.: perguntou se precisa de mais algo e o atendimento esfriou) e se o contexto indica que o caso foi resolvido."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "motivo": {
                                "type": "string",
                                "description": "Opcional. Resumo interno do motivo do encerramento.",
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
            "finalizar_conversa_painel": lambda motivo=None, **_: tool_finalizar_conversa(
                str(motivo).strip() if motivo is not None and str(motivo).strip() != "" else None
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
            "Regra obrigatória: toda resposta ao cliente no WhatsApp deve ser enviada pela ferramenta "
            "enviar_mensagem_texto_ao_cliente. Não basta escrever texto na sua mensagem sem chamar essa ferramenta "
            "(o cliente não recebe o que você escreve fora da tool). "
            "Se a mensagem do cliente for só um marcador como [áudio], [imagem], [vídeo], [documento], [sticker] ou [modelo], "
            "significa que ele enviou esse tipo de mídia (muitas vezes sem legenda): reconheça isso, responda de forma útil "
            "via enviar_mensagem_texto_ao_cliente e, se precisar de detalhes, peça que escreva por texto ou use buscar_contexto_cliente. "
            "Para dados de contrato, parcelas ou histórico da conversa, use buscar_contexto_cliente (lá vêm mensagens_recentes, "
            "regras de pausa pós-atendente humano e status). "
            "Intervenção humana: se um atendente enviou mensagem pelo painel, a IA fica pausada por um período indicado em "
            "intervencao_humana_minutos_inatividade; cada nova mensagem humana recomeça esse prazo. "
            "Mensagem do cliente não encerra essa pausa — você só é chamado de novo quando já passou o silêncio humano exigido. "
            "Ao retomar, leia mensagens_recentes e não desfaça o que o humano acordou com o cliente. "
            "Se o cliente pedir boleto, linha digitável, código de barras ou PIX, use enviar_link_boleto_parcela (a ferramenta envia tudo o que estiver disponível). "
            "Encerramento: quando o pedido estiver resolvido, pergunte de forma breve se precisa de mais algo; se o cliente indicar que não ou agradecer e encerrar, "
            "pode usar finalizar_conversa_painel após sua última mensagem ao cliente (essa tool só muda o status no painel). "
            "Não invente valores ou links; use apenas o retorno das ferramentas.\n\n"
            f"Instruções adicionais da empresa:\n{extra_system_instructions or '(nenhuma)'}"
        )

        messages: list[Any] = [
            SystemMessage(content=system),
            HumanMessage(content=user_message),
        ]

        max_turns = 10
        last_text = ""
        entregou_algo_ao_cliente_whatsapp = False

        for turn in range(max_turns):
            ai: AIMessage = llm_tools.invoke(messages)
            messages.append(ai)
            calls = getattr(ai, "tool_calls", None) or []
            raw_content = str(ai.content or "").strip()
            logger.info(
                "[agente] openai_turn correlation_id=%s turn=%s conversation_id=%s "
                "tem_tool_calls=%s content_chars=%s content=%s",
                cid,
                turn,
                conversation_id,
                bool(calls),
                len(raw_content),
                _safe_log_text(raw_content, 3500) if raw_content else "(vazio)",
            )
            if not calls:
                last_text = str(ai.content or "")
                logger.info(
                    "[agente] openai_resposta_sem_tool correlation_id=%s turn=%s output=%s",
                    cid,
                    turn,
                    _safe_log_text(last_text, 4000),
                )
                break

            for call in calls:
                name = str(call.get("name", ""))
                args = dict(call.get("args") or {})
                tid = str(call.get("id") or f"call_{turn}_{name}")
                logger.info(
                    "[agente] llm_tool_call correlation_id=%s turn=%s tool=%s conversation_id=%s",
                    cid,
                    turn,
                    name,
                    conversation_id,
                )
                _log_tool_args_para_diagnostico(cid, turn, name, args)
                fn = tool_dispatch.get(name)
                if fn is None:
                    payload = json.dumps({"erro": f"ferramenta_desconhecida:{name}"}, ensure_ascii=False)
                else:
                    try:
                        if name in (
                            "enviar_mensagem_texto_ao_cliente",
                            "enviar_link_boleto_parcela",
                            "enviar_recibo_parcela",
                        ):
                            entregou_algo_ao_cliente_whatsapp = True
                        payload = fn(**args)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "[agente] llm_tool_exec_erro correlation_id=%s tool=%s erro=%s",
                            cid,
                            name,
                            exc,
                        )
                        payload = json.dumps({"erro": str(exc)}, ensure_ascii=False)
                messages.append(ToolMessage(content=payload, tool_call_id=tid))

        if not entregou_algo_ao_cliente_whatsapp:
            msg = (last_text or "").strip()
            if not msg:
                msg = (
                    "Olá! Recebemos sua mensagem. Como podemos ajudar você hoje? "
                    "Se for sobre contrato, parcelas ou pagamentos, pode detalhar por aqui."
                )
            logger.info(
                "[agente] envio_whatsapp_fallback correlation_id=%s conversation_id=%s chars=%s",
                cid,
                conversation_id,
                len(msg),
            )
            _post_tool(
                http,
                "/agente-atendimento/tools/enviar-texto",
                {"conversation_id": conversation_id, "texto": msg},
                correlation_id=cid,
            )
            last_text = msg

        logger.info(
            "[agente] openai_output_final correlation_id=%s conversation_id=%s chars=%s texto=%s",
            cid,
            conversation_id,
            len(last_text or ""),
            _safe_log_text(last_text or "", 8000),
        )
        logger.info(
            "[agente] run_agent_turn_fim correlation_id=%s turnos_llm=%s conversation_id=%s",
            cid,
            turn + 1,
            conversation_id,
        )

        return {"ok": True, "output": last_text}
