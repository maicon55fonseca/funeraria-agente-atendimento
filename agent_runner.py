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


TOOLS_ENTREGA_AO_CLIENTE = frozenset(
    {
        "enviar_mensagem_texto_ao_cliente",
        "enviar_link_boleto_parcela",
        "enviar_recibo_parcela",
    }
)


def _tool_resposta_foi_sucesso(payload: str) -> bool:
    """True se HTTP ok e body não indica falha explícita (ok:false)."""
    try:
        envio = json.loads(payload)
    except Exception:
        return False
    st = envio.get("http_status")
    if not isinstance(st, int) or st >= 400:
        return False
    body = envio.get("body")
    if not isinstance(body, dict):
        return True
    if body.get("ok") is False:
        return False
    return True


def _ordenar_tool_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Garante contexto antes de envio ao cliente; finalizar por último (se vier no lote)."""
    prioridade = {
        "buscar_contexto_cliente": 0,
        "enviar_mensagem_texto_ao_cliente": 10,
        "enviar_link_boleto_parcela": 11,
        "enviar_recibo_parcela": 12,
        "finalizar_conversa_painel": 90,
    }

    def _chave(c: dict[str, Any]) -> tuple[int, str]:
        nome = str(c.get("name", ""))
        return (prioridade.get(nome, 50), str(c.get("id") or ""))

    return sorted(calls, key=_chave)


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

        def tool_enviar_boleto(
            parcela_id: int | None = None,
            parcela_ids: list | None = None,
            **_: Any,
        ) -> str:
            body: dict[str, Any] = {"conversation_id": conversation_id}
            ids_list = parcela_ids if isinstance(parcela_ids, list) else None
            if ids_list:
                body["parcela_ids"] = [int(x) for x in ids_list]
            elif parcela_id is not None:
                body["parcela_id"] = int(parcela_id)
            else:
                return json.dumps(
                    {"erro": "Informe parcela_id (uma parcela) ou parcela_ids (várias)."},
                    ensure_ascii=False,
                )
            return _post_tool(
                http,
                "/agente-atendimento/tools/enviar-boleto",
                body,
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
                        "Carrega dados do cliente, parcelas, histórico recente de mensagens, regras de intervenção humana e o objeto "
                        "data.saudacao (horário America/Sao_Paulo, periodo_dia, conversa_ja_tem_historico, evitar_repetir_saudacao_ciclo_completo, nome_primeiro, "
                        "modelo_com_nome, modelo_sem_nome_primeiro_contato, exemplo_saudacao_continuidade, instrucao_saudacao). "
                        "Também instrucao_proxima_parcela_vencimento, instrucao_como_chamar_o_cliente, instrucao_mensagens_agrupadas_debounce (várias frases do cliente podem vir num texto só), "
                        "contratos_cancelados (planos cancelados: numero_contrato, plano_nome, data_cancelamento), "
                        "qtd_contratos_cancelados_no_cadastro, instrucao_contratos_cancelados, parcelas_em_aberto_lista (parcela_id por mensalidade), "
                        "instrucao_boleto_pix_por_parcela e contratos_ativos[].proxima_parcela_em_aberto para data/valor da próxima parcela. "
                        "Para cumprimentos (oi, olá, bom dia etc.), siga data.saudacao e instrucao_saudacao. "
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
                    "description": (
                        "Envia texto ao cliente no WhatsApp. "
                        "Tom: direto e natural. Na conversa ativa NÃO finalize com blocos prontos como "
                        "\"se precisar de mais informações\", \"se desejar contratar um novo plano\", "
                        "\"estou aqui para ajudar\", \"estou à disposição\" ou similares a cada mensagem — "
                        "use isso só quando o cliente agradecer, disser que não precisa de mais nada ou o assunto tiver findado de fato. "
                        "Se a mensagem for saudação ou retomada leve, alinhe ao objeto data.saudacao retornado por buscar_contexto_cliente."
                    ),
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
                        "Envia cobrança ao cliente em mensagens separadas no WhatsApp (cabeçalho/link; aviso + só código de barras; "
                        "aviso + só PIX) para facilitar copiar. Uma mensalidade = um boleto/PIX próprios — não invente PIX único para várias parcelas. "
                        "parcela_id ou parcela_ids (até 8). Não repita códigos em enviar_mensagem_texto_ao_cliente depois."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "parcela_id": {
                                "type": "integer",
                                "description": "ID de uma parcela em aberto (parcelas_em_aberto_lista[].parcela_id).",
                            },
                            "parcela_ids": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "description": "Várias parcelas em aberto — cada uma com boleto/PIX separado (máx. 8).",
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
                        "Marca a conversa como FINALIZADA só no painel (sem WhatsApp). "
                        "Use RARAMENTE e apenas com sinal CLARO de encerramento: cliente agradeceu e encerrou, disse que não precisa de mais nada, "
                        "ou o atendimento naturalmente findou DEPOIS de você já ter tratado tudo. "
                        "Nunca finalize após cada resposta nem só porque você respondeu uma dúvida. Não chame esta tool em conversa ativa com perguntas em aberto."
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
            "enviar_link_boleto_parcela": lambda parcela_id=None, parcela_ids=None, **kw: tool_enviar_boleto(
                parcela_id, parcela_ids, **kw
            ),
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
            "FLUXO OBRIGATÓRIO (uma interação = uma resposta ao cliente): "
            "(1) buscar_contexto_cliente quando precisar de dados do cadastro — no máximo uma vez; "
            "(2) enviar_mensagem_texto_ao_cliente UMA vez com a resposta completa; "
            "(3) PARAR — não chame mais ferramentas, não envie segunda mensagem, não finalize a conversa no painel. "
            "Proibido: buscar → responder → buscar de novo → responder de novo → finalizar. "
            "O cliente pode mandar várias mensagens seguidas (Oi, depois Boa tarde): o sistema agrupa antes de você ser chamado — "
            "trate como um único pedido e não mande duas saudações. "
            "Regra obrigatória: toda resposta ao cliente no WhatsApp deve ser enviada pela ferramenta "
            "enviar_mensagem_texto_ao_cliente. Não basta escrever texto na sua mensagem sem chamar essa ferramenta "
            "(o cliente não recebe o que você escreve fora da tool). "
            "Não chame finalizar_conversa_painel após responder ao cliente; só em encerramento explícito raro (na prática evite). "
            "Tom e continuidade: durante conversa ativa, responda de forma direta; mantenha o fluxo natural. "
            "Se faltar dado, pergunte de forma objetiva em vez de encerrar. "
            "PROIBIDO encerrar praticamente toda mensagem com blocos de oferta genérica como \"Se precisar de mais informações\", "
            "\"se desejar contratar um novo plano\", \"estou aqui para ajudar\", \"estou à disposição\", \"conte comigo/conosco\", "
            "\"qualquer coisa é só chamar\" ou \"se tiver alguma dúvida\" — não repita esse fechamento a cada resposta; "
            "na conversa ativa prefira ZERO dessas frases (resposta direta e pare). "
            "Use cordialidade de encerramento SOMENTE quando o cliente agradeceu, disse que não precisa de mais nada ou o assunto findou de fato. "
            "Não finalize mentalmente o atendimento após cada mensagem sua. "
            "Não chame finalizar_conversa_painel com frequência; só com encerramento explícito do cliente ou trato totalmente concluído e cliente satisfeito. "
            "Se a mensagem do cliente for só um marcador como [áudio], [imagem], [vídeo], [documento], [sticker] ou [modelo], "
            "significa que ele enviou esse tipo de mídia (muitas vezes sem legenda): reconheça isso, responda de forma útil "
            "via enviar_mensagem_texto_ao_cliente e, se precisar de detalhes, peça que escreva por texto ou use buscar_contexto_cliente. "
            "Para dados de contrato, parcelas ou histórico da conversa, use buscar_contexto_cliente (lá vêm mensagens_recentes, "
            "regras de pausa pós-atendente humano, status e o objeto data.saudacao). "
            "Saudação e continuidade: após buscar_contexto_cliente, leia body.data.saudacao (se o retorno vier em envelope HTTP, use body.data). "
            "Se conversa_ja_tem_historico for true OU evitar_repetir_saudacao_ciclo_completo for true: NÃO comece com \"Oi\", \"Olá\", \"Oi, Nome\", nem \"Bom dia/Boa tarde/Boa noite\" como abertura — isso reinicia o atendimento; responda direto ao que o cliente disse. "
            "modelo_com_nome e modelo_sem_nome_primeiro_contato são só para conversa_ja_tem_historico false (primeiro contato no chat). "
            "Se conversa_ja_tem_historico for true, o atendimento já começou — sem tom de primeira conversa. "
            "Nunca use 'Seja bem-vindo(a)', 'bem-vindo à empresa' nem reinicie como primeiro contato. "
            "Com nome_primeiro (e nome_declarado_pelo_cliente_nas_mensagens em saudacao) use quem está FALANDO no chat; quando financeiro_mesmo_cadastro_que_vinculo_conversa for false, "
            "leia instrucao_como_chamar_o_cliente e não trate o titular do contrato (cliente_nome) como o nome do interlocutor. "
            "Com historico mas sem nome em saudacao: mesmo assim não abra com Oi/Bom dia/Boa noite — vá direto ao ponto. "
            "Só com conversa_ja_tem_historico false use modelo_sem_nome_primeiro_contato ou modelo_com_nome para abrir o primeiro atendimento neste chat. "
            "Horário e manhã/tarde/noite vêm de saudacao (timezone e hora_local); não contradiga o periodo_dia. "
            "Intervenção humana: se um atendente enviou mensagem pelo painel, a IA fica pausada por um período indicado em "
            "intervencao_humana_minutos_inatividade; cada nova mensagem humana recomeça esse prazo. "
            "Mensagem do cliente não encerra essa pausa — você só é chamado de novo quando já passou o silêncio humano exigido. "
            "Ao retomar, leia mensagens_recentes e não desfaça o que o humano acordou com o cliente. "
            "Se o cliente pedir boleto, linha digitável, código de barras ou PIX: use enviar_link_boleto_parcela com parcela_id ou parcela_ids "
            "(cada mensalidade tem cobrança própria — veja parcelas_em_aberto_lista e instrucao_boleto_pix_por_parcela). "
            "Não envie texto depois confirmando o boleto/PIX — a ferramenta já manda os dados. Não invente um PIX para o total de várias parcelas. "
            "Para valor da mensalidade, quanto paga ou preço do plano, siga instrucao_valor_mensalidade_cliente e o objeto contratos_ativos em buscar_contexto_cliente. "
            "Para próxima parcela a vencer (data e valor), siga estritamente instrucao_proxima_parcela_vencimento e proxima_parcela_em_aberto; não infira só com dia_vencimento do contrato. "
            "Para dizer se há parcelas em aberto, atraso ou se o cliente 'está em dia', siga instrucao_parcelas_aberto_atraso e os números em financeiro_resumo (parcelas_em_aberto, parcelas_em_atraso); não contradiga esses contadores. "
            "Se data.instrucao_cadastro_sem_contrato_ativo_listado vier preenchida ou cadastro_financeiro_sem_contrato_ativo_listado for true, siga essa instrução: "
            "se contratos_cancelados tiver itens ou qtd_contratos_cancelados_no_cadastro > 0 ou instrucao_contratos_cancelados vier preenchida, informe que o plano consta cancelado no cadastro e cite numero_contrato/plano_nome/data_cancelamento; "
            "mencione contrato suspenso se qtd_contratos_suspenso_no_cadastro > 0; não diga que 'não há contrato' de forma absoluta quando houver cancelados ou suspensos listados. "
            "Se o cliente pedir para falar com o financeiro, confirme de forma breve e evite repetir o mesmo bloco inteiro sobre 'sem contrato ativo' das mensagens anteriores. "
            "Se o cliente perguntar de onde vieram plano, valores ou datas (ex.: 'de onde você pegou', 'confirma aí'), siga instrucao_proveniencia_dados: cite contratos_ativos[].id, numero_contrato e plano_nome do JSON; "
            "não responda com frase genérica que não explica a origem. Se instrucao_multiplos_contratos_ativos vier preenchida, há mais de um contrato ativo — não misture dados entre eles. "
            "Não invente valores ou links; use apenas o retorno das ferramentas. "
            "Após chamar enviar_mensagem_texto_ao_cliente com sucesso nesta rodada, não produza mensagem adicional ao usuário: "
            "não gere novo raciocínio nem nova bolha — a ferramenta já enviou a resposta. "
            "Em cada passagem de ferramentas: no máximo UMA chamada a enviar_mensagem_texto_ao_cliente (uma bolha por vez); "
            "não envie duas saudações ou duas perguntas seguidas na mesma rodada.\n\n"
            f"Instruções adicionais da empresa:\n{extra_system_instructions or '(nenhuma)'}"
        )

        messages: list[Any] = [
            SystemMessage(content=system),
            HumanMessage(content=user_message),
        ]

        max_turns = 4
        last_text = ""
        entregou_algo_ao_cliente_whatsapp = False
        contexto_obtido_no_turno: int | None = None

        for turn in range(max_turns):
            if entregou_algo_ao_cliente_whatsapp:
                logger.info(
                    "[agente] parada_entrega_ja_realizada correlation_id=%s conversation_id=%s turn=%s",
                    cid,
                    conversation_id,
                    turn,
                )
                break

            if (
                contexto_obtido_no_turno is not None
                and turn > contexto_obtido_no_turno + 1
            ):
                logger.info(
                    "[agente] parada_max_uma_rodada_apos_contexto correlation_id=%s conversation_id=%s "
                    "contexto_turn=%s turn_atual=%s",
                    cid,
                    conversation_id,
                    contexto_obtido_no_turno,
                    turn,
                )
                break

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

            texto_whatsapp_ja_enviado_neste_batch = False
            cobranca_enviada_neste_batch = False
            for call in _ordenar_tool_calls(list(calls)):
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

                if name == "finalizar_conversa_painel" and (
                    entregou_algo_ao_cliente_whatsapp or texto_whatsapp_ja_enviado_neste_batch
                ):
                    logger.info(
                        "[agente] finalizar_conversa_ignorado_apos_resposta correlation_id=%s turn=%s conversation_id=%s",
                        cid,
                        turn,
                        conversation_id,
                    )
                    payload = json.dumps(
                        {
                            "ok": True,
                            "skipped": True,
                            "message": (
                                "Finalização no painel não permitida na mesma interação em que você já "
                                "respondeu ao cliente. A conversa permanece aberta."
                            ),
                        },
                        ensure_ascii=False,
                    )
                    messages.append(ToolMessage(content=payload, tool_call_id=tid))
                    continue

                if name in TOOLS_ENTREGA_AO_CLIENTE and entregou_algo_ao_cliente_whatsapp:
                    logger.warning(
                        "[agente] entrega_duplicada_ignorada correlation_id=%s turn=%s tool=%s conversation_id=%s",
                        cid,
                        turn,
                        name,
                        conversation_id,
                    )
                    payload = json.dumps(
                        {
                            "ok": True,
                            "skipped_duplicate": True,
                            "message": "Já houve entrega ao cliente nesta interação; não envie outra mensagem.",
                        },
                        ensure_ascii=False,
                    )
                    messages.append(ToolMessage(content=payload, tool_call_id=tid))
                    continue

                if name == "enviar_mensagem_texto_ao_cliente" and cobranca_enviada_neste_batch:
                    logger.info(
                        "[agente] enviar_texto_ignorado_apos_cobranca correlation_id=%s turn=%s conversation_id=%s",
                        cid,
                        turn,
                        conversation_id,
                    )
                    payload = json.dumps(
                        {
                            "ok": True,
                            "skipped": True,
                            "message": (
                                "Dados de boleto/PIX já foram enviados nesta interação. "
                                "Não envie mensagem de texto repetindo ou confirmando o envio."
                            ),
                        },
                        ensure_ascii=False,
                    )
                    messages.append(ToolMessage(content=payload, tool_call_id=tid))
                    continue

                if name == "enviar_mensagem_texto_ao_cliente" and texto_whatsapp_ja_enviado_neste_batch:
                    logger.warning(
                        "[agente] enviar_texto_duplicado_na_mesma_rodada correlation_id=%s turn=%s conversation_id=%s tool_call_id=%s",
                        cid,
                        turn,
                        conversation_id,
                        tid,
                    )
                    payload = json.dumps(
                        {
                            "ok": True,
                            "skipped_duplicate": True,
                            "message": "Mensagem ao cliente já enviada nesta rodada; não chame esta ferramenta duas vezes.",
                        },
                        ensure_ascii=False,
                    )
                    messages.append(ToolMessage(content=payload, tool_call_id=tid))
                    continue
                fn = tool_dispatch.get(name)
                if fn is None:
                    payload = json.dumps({"erro": f"ferramenta_desconhecida:{name}"}, ensure_ascii=False)
                else:
                    try:
                        payload = fn(**args)
                        if name == "buscar_contexto_cliente" and _tool_resposta_foi_sucesso(payload):
                            contexto_obtido_no_turno = turn
                        if name in TOOLS_ENTREGA_AO_CLIENTE and _tool_resposta_foi_sucesso(payload):
                            entregou_algo_ao_cliente_whatsapp = True
                            if name == "enviar_link_boleto_parcela":
                                cobranca_enviada_neste_batch = True
                            if name == "enviar_mensagem_texto_ao_cliente":
                                texto_whatsapp_ja_enviado_neste_batch = True
                                last_text = str(args.get("texto") or "")
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "[agente] llm_tool_exec_erro correlation_id=%s tool=%s erro=%s",
                            cid,
                            name,
                            exc,
                        )
                        payload = json.dumps({"erro": str(exc)}, ensure_ascii=False)
                messages.append(ToolMessage(content=payload, tool_call_id=tid))

            if entregou_algo_ao_cliente_whatsapp:
                logger.info(
                    "[agente] parada_apos_entrega_cliente correlation_id=%s turn=%s conversation_id=%s "
                    "(fluxo: contexto → resposta → parar; sem nova invocação do modelo)",
                    cid,
                    turn,
                    conversation_id,
                )
                break

        if not entregou_algo_ao_cliente_whatsapp:
            msg = (last_text or "").strip()
            if not msg:
                msg = (
                    "Recebi sua mensagem. O que você precisa — contrato, parcela, boleto ou outra dúvida? "
                    "Se puder, responda em uma frase."
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
