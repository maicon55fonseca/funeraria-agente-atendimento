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
        "cadastrar_cliente_pelo_documento": 3,
        "avisar_equipe_escalonamento": 8,
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
    deve_usar_instrucoes_audio: bool = False,
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

        def tool_avisar_equipe_escalonamento(motivo: str, resumo: str | None = None) -> str:
            body: dict[str, Any] = {
                "conversation_id": conversation_id,
                "motivo": str(motivo).strip()[:2000],
            }
            if resumo is not None and str(resumo).strip() != "":
                body["resumo"] = str(resumo).strip()[:4000]
            return _post_tool(
                http,
                "/agente-atendimento/tools/avisar-equipe-escalonamento",
                body,
                correlation_id=cid,
            )

        def tool_cadastrar_cliente_documento() -> str:
            return _post_tool(
                http,
                "/agente-atendimento/tools/cadastrar-cliente-documento",
                {"conversation_id": conversation_id},
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
                        "Campos deve_usar_instrucoes_audio_agora, instrucoes_atendimento_audio e conversa_em_modo_audio: "
                        "quando deve_usar_instrucoes_audio_agora for true, siga instrucoes_atendimento_audio para TODO o conteúdo "
                        "(dependentes, filhos, casamento, inclusões no plano, tom falado) — prioridade sobre instrucoes_atendimento_geral. "
                        "Também instrucao_proxima_parcela_vencimento, instrucao_como_chamar_o_cliente, instrucao_mensagens_agrupadas_debounce (várias frases do cliente podem vir num texto só), "
                        "contratos_cancelados (planos cancelados: numero_contrato, plano_nome, data_cancelamento), "
                        "qtd_contratos_cancelados_no_cadastro, instrucao_contratos_cancelados, parcelas_em_aberto_lista (parcela_id por mensalidade), "
                        "instrucao_boleto_pix_por_parcela, parcelas_em_aberto_lista (parcela_id, mes_referencia_vencimento) e contratos_ativos[].proxima_parcela_em_aberto. "
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
                    "name": "avisar_equipe_escalonamento",
                    "description": (
                        "OBRIGATÓRIO quando não souber responder ou precisar de humano: envia WhatsApp aos contatos "
                        "cadastrados em Comportamento → Contatos escalonamento, com resumo automático da conversa. "
                        "Chame ANTES de enviar_mensagem_texto_ao_cliente. Passe motivo claro e resumo curto do pedido do cliente. "
                        "O sistema também avisa sozinho em falhas de boleto. Depois informe o cliente que a equipe foi avisada."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "motivo": {
                                "type": "string",
                                "description": "Motivo do escalonamento (ex.: boleto não vinculado ao contato).",
                            },
                            "resumo": {
                                "type": "string",
                                "description": (
                                    "Opcional. Resumo narrativo para a equipe (frases com A cliente/A IA/A Agente). "
                                    "O sistema também gera resumo automático se omitir."
                                ),
                            },
                        },
                        "required": ["motivo"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "cadastrar_cliente_pelo_documento",
                    "description": (
                        "Cadastra automaticamente o cliente no sistema (equivalente a Clientes → Novo) usando os dados "
                        "extraídos do RG/CNH enviado no WhatsApp: nome completo, CPF, data de nascimento, filiação (nome da mãe/pai), "
                        "RG, sexo e endereço quando legíveis. Também vincula o telefone do WhatsApp e associa à conversa. "
                        "OBRIGATÓRIO quando contato_e_cliente_cadastrado for false, o cliente enviou documento e quer contratar/cadastrar plano "
                        "(cliente_indicou_documento_proprio ou pode_cadastrar_cliente_pelo_documento true). "
                        "Chame ANTES de pedir nome, CPF ou data de nascimento manualmente. "
                        "PROIBIDO afirmar 'cadastro realizado/com sucesso' sem esta tool retornar ok=true. "
                        "Depois confirme ao cliente com enviar_mensagem_texto_ao_cliente e siga com valores/plano."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "motivo": {
                                "type": "string",
                                "description": "Opcional. Ex.: cliente enviou CNH e quer se cadastrar no plano.",
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
            "avisar_equipe_escalonamento": lambda motivo, resumo=None, **_: tool_avisar_equipe_escalonamento(
                str(motivo), str(resumo) if resumo is not None else None
            ),
            "enviar_mensagem_texto_ao_cliente": lambda texto, **_: tool_enviar_texto(str(texto)),
            "enviar_link_boleto_parcela": lambda parcela_id=None, parcela_ids=None, **kw: tool_enviar_boleto(
                parcela_id, parcela_ids, **kw
            ),
            "enviar_recibo_parcela": lambda parcela_id=None, **_: tool_enviar_recibo(
                int(parcela_id) if parcela_id is not None else None
            ),
            "cadastrar_cliente_pelo_documento": lambda **_: tool_cadastrar_cliente_documento(),
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
            "(0) SEMPRE chame buscar_contexto_cliente como PRIMEIRA ferramenta — antes de enviar_mensagem_texto_ao_cliente; "
            "(1) leia instrucoes_atendimento (Comportamento por situação do painel vem no topo), situacao_atendimento_atual, "
            "situacoes_comportamento e instrucoes_comportamento_consolidadas; "
            "(2) enviar_mensagem_texto_ao_cliente UMA vez com a resposta completa (máximo 180 caracteres por bolha — "
            "se passar, o sistema divide em blocos; prefira linha em branco ou [[BLOCO]] entre blocos); "
            "se forem vários tópicos distintos (ex.: dois planos diferentes), separe cada bloco com uma linha em branco "
            "(duas quebras de linha) ou com [[BLOCO]] entre eles — o sistema enviará cada bloco como mensagem separada "
            "com o intervalo configurado no painel; "
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
            "Se a mensagem do cliente for só um marcador como [áudio], [imagem], [vídeo], [documento], [documento: arquivo.pdf], [sticker] ou [modelo], "
            "significa que ele enviou esse tipo de mídia (muitas vezes sem legenda): reconheça isso, responda de forma útil "
            "via enviar_mensagem_texto_ao_cliente e, se precisar de detalhes, peça que escreva por texto ou use buscar_contexto_cliente. "
            "Se mensagem_anterior_enviada_pela_empresa for true OU conversa_iniciada_pela_empresa for true: "
            "a empresa enviou o comprovante/mensagem anterior — PROIBIDO perguntar o que fazer com imagem ou documento; "
            "não reaja a arquivos enviados pela empresa; só responda se o cliente fez um pedido claro (boleto, plano, cadastro, etc.). "
            "Se saudacao_sem_pedido_rodada for true e documento_foco_rodada for false: o cliente mandou só cumprimento (bom dia, tudo bem). "
            "Responda de forma breve e cordial ao cumprimento. PROIBIDO mencionar documento, CNH, RG ou perguntar o que fazer com arquivo — "
            "documento antigo no histórico NÃO é assunto desta rodada. "
            "Se situacao_atendimento_atual for imagem_ou_documento OU cliente_enviou_arquivo_nesta_rodada for true "
            "(e documento_foco_rodada for true quando indicado no prefixo): "
            "OBRIGATÓRIO ler instrucao_obrigatoria_imagem_ou_documento, instrucao_leitura_documento, instrucao_situacao_atual, "
            "exemplo_resposta_documento_recebido, instrucao_resposta_documento_recebido, "
            "instrucoes_comportamento_consolidadas e situacoes_comportamento.imagem_ou_documento (texto cadastrado no painel Comportamento). "
            "NÃO responda com saudação genérica \"Como posso te ajudar\" — siga as orientações do painel para imagem/documento. "
            "Se mensagens_recentes ou mensagem_texto tiverem \"leitura automática do documento\", USE TIPO_DOCUMENTO, NOME_TITULAR e PRIMEIRO_NOME_TITULAR. "
            "Na PRIMEIRA resposta após receber o arquivo, siga exemplo_resposta_documento_recebido "
            "(ex.: \"Recebi a CNH do Douglas, o que você gostaria que eu fizesse com o documento dele?\") "
            "SOMENTE se o cliente NÃO tiver dito no histórico o que quer (cadastro, plano, atualizar/completar cadastro). "
            "Se mensagens_recentes ou a mensagem atual tiverem pedido de cadastro/atualização (ex.: \"atualiza meu cadastro\"), "
            "OBRIGATÓRIO chamar cadastrar_cliente_pelo_documento — PROIBIDO perguntar o que fazer com o documento. "
            "Se o cliente perguntar \"qual o nome\" ou sobre o documento, leia nome_titular_documento_recente, "
            "tipo_documento_recente e instrucao_resposta_pergunta_sobre_documento — responda com o titular do documento; "
            "NÃO peça o nome de quem está no WhatsApp. "
            "Se cliente_indicou_documento_proprio for true OU o cliente disser que o documento é dele / quer se cadastrar no plano, "
            "use nome_cliente_ja_informado (nome lido no documento) e instrucao_documento_pertence_ao_cliente — "
            "NÃO peça o nome de novo; avance no cadastro ou no assunto pedido. "
            "Se cadastro_automatico_realizado for true OU cadastro_automatico_cliente_id existir: o cadastro JÁ FOI FEITO com dados do documento — "
            "confirme ao cliente (nome em cadastro_automatico_nome) e siga com o plano; PROIBIDO pedir nome completo, CPF, nova foto ou nascimento. "
            "PROIBIDO dizer 'cadastro realizado com sucesso' ou equivalente sem cadastro_automatico_realizado=true "
            "ou sem cadastrar_cliente_pelo_documento retornando ok=true nesta rodada. "
            "Se precisa_cadastrar_pelo_documento_agora for true OU cadastro_automatico_falhou existir: "
            "OBRIGATÓRIO chamar cadastrar_cliente_pelo_documento como PRIMEIRA ferramenta nesta rodada — "
            "NÃO diga que a equipe vai finalizar manualmente antes de chamar a tool. "
            "Se a tool retornar ok=true: confirme o cadastro pelo nome do documento e informe valores/planos (buscar_contexto_cliente). "
            "Só chame avisar_equipe_escalonamento se cadastrar_cliente_pelo_documento retornar ok=false. "
            "Se proibido_pedir_cpf_ou_dados_documento for true: NÃO inicie fluxo manual de cadastro nem peça CPF/dados que já constam na leitura do documento — "
            "o sistema grava automaticamente via cadastrar_cliente_pelo_documento. "
            "Se contato_e_cliente_cadastrado for false e o cliente enviou RG/CNH querendo plano/cadastro: "
            "OBRIGATÓRIO chamar cadastrar_cliente_pelo_documento (lê nome, CPF, nascimento, filiação do documento e grava no cadastro) "
            "antes de pedir dados manualmente — veja instrucao_cadastro_cliente_whatsapp e dados_extraidos_documento em buscar_contexto_cliente. "
            "Para dados de contrato, parcelas ou histórico da conversa, use buscar_contexto_cliente (lá vêm mensagens_recentes, "
            "regras de pausa pós-atendente humano, status, instrucao_limite_caracteres_resposta, max_caracteres_bloco_resposta e o objeto data.saudacao). "
            "Saudação e continuidade: após buscar_contexto_cliente, leia body.data.saudacao (se o retorno vier em envelope HTTP, use body.data). "
            "Se conversa_ja_tem_historico for true OU evitar_repetir_saudacao_ciclo_completo for true: NÃO comece com \"Oi\", \"Olá\", \"Oi, Nome\", nem \"Bom dia/Boa tarde/Boa noite\" como abertura — isso reinicia o atendimento; responda direto ao que o cliente disse. "
            "modelo_com_nome e modelo_sem_nome_primeiro_contato são só para conversa_ja_tem_historico false (primeiro contato no chat). "
            "Se conversa_ja_tem_historico for true, o atendimento já começou — sem tom de primeira conversa. "
            "Nunca use 'Seja bem-vindo(a)', 'bem-vindo à empresa' nem reinicie como primeiro contato. "
            "Com nome_primeiro (e nome_declarado_pelo_cliente_nas_mensagens em saudacao) use quem está FALANDO no chat; NUNCA use termos de plano/contrato como nome (carência, mensalidade, contrato, parcela, titular). "
            "Se não houver nome de pessoa em saudacao nem nome_cliente_ja_informado, cumprimente sem inventar nome — "
            "PROIBIDO \"Oi, **!\", \"Oi, !\" ou asteriscos no lugar do nome. "
            "Quando financeiro_mesmo_cadastro_que_vinculo_conversa for false, "
            "leia instrucao_como_chamar_o_cliente e não trate o titular do contrato (cliente_nome) como o nome do interlocutor. "
            "Com historico mas sem nome em saudacao: mesmo assim não abra com Oi/Bom dia/Boa noite — vá direto ao ponto. "
            "Só com conversa_ja_tem_historico false use modelo_sem_nome_primeiro_contato ou modelo_com_nome para abrir o primeiro atendimento neste chat. "
            "Horário e manhã/tarde/noite vêm de saudacao (timezone e hora_local); não contradiga o periodo_dia. "
            "Intervenção humana: se um atendente enviou mensagem pelo painel, a IA fica pausada por um período indicado em "
            "intervencao_humana_minutos_inatividade; cada nova mensagem humana recomeça esse prazo. "
            "Mensagem do cliente não encerra essa pausa — você só é chamado de novo quando já passou o silêncio humano exigido. "
            "Ao retomar, leia mensagens_recentes e não desfaça o que o humano acordou com o cliente. "
            "Se o cliente pedir boleto, linha digitável, código de barras ou PIX: use enviar_link_boleto_parcela com parcela_id ou parcela_ids "
            "(parcelas em atraso = parcelas_em_atraso_lista; parcelas a vencer = itens de parcelas_em_aberto_lista com em_atraso=false). "
            "NÃO reenvie a mesma parcela se a tool retornar skipped_duplicate — diga que já enviou há pouco. "
            "Não envie texto depois repetindo link/linha/PIX — a ferramenta já manda em bolhas separadas. "
            "Para 'de qual mês é essa parcela', use mes_referencia_vencimento do item correspondente. "
            "Para valor da mensalidade, quanto paga ou preço do plano, siga instrucao_valor_mensalidade_cliente e o objeto contratos_ativos em buscar_contexto_cliente. "
            "Para 'quem está no meu plano', 'dependentes', 'nomes das pessoas no plano', filhos ou cônjuge: leia instrucao_dependentes_plano e dependentes_cadastro_cliente "
            "(lista_nomes_para_resposta = titular + dependentes cadastrados no sistema). Liste os NOMES reais; PROIBIDO só dizer regra genérica "
            "(ex. 'você, esposo e filhos') sem citar os nomes de lista_nomes_para_resposta. "
            "Para próxima parcela a vencer (data e valor), siga estritamente instrucao_proxima_parcela_vencimento e proxima_parcela_em_aberto; não infira só com dia_vencimento do contrato. "
            "Para dizer se há parcelas em aberto, atraso ou se o cliente 'está em dia', siga instrucao_parcelas_aberto_atraso e os números em financeiro_resumo (parcelas_em_aberto, parcelas_em_atraso, data_referencia_hoje). "
            "Quando pedirem parcelas EM ATRASO ou VENCIDAS, liste SOMENTE parcelas_em_atraso_lista — NUNCA trate parcelas futuras (em_atraso=false) como vencidas. "
            "Siga instrucao_dados_financeiros_vs_historico: se o histórico (mensagens_recentes do agente_ia) disser outra quantidade ou meses de parcelas vencidas, IGNORE o histórico e use os arrays financeiros atuais do JSON. "
            "Não contradiga os contadores de financeiro_resumo. "
            "Se data.instrucao_cadastro_sem_contrato_ativo_listado vier preenchida ou cadastro_financeiro_sem_contrato_ativo_listado for true, siga essa instrução: "
            "se contratos_cancelados tiver itens ou qtd_contratos_cancelados_no_cadastro > 0 ou instrucao_contratos_cancelados vier preenchida, informe que o plano consta cancelado no cadastro e cite numero_contrato/plano_nome/data_cancelamento; "
            "mencione contrato suspenso se qtd_contratos_suspenso_no_cadastro > 0; não diga que 'não há contrato' de forma absoluta quando houver cancelados ou suspensos listados. "
            "Se o cliente pedir para falar com o financeiro, confirme de forma breve e evite repetir o mesmo bloco inteiro sobre 'sem contrato ativo' das mensagens anteriores. "
            "Se o cliente perguntar de onde vieram plano, valores ou datas (ex.: 'de onde você pegou', 'confirma aí'), siga instrucao_proveniencia_dados: cite contratos_ativos[].id, numero_contrato e plano_nome do JSON; "
            "não responda com frase genérica que não explica a origem. Se instrucao_multiplos_contratos_ativos vier preenchida, há mais de um contrato ativo — não misture dados entre eles. "
            "Escalonamento humano: contatos_escalonamento_whatsapp e instrucao_contatos_escalonamento em buscar_contexto_cliente. "
            "Se enviar_link_boleto_parcela falhar (parcela não vinculada, sem cobrança, etc.), o sistema já avisa a equipe no WhatsApp — "
            "informe o cliente com cordialidade e NÃO invente boleto. Se não souber responder ou for dizer que vai verificar/confirmar informação: "
            "chame avisar_equipe_escalonamento (motivo + resumo) ANTES de enviar_mensagem_texto_ao_cliente — o sistema também avisa automaticamente "
            "os contatos de escalonamento quando você promete verificar; use a ferramenta para reforço. "
            "Modo áudio: se buscar_contexto_cliente retornar deve_usar_instrucoes_audio_agora=true, "
            "aplique instrucoes_atendimento_audio do JSON (não só a aba Texto) para dependentes, filhos, casamento e inclusões. "
            "Não invente valores ou links; use apenas o retorno das ferramentas. "
            "Após chamar enviar_mensagem_texto_ao_cliente (sucesso ou recusa do sistema), não chame essa ferramenta de novo nesta interação — "
            "aguarde o cliente responder. Se a tool retornar ok:false, corrija na PRÓXIMA mensagem do cliente, não envie várias bolhas seguidas. "
            "PROIBIDO mandar sequência de mensagens sem o cliente falar entre elas (ex.: recebi documento + vou cadastrar + peça foto + tentei de novo). "
            "Uma interação = no máximo UMA bolha de texto ao cliente (o sistema pode dividir só se passar de 180 caracteres). "
            "Após chamar enviar_mensagem_texto_ao_cliente com sucesso nesta rodada, não produza mensagem adicional ao usuário: "
            "não gere novo raciocínio nem nova bolha — a ferramenta já enviou a resposta. "
            "Em cada passagem de ferramentas: no máximo UMA chamada a enviar_mensagem_texto_ao_cliente (uma bolha por vez); "
            "não envie duas saudações ou duas perguntas seguidas na mesma rodada.\n\n"
            f"Instruções adicionais da empresa:\n{extra_system_instructions or '(nenhuma)'}"
            + (
                "\n\nATENÇÃO MODO ÁUDIO: deve_usar_instrucoes_audio_agora=true nesta conversa. "
                "Perguntas sobre dependentes, filho casado, inclusão no plano etc. devem seguir "
                "instrucoes_atendimento_audio (também repetido no fim deste bloco se vier do Laravel)."
                if deve_usar_instrucoes_audio
                else ""
            )
        )

        messages: list[Any] = [
            SystemMessage(content=system),
            HumanMessage(content=user_message),
        ]

        max_turns = 4
        last_text = ""
        entregou_algo_ao_cliente_whatsapp = False
        tentou_enviar_texto_whatsapp = False
        contexto_obtido_no_turno: int | None = None

        for turn in range(max_turns):
            if entregou_algo_ao_cliente_whatsapp or tentou_enviar_texto_whatsapp:
                logger.info(
                    "[agente] parada_entrega_ja_realizada correlation_id=%s conversation_id=%s turn=%s "
                    "entregou=%s tentou_texto=%s",
                    cid,
                    conversation_id,
                    turn,
                    entregou_algo_ao_cliente_whatsapp,
                    tentou_enviar_texto_whatsapp,
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

                if name == "enviar_mensagem_texto_ao_cliente" and tentou_enviar_texto_whatsapp:
                    logger.warning(
                        "[agente] enviar_texto_bloqueado_ja_tentou_nesta_interacao correlation_id=%s turn=%s conversation_id=%s",
                        cid,
                        turn,
                        conversation_id,
                    )
                    payload = json.dumps(
                        {
                            "ok": True,
                            "skipped_duplicate": True,
                            "message": (
                                "Já houve tentativa de enviar mensagem ao cliente nesta interação. "
                                "Aguarde a resposta do cliente — não envie outra bolha."
                            ),
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
                        if name == "enviar_mensagem_texto_ao_cliente":
                            tentou_enviar_texto_whatsapp = True
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "[agente] llm_tool_exec_erro correlation_id=%s tool=%s erro=%s",
                            cid,
                            name,
                            exc,
                        )
                        payload = json.dumps({"erro": str(exc)}, ensure_ascii=False)
                messages.append(ToolMessage(content=payload, tool_call_id=tid))

            if entregou_algo_ao_cliente_whatsapp or tentou_enviar_texto_whatsapp:
                logger.info(
                    "[agente] parada_apos_entrega_ou_tentativa_texto correlation_id=%s turn=%s conversation_id=%s "
                    "(uma mensagem por interação; aguardar resposta do cliente)",
                    cid,
                    turn,
                    conversation_id,
                )
                break

        if not entregou_algo_ao_cliente_whatsapp and not tentou_enviar_texto_whatsapp:
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
