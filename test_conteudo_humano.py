from agent_runner import _conteudo_humano, _serialize_lc_messages
from langchain_core.messages import HumanMessage


def test_somente_texto_permanece_string():
    assert _conteudo_humano("olá", None) == "olá"
    assert _conteudo_humano("olá", []) == "olá"


def test_imagem_vira_lista_multimodal():
    parts = _conteudo_humano(
        "histórico",
        [
            {"type": "text", "text": "veja"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,abc", "detail": "auto"},
            },
        ],
    )
    assert isinstance(parts, list)
    assert parts[0]["type"] == "text"
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"].startswith("data:image/")


def test_serialize_omite_base64_no_debug():
    msg = HumanMessage(
        content=[
            {"type": "text", "text": "oi"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,SECRET", "detail": "auto"}},
        ]
    )
    dumped = _serialize_lc_messages([msg])
    blob = str(dumped)
    assert "SECRET" not in blob
    assert "data_url_omitido" in blob
