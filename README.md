# DuAmorPaper Button Renderer

Serviço HTTP/MCP autenticado para montar uma arte circular de button a partir de camadas PNG separadas.

## Perfis oficiais

| Button | Área visível | Dobra | Corte total |
|---|---:|---:|---:|
| 32 mm | 32 mm | 41 mm | 48 mm |
| 44 mm | 44 mm | 53 mm | 60 mm |
| 58 mm | 58 mm | 67 mm | 74 mm |

## Ferramentas MCP

- `get_gabarito_profile`: devolve as medidas oficiais do tamanho selecionado.
- `render_button_from_layers`: recebe o fundo contínuo, uma camada transparente para o personagem, decoração secundária opcional e o texto exato. Faz composição a 300 DPI, limita a decoração ao anel entre área visível e dobra e torna transparente o exterior do corte.

**Importante:** envie camadas separadas. O renderizador não separa automaticamente personagem e enfeites de uma imagem achatada nem consegue detectar semântica já desenhada na camada do fundo. A imagem final ainda precisa passar por inspeção visual.

## Execução local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
export MCP_API_KEY='use-uma-chave-longa-e-aleatoria'
uvicorn app:app --host 127.0.0.1 --port 8000
pytest -q
```

Health check: `GET /healthz`  
MCP HTTP endpoint: `POST /mcp`, com `Authorization: Bearer $MCP_API_KEY`.

## Implantação no Render

O arquivo `render.yaml` declara o serviço web, instala os requisitos, inicia Uvicorn, configura o health check e pede ao Render que gere `MCP_API_KEY`. Inicie a implantação com o Blueprint do Render usando [este link direto](https://render.com/deploy?repo=https://github.com/dimitrirj2005-a11y/duamorpaper-button-renderer). Revise o serviço e confirme a criação no Render. O Blueprint gera `MCP_API_KEY`; copie o valor pelo painel seguro do Render somente para configurar o cliente autorizado. Nunca coloque a chave secreta no código ou no Git.

## Limitação importante

O serviço aplica regras geométricas de forma determinística às camadas recebidas. Isso não prova que o personagem, texto ou fundo fornecidos estejam semanticamente corretos; a auditoria visual da arte específica continua obrigatória.
