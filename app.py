from __future__ import annotations

import base64
import hmac
import io
import json
import math
import os
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps
from starlette.middleware.cors import CORSMiddleware

PROFILES = {
    32: {"visible": 32.0, "fold": 41.0, "total": 48.0},
    44: {"visible": 44.0, "fold": 53.0, "total": 60.0},
    58: {"visible": 58.0, "fold": 67.0, "total": 74.0},
}
DPI = 300
MAX_BODY = 24 * 1024 * 1024
app = FastAPI(title="DuAmorPaper Button Renderer", docs_url=None, redoc_url=None, openapi_url=None)
ALLOWED_ORIGINS = [v.strip() for v in os.getenv("ALLOWED_ORIGINS", "https://chatgpt.com,https://chat.openai.com").split(",") if v.strip()]
app.add_middleware(
    CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept", "MCP-Protocol-Version"],
)

def circle_mask(size: int, radius: float) -> Image.Image:
    scale = 3
    high = Image.new("L", (size * scale, size * scale), 0)
    ImageDraw.Draw(high).ellipse((size*scale/2-radius*scale, size*scale/2-radius*scale,
                                  size*scale/2+radius*scale, size*scale/2+radius*scale), fill=255)
    mask = high.resize((size, size), Image.Resampling.LANCZOS)
    pix = mask.load()
    c = size / 2
    for y in range(size):
        for x in range(size):
            if math.hypot(x + .5 - c, y + .5 - c) > radius:
                pix[x, y] = 0
    return mask

def decode_png(value: str, field: str) -> Image.Image:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Campo obrigatório ausente: {field}")
    try:
        raw = base64.b64decode(value, validate=True)
        with Image.open(io.BytesIO(raw)) as im:
            if im.format != "PNG":
                raise ValueError(f"{field} precisa ser PNG")
            return im.convert("RGBA")
    except (ValueError, OSError) as exc:
        raise ValueError(f"PNG inválido em {field}") from exc

def apply_mask(layer: Image.Image, mask: Image.Image) -> Image.Image:
    out = layer.copy()
    out.putalpha(ImageChops.multiply(out.getchannel("A"), mask))
    return out

def load_font(path: str | None, size: int):
    candidates = [path] if path else []
    candidates += [
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()

def wrap_words(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    probe = ImageDraw.Draw(Image.new("L", (1, 1), 0))
    lines: list[str] = []
    for paragraph in text.split("\n"):
        words = paragraph.split(" ")
        line = ""
        for word in words:
            test = word if not line else line + " " + word
            if line and probe.textbbox((0, 0), test, font=font)[2] > max_width:
                lines.append(line)
                line = word
            else:
                line = test
        lines.append(line)
    return lines

def render_layers(args: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    size_mm = args.get("size_mm")
    if isinstance(size_mm, bool) or size_mm not in PROFILES:
        raise ValueError("size_mm deve ser 32, 44 ou 58")
    p = PROFILES[size_mm]
    pixels = int(round(p["total"] / 25.4 * DPI))
    canvas_size = (pixels, pixels)
    radius = pixels / 2
    visible_r = radius * p["visible"] / p["total"]
    fold_r = radius * p["fold"] / p["total"]
    background = decode_png(args.get("background_png_base64"), "background_png_base64")
    background = ImageOps.fit(background, canvas_size, method=Image.Resampling.LANCZOS)
    # Background must cover the whole cut disk; transparency is added only outside the cut.
    background.putalpha(255)
    canvas = background
    # Decorative pixels can exist only in the ring from the visible edge to the fold.
    decorations_b64 = args.get("decorations_png_base64")
    if decorations_b64:
        decorations = ImageOps.fit(decode_png(decorations_b64, "decorations_png_base64"), canvas_size,
                                   method=Image.Resampling.LANCZOS)
        outer = circle_mask(pixels, fold_r)
        inner = circle_mask(pixels, visible_r)
        annulus = ImageChops.subtract(outer, inner)
        canvas.alpha_composite(apply_mask(decorations, annulus))
    # The subject layer must have transparent background. Scale the nontransparent subject
    # into the upper visible zone, then clip by the safe circle.
    subject = decode_png(args.get("subject_png_base64"), "subject_png_base64")
    bbox = subject.getchannel("A").getbbox()
    if bbox:
        subject = subject.crop(bbox)
        # Choose a bounded box and then enforce a radial bound on all four corners,
        # so the circle mask will not cut off the supplied character's silhouette.
        box_w = int(visible_r * 1.25)
        box_h = int(visible_r * .95)
        subject.thumbnail((box_w, box_h), Image.Resampling.LANCZOS)
        layer = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
        cx, cy = pixels / 2, pixels / 2 - visible_r * .15
        safe_radius = visible_r * .94
        diagonal = math.hypot(subject.width / 2, subject.height / 2)
        radial_available = safe_radius - abs(cy - pixels / 2)
        if diagonal > radial_available:
            factor = radial_available / diagonal
            subject = subject.resize((max(1, int(subject.width * factor)), max(1, int(subject.height * factor))), Image.Resampling.LANCZOS)
        layer.alpha_composite(subject, (int(cx-subject.width/2), int(cy-subject.height/2)))
        canvas.alpha_composite(apply_mask(layer, circle_mask(pixels, safe_radius)))
    exact_text = args.get("exact_text", "")
    if not isinstance(exact_text, str):
        raise ValueError("exact_text deve ser texto")
    if exact_text.strip().upper() != "SEM TEXTO" and exact_text.strip():
        max_w = int(visible_r * 1.52)
        max_h = int(visible_r * .62)
        options = args.get("text_options") or {}
        if not isinstance(options, dict):
            raise ValueError("text_options deve ser objeto")
        start = int(max_w * float(options.get("font_fraction", .20)))
        stroke = max(0, min(8, int(options.get("stroke_width", 1))))
        spacing = max(0, min(30, int(options.get("spacing", 4))))
        selected = None
        temp_draw = ImageDraw.Draw(Image.new("RGBA", (1, 1), (0, 0, 0, 0)))
        for fs in range(max(10, start), 7, -1):
            font = load_font(options.get("font_path"), fs)
            lines = wrap_words(exact_text, font, max_w)
            bb = temp_draw.multiline_textbbox((0, 0), "\n".join(lines), font=font,
                                               spacing=spacing, stroke_width=stroke)
            if bb[2]-bb[0] <= max_w and bb[3]-bb[1] <= max_h:
                selected = (font, lines, bb)
                break
        if not selected:
            raise ValueError("Texto longo demais para a área visível; reduza a frase somente após alterar o briefing.")
        font, lines, bb = selected
        tw, th = bb[2]-bb[0], bb[3]-bb[1]
        text_layer = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(text_layer)
        x = (pixels - tw) // 2 - bb[0]
        y = int(pixels/2 + visible_r*.16) - bb[1]
        draw.multiline_text((x, y), "\n".join(lines), font=font,
                            fill=options.get("color", "#F5D36C"),
                            stroke_width=stroke, stroke_fill=options.get("stroke_fill", "#42205B"),
                            spacing=spacing, align="center")
        canvas.alpha_composite(apply_mask(text_layer, circle_mask(pixels, visible_r * .94)))
    # The final file has real transparency outside the total cut, never a white mockup.
    canvas = apply_mask(canvas, circle_mask(pixels, radius))
    output = io.BytesIO()
    canvas.save(output, format="PNG", dpi=(DPI, DPI), optimize=True)
    summary = {
        "status": "rendered",
        "size_mm": size_mm,
        "visible_mm": p["visible"],
        "fold_mm": p["fold"],
        "total_cut_mm": p["total"],
        "pixels": pixels,
        "dpi": DPI,
        "geometry_masks_applied": True,
        "outer_cut_is_transparent": True,
        "decorations_clipped_to_fold_ring": True,
        "text_is_separate_layer": exact_text.strip().upper() != "SEM TEXTO",
        "note": "For semantic QA, inspect the supplied layer content; geometric masks cannot remove decorations already baked into the background layer."
    }
    return output.getvalue(), summary

TOOLS = [
    {"name":"get_gabarito_profile","description":"Return exact official button diameters and zone rules.","inputSchema":{"type":"object","properties":{"size_mm":{"type":"integer","enum":[32,44,58]}},"required":["size_mm"]}},
    {"name":"render_button_from_layers","description":"Compose separate PNG layers into one circular button image using strict visible/fold/cut masks. Subject PNG must have transparency; decorations are clipped to visible-to-fold ring; background fills the cut.","inputSchema":{"type":"object","properties":{
        "size_mm":{"type":"integer","enum":[32,44,58]},
        "background_png_base64":{"type":"string","description":"PNG base64, continuous background without isolated decorations."},
        "subject_png_base64":{"type":"string","description":"PNG base64 with transparent background and the main character/object."},
        "decorations_png_base64":{"type":"string","description":"Optional transparent PNG base64 containing secondary decoration only."},
        "exact_text":{"type":"string","description":"Exact phrase or SEM TEXTO."},
        "text_options":{"type":"object"}
    },"required":["size_mm","background_png_base64","subject_png_base64","exact_text"]}}
]

def handle_rpc(msg: dict[str, Any]):
    method, ident = msg.get("method"), msg.get("id")
    if method == "notifications/initialized":
        return None
    if method == "initialize":
        return {"jsonrpc":"2.0","id":ident,"result":{"protocolVersion":"2024-11-05","capabilities":{"tools":{}},"serverInfo":{"name":"duamorpaper-button-renderer","version":"1.1.0"}}}
    if method == "ping":
        return {"jsonrpc":"2.0","id":ident,"result":{}}
    if method == "tools/list":
        return {"jsonrpc":"2.0","id":ident,"result":{"tools":TOOLS}}
    if method == "tools/call":
        params = msg.get("params") or {}
        name, args = params.get("name"), params.get("arguments") or {}
        try:
            if name == "get_gabarito_profile":
                size = args.get("size_mm")
                if size not in PROFILES:
                    raise ValueError("size_mm deve ser 32, 44 ou 58")
                p = PROFILES[size]
                result = {"size_mm":size,"visible_mm":p["visible"],"fold_mm":p["fold"],"total_cut_mm":p["total"],
                          "rule":"Texto e conteúdo principal dentro da área visível; decoração secundária somente até a dobra; dobra ao corte apenas fundo contínuo."}
                return {"jsonrpc":"2.0","id":ident,"result":{"content":[{"type":"text","text":json.dumps(result,ensure_ascii=False)}]}}
            if name == "render_button_from_layers":
                raw, summary = render_layers(args)
                return {"jsonrpc":"2.0","id":ident,"result":{"content":[
                    {"type":"text","text":json.dumps(summary,ensure_ascii=False)},
                    {"type":"image","mimeType":"image/png","data":base64.b64encode(raw).decode("ascii")}
                ]}}
            raise ValueError("Ferramenta desconhecida")
        except Exception as exc:
            return {"jsonrpc":"2.0","id":ident,"error":{"code":-32602,"message":str(exc)}}
    if ident is None:
        return None
    return {"jsonrpc":"2.0","id":ident,"error":{"code":-32601,"message":"Method not found"}}

@app.get("/healthz")
async def healthz():
    return {"status":"ok","service":"duamorpaper-button-renderer-mcp"}

@app.post("/mcp")
async def mcp_post(request: Request):
    key = os.getenv("MCP_API_KEY", "")
    if not key:
        return JSONResponse({"error":"MCP_API_KEY is not configured"}, status_code=503)
    origin = request.headers.get("origin")
    if origin and origin not in ALLOWED_ORIGINS:
        return JSONResponse({"error":"Origin not allowed"}, status_code=403)
    auth = request.headers.get("authorization", "")
    supplied = auth[7:].strip() if auth.lower().startswith("bearer ") else request.headers.get("x-mcp-api-key", "")
    if not supplied or not hmac.compare_digest(supplied, key):
        return JSONResponse({"error":"Unauthorized"}, status_code=401)
    raw = await request.body()
    if len(raw) > MAX_BODY:
        return JSONResponse({"error":"Request body too large"}, status_code=413)
    try:
        msg = json.loads(raw)
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0" or not isinstance(msg.get("method"), str):
            raise ValueError
    except Exception:
        return JSONResponse({"error":"Invalid JSON-RPC request"}, status_code=400)
    response = handle_rpc(msg)
    if response is None:
        return Response(status_code=202)
    headers = {}
    if msg.get("method") == "initialize":
        headers["MCP-Protocol-Version"] = response["result"]["protocolVersion"]
    return JSONResponse(response, headers=headers)

@app.get("/mcp")
async def mcp_get():
    return Response(status_code=405, headers={"Allow":"POST, OPTIONS"})
