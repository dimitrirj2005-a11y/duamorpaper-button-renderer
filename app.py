from __future__ import annotations

import base64
import hmac
import io
import json
import math
import os
import re
import time
import html
import secrets
import hashlib
from urllib.parse import urlencode, urlparse
from typing import Any

import httpx
import jwt
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
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

# OAuth 2.1 authorization server + protected MCP resource.
SERVICE_BASE = os.getenv("PUBLIC_BASE_URL", "https://duamorpaper-button-renderer-mcp.onrender.com").rstrip("/")
ISSUER = os.getenv("OAUTH_ISSUER", SERVICE_BASE).rstrip("/")
RESOURCE = os.getenv("OAUTH_RESOURCE", SERVICE_BASE).rstrip("/")
PROFILE_SCOPE = "buttons:profile"
RENDER_SCOPE = "buttons:render"
SUPPORTED_SCOPES = {PROFILE_SCOPE, RENDER_SCOPE}
TOKEN_URL = f"{ISSUER}/token"
AUTH_URL = f"{ISSUER}/authorize"
REGISTER_URL = f"{ISSUER}/register"
RESOURCE_METADATA_URL = f"{RESOURCE}/.well-known/oauth-protected-resource"
AUTH_CODES: dict[str, dict[str, Any]] = {}
USED_REFRESH_JTIS: set[str] = set()
USED_ASSERTION_JTIS: dict[str, float] = {}
LOGIN_FAILURES: dict[str, list[float]] = {}
AUTH_CODE_LIFETIME = 180
ACCESS_TOKEN_LIFETIME = 900
REFRESH_TOKEN_LIFETIME = 60 * 60 * 24 * 30
MAX_LOGIN_FAILURES = 8
LOGIN_FAILURE_WINDOW = 15 * 60

TOOLS = [
    {
        "name": "get_gabarito_profile",
        "description": "Return exact official button diameters and zone rules.",
        "securitySchemes": [{"type": "oauth2", "scopes": [PROFILE_SCOPE]}],
        "inputSchema": {
            "type": "object",
            "properties": {"size_mm": {"type": "integer", "enum": [32, 44, 58]}},
            "required": ["size_mm"],
        },
    },
    {
        "name": "render_button_from_layers",
        "description": "Compose separate PNG layers into one circular button image using strict visible/fold/cut masks. Subject PNG must have transparency; decorations are clipped to visible-to-fold ring; background fills the cut.",
        "securitySchemes": [{"type": "oauth2", "scopes": [RENDER_SCOPE]}],
        "inputSchema": {
            "type": "object",
            "properties": {
                "size_mm": {"type": "integer", "enum": [32, 44, 58]},
                "background_png_base64": {"type": "string", "description": "PNG base64, continuous background without isolated decorations."},
                "subject_png_base64": {"type": "string", "description": "PNG base64 with transparent background and the main character/object."},
                "decorations_png_base64": {"type": "string", "description": "Optional transparent PNG base64 containing secondary decoration only."},
                "exact_text": {"type": "string", "description": "Exact phrase or SEM TEXTO."},
                "text_options": {"type": "object"},
            },
            "required": ["size_mm", "background_png_base64", "subject_png_base64", "exact_text"],
        },
    },
]

def _secret() -> str:
    value = os.getenv("OAUTH_SIGNING_SECRET", "")
    if len(value) < 32:
        raise RuntimeError("OAUTH_SIGNING_SECRET is not configured or is too short")
    return value

def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

def _b64url_decode(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * ((4 - len(text) % 4) % 4))

def _valid_redirect_uri(uri: str) -> bool:
    try:
        parsed = urlparse(uri)
        if parsed.scheme != "https" or parsed.hostname != "chatgpt.com" or parsed.query or parsed.fragment:
            return False
        if parsed.path == "/connector_platform_oauth_redirect":
            return True
        return bool(re.fullmatch(r"/connector/oauth/[A-Za-z0-9_-]{1,160}", parsed.path))
    except Exception:
        return False

def _client_auth_methods(metadata: dict[str, Any]) -> set[str]:
    methods = metadata.get("token_endpoint_auth_methods_supported")
    if isinstance(methods, list) and methods:
        return {str(item) for item in methods}
    legacy = metadata.get("token_endpoint_auth_method", "none")
    return {str(legacy)}

def _validate_client_metadata(metadata: Any, expected_client_id: str | None = None) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise ValueError("Client metadata must be a JSON object")
    if expected_client_id and metadata.get("client_id") != expected_client_id:
        raise ValueError("client_id does not match the client metadata document")
    redirect_uris = metadata.get("redirect_uris")
    if not isinstance(redirect_uris, list) or not redirect_uris or len(redirect_uris) > 10:
        raise ValueError("Client must provide redirect_uris")
    if any(not isinstance(uri, str) or not _valid_redirect_uri(uri) for uri in redirect_uris):
        raise ValueError("Only approved ChatGPT HTTPS redirect URIs are allowed")
    methods = _client_auth_methods(metadata)
    if not methods or not methods.issubset({"none", "private_key_jwt"}):
        raise ValueError("Unsupported token endpoint authentication method")
    if "private_key_jwt" in methods:
        jwks_uri = metadata.get("jwks_uri")
        parsed = urlparse(jwks_uri or "")
        if (parsed.scheme != "https" or parsed.hostname != "chatgpt.com"
                or parsed.path != "/oauth/jwks.json" or parsed.query or parsed.fragment):
            raise ValueError("private_key_jwt requires the published ChatGPT JWKS URI")
    return metadata

def _is_chatgpt_cimd(client_id: str) -> bool:
    try:
        parsed = urlparse(client_id)
        return (
            parsed.scheme == "https"
            and parsed.hostname == "chatgpt.com"
            and parsed.port is None
            and parsed.query == ""
            and parsed.fragment == ""
            and bool(re.fullmatch(r"/oauth/(?:client\.json|[A-Za-z0-9_-]{1,160}/client\.json)", parsed.path))
        )
    except Exception:
        return False

def _dcr_client_id(metadata: dict[str, Any]) -> str:
    payload = _b64url(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    sig = _b64url(hmac.new(_secret().encode("utf-8"), payload.encode("ascii"), hashlib.sha256).digest())
    return f"dcr.{payload}.{sig}"

def _decode_dcr_client_id(client_id: str) -> dict[str, Any]:
    try:
        prefix, payload, sig = client_id.split(".", 2)
        expected = _b64url(hmac.new(_secret().encode("utf-8"), payload.encode("ascii"), hashlib.sha256).digest())
        if prefix != "dcr" or not hmac.compare_digest(sig, expected):
            raise ValueError("Invalid registered client identifier")
        metadata = json.loads(_b64url_decode(payload))
        metadata["client_id"] = client_id
        return _validate_client_metadata(metadata, client_id)
    except Exception as exc:
        raise ValueError("Invalid registered client identifier") from exc

async def _get_client_metadata(client_id: str) -> dict[str, Any]:
    if client_id.startswith("dcr."):
        return _decode_dcr_client_id(client_id)
    if not _is_chatgpt_cimd(client_id):
        raise ValueError("Unrecognized OAuth client_id")
    async with httpx.AsyncClient(timeout=7.0, follow_redirects=False) as client:
        response = await client.get(client_id, headers={"Accept": "application/json"})
        response.raise_for_status()
        metadata = response.json()
    return _validate_client_metadata(metadata, client_id)

def _clean_scope(scope: str | None) -> str:
    requested = set((scope or f"{PROFILE_SCOPE} {RENDER_SCOPE}").split())
    if not requested or not requested.issubset(SUPPORTED_SCOPES):
        raise ValueError("Invalid or unsupported scope")
    return " ".join(sorted(requested))

async def _verify_private_key_jwt(client_id: str, assertion: str, metadata: dict[str, Any]) -> None:
    header = jwt.get_unverified_header(assertion)
    if header.get("alg") != "RS256" or not header.get("kid"):
        raise ValueError("Invalid client assertion algorithm or key ID")
    jwks_uri = metadata.get("jwks_uri")
    async with httpx.AsyncClient(timeout=7.0, follow_redirects=False) as client:
        response = await client.get(jwks_uri, headers={"Accept": "application/json"})
        response.raise_for_status()
        jwks = response.json()
    keys = jwks.get("keys", []) if isinstance(jwks, dict) else []
    jwk_data = next((item for item in keys if item.get("kid") == header["kid"] and item.get("kty") == "RSA"), None)
    if not jwk_data:
        raise ValueError("Client assertion signing key not found")
    key = jwt.PyJWK.from_dict(jwk_data).key
    claims = jwt.decode(
        assertion, key, algorithms=["RS256"], audience=TOKEN_URL,
        options={"require": ["exp", "iat", "iss", "sub", "jti"]}, leeway=30,
    )
    if claims.get("iss") != client_id or claims.get("sub") != client_id:
        raise ValueError("Invalid client assertion issuer or subject")
    jti = str(claims["jti"])
    now = time.time()
    for old_jti, expiry in list(USED_ASSERTION_JTIS.items()):
        if expiry < now:
            USED_ASSERTION_JTIS.pop(old_jti, None)
    if jti in USED_ASSERTION_JTIS:
        raise ValueError("Client assertion has already been used")
    USED_ASSERTION_JTIS[jti] = float(claims["exp"])

async def _authenticate_oauth_client(form: dict[str, Any], client_id: str, metadata: dict[str, Any]) -> None:
    methods = _client_auth_methods(metadata)
    assertion = form.get("client_assertion")
    assertion_type = form.get("client_assertion_type")
    if assertion:
        if "private_key_jwt" not in methods:
            raise ValueError("This OAuth client cannot use private_key_jwt")
        if assertion_type != "urn:ietf:params:oauth:client-assertion-type:jwt-bearer":
            raise ValueError("Invalid client_assertion_type")
        await _verify_private_key_jwt(client_id, str(assertion), metadata)
        return
    if "none" not in methods:
        raise ValueError("OAuth client authentication required")

def _create_token(*, client_id: str, scope: str, resource: str, token_use: str, lifetime: int) -> str:
    now = int(time.time())
    claims = {
        "iss": ISSUER,
        "sub": "duamorpaper-owner",
        "aud": resource,
        "resource": resource,
        "client_id": client_id,
        "scope": scope,
        "token_use": token_use,
        "iat": now,
        "nbf": now - 2,
        "exp": now + lifetime,
        "jti": secrets.token_urlsafe(18),
    }
    return jwt.encode(claims, _secret(), algorithm="HS256")

def _token_response(client_id: str, scope: str, resource: str) -> dict[str, Any]:
    return {
        "access_token": _create_token(client_id=client_id, scope=scope, resource=resource, token_use="access", lifetime=ACCESS_TOKEN_LIFETIME),
        "token_type": "Bearer",
        "expires_in": ACCESS_TOKEN_LIFETIME,
        "refresh_token": _create_token(client_id=client_id, scope=scope, resource=resource, token_use="refresh", lifetime=REFRESH_TOKEN_LIFETIME),
        "scope": scope,
    }

def _error_response(error: str, description: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"error": error, "error_description": description}, status_code=status, headers={"Cache-Control": "no-store", "Pragma": "no-cache"})

def _oauth_login_page(params: dict[str, str], client_name: str, error: str = "") -> HTMLResponse:
    fields = ("client_id", "redirect_uri", "response_type", "scope", "state", "code_challenge", "code_challenge_method", "resource")
    hidden = "".join(
        f'<input type="hidden" name="{key}" value="{html.escape(params.get(key, ""), quote=True)}">'
        for key in fields
    )
    msg = f'<p class="error">{html.escape(error)}</p>' if error else ""
    body = f"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Autorizar DuAmorPaper</title>
<style>
body{{font:16px system-ui,sans-serif;background:#f7f4fb;color:#21162c;margin:0;padding:28px}}
main{{max-width:480px;margin:5vh auto;background:#fff;border:1px solid #ded5e9;border-radius:18px;padding:28px;box-shadow:0 8px 32px #21162c12}}
h1{{font-size:25px;margin:0 0 8px}} p{{line-height:1.5}} label{{display:block;margin:14px 0 6px;font-weight:600}}
input{{box-sizing:border-box;width:100%;padding:12px;border:1px solid #bfb3cc;border-radius:8px;font:inherit}}
button{{padding:12px 16px;border:0;border-radius:8px;font:inherit;cursor:pointer;margin:18px 6px 0 0}}
.primary{{background:#5b2b84;color:white}} .secondary{{background:#ece7f2;color:#382648}} .error{{color:#a4162a}}
small{{color:#655b6d}}
</style></head><body><main><h1>DuAmorPaper • Autorizar renderizador</h1>
<p>O cliente <strong>{html.escape(client_name)}</strong> solicita acesso às ferramentas de montagem de buttons. Autorize somente se você iniciou esta conexão.</p>
{msg}
<form method="post" action="/authorize/decision">
{hidden}
<label for="username">Usuário</label><input id="username" name="username" autocomplete="username" required>
<label for="password">Senha de autorização</label><input id="password" name="password" type="password" autocomplete="current-password" required>
<button class="primary" name="decision" value="approve">Autorizar acesso</button>
<button class="secondary" name="decision" value="deny" formnovalidate>Cancelar</button>
</form>
<p><small>O acesso é protegido por OAuth, código de autorização e PKCE. A senha não é enviada ao ChatGPT.</small></p>
</main></body></html>"""
    return HTMLResponse(body, headers={"Cache-Control": "no-store", "Pragma": "no-cache", "X-Frame-Options": "DENY", "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'"})

async def _validate_authorize_params(params: dict[str, str]) -> tuple[dict[str, Any], str]:
    client_id = params.get("client_id", "")
    redirect_uri = params.get("redirect_uri", "")
    if params.get("response_type") != "code":
        raise ValueError("Only response_type=code is supported")
    if not params.get("state"):
        raise ValueError("OAuth state is required")
    if params.get("code_challenge_method") != "S256" or not params.get("code_challenge"):
        raise ValueError("PKCE S256 is required")
    if not re.fullmatch(r"[A-Za-z0-9_-]{43,128}", params.get("code_challenge", "")):
        raise ValueError("Invalid PKCE challenge")
    if params.get("resource") != RESOURCE:
        raise ValueError("Missing or invalid resource parameter")
    metadata = await _get_client_metadata(client_id)
    if redirect_uri not in metadata.get("redirect_uris", []):
        raise ValueError("redirect_uri is not registered for this client")
    scope = _clean_scope(params.get("scope"))
    params["scope"] = scope
    return metadata, scope

@app.get("/.well-known/oauth-protected-resource")
async def protected_resource_metadata():
    return {
        "resource": RESOURCE,
        "authorization_servers": [ISSUER],
        "scopes_supported": sorted(SUPPORTED_SCOPES),
        "resource_documentation": "https://github.com/dimitrirj2005-a11y/duamorpaper-button-renderer",
    }

@app.get("/.well-known/oauth-authorization-server")
async def authorization_server_metadata():
    return {
        "issuer": ISSUER,
        "authorization_endpoint": AUTH_URL,
        "token_endpoint": TOKEN_URL,
        "registration_endpoint": REGISTER_URL,
        "response_types_supported": ["code"],
        "response_modes_supported": ["query"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "token_endpoint_auth_methods_supported": ["none", "private_key_jwt"],
        "code_challenge_methods_supported": ["S256"],
        "scopes_supported": sorted(SUPPORTED_SCOPES),
        "client_id_metadata_document_supported": True,
        "authorization_response_iss_parameter_supported": True,
    }

@app.post("/register")
async def register_oauth_client(request: Request):
    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError("Client metadata must be an object")
        metadata = _validate_client_metadata(body)
        methods = _client_auth_methods(metadata)
        # DCR public clients may use PKCE without a client secret. private_key_jwt
        # registrations must provide a trusted ChatGPT JWKS URL.
        client_id = _dcr_client_id(metadata)
        response = {
            "client_id": client_id,
            "client_id_issued_at": int(time.time()),
            "client_name": metadata.get("client_name", "ChatGPT MCP client"),
            "redirect_uris": metadata["redirect_uris"],
            "token_endpoint_auth_method": "none" if "none" in methods else "private_key_jwt",
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "scope": " ".join(sorted(SUPPORTED_SCOPES)),
        }
        return JSONResponse(response, status_code=201, headers={"Cache-Control": "no-store"})
    except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
        return _error_response("invalid_client_metadata", str(exc))

@app.get("/authorize")
async def authorize(request: Request):
    params = {k: v for k, v in request.query_params.items()}
    try:
        metadata, scope = await _validate_authorize_params(params)
        name = str(metadata.get("client_name") or "ChatGPT")
        if not os.getenv("OAUTH_LOGIN_PASSWORD"):
            return HTMLResponse("OAuth login is not configured. Set OAUTH_LOGIN_PASSWORD in Render.", status_code=503)
        _secret()
        return _oauth_login_page(params, name)
    except (ValueError, httpx.HTTPError, RuntimeError) as exc:
        return HTMLResponse(
            f"OAuth authorization request rejected: {html.escape(str(exc))}",
            status_code=400,
            headers={"Cache-Control": "no-store", "X-Frame-Options": "DENY"},
        )

@app.post("/authorize/decision")
async def authorize_decision(request: Request):
    try:
        form = {k: str(v) for k, v in (await request.form()).items()}
    except Exception:
        return _error_response("invalid_request", "Expected form-encoded authorization response.")
    decision = form.get("decision", "deny")
    params = {k: form.get(k, "") for k in ("client_id", "redirect_uri", "response_type", "scope", "state", "code_challenge", "code_challenge_method", "resource")}
    try:
        metadata, scope = await _validate_authorize_params(params)
    except (ValueError, httpx.HTTPError, RuntimeError) as exc:
        return HTMLResponse(f"OAuth request rejected: {html.escape(str(exc))}", status_code=400, headers={"Cache-Control": "no-store"})
    redirect_uri = params["redirect_uri"]
    state = params["state"]
    if decision != "approve":
        return RedirectResponse(redirect_uri + "?" + urlencode({"error": "access_denied", "state": state, "iss": ISSUER}), status_code=303)
    user = os.getenv("OAUTH_LOGIN_USERNAME", "duamorpaper")
    password = os.getenv("OAUTH_LOGIN_PASSWORD", "")
    remote = request.client.host if request.client else "unknown"
    attempts = [stamp for stamp in LOGIN_FAILURES.get(remote, []) if stamp > time.time() - LOGIN_FAILURE_WINDOW]
    LOGIN_FAILURES[remote] = attempts
    if len(attempts) >= MAX_LOGIN_FAILURES:
        return _oauth_login_page(params, str(metadata.get("client_name") or "ChatGPT"), "Muitas tentativas. Aguarde 15 minutos.") 
    supplied_user = form.get("username", "")
    supplied_password = form.get("password", "")
    if not password or not hmac.compare_digest(supplied_user, user) or not hmac.compare_digest(supplied_password, password):
        attempts.append(time.time())
        LOGIN_FAILURES[remote] = attempts
        return _oauth_login_page(params, str(metadata.get("client_name") or "ChatGPT"), "Usuário ou senha incorretos.")
    LOGIN_FAILURES.pop(remote, None)
    code = secrets.token_urlsafe(32)
    AUTH_CODES[code] = {
        "client_id": params["client_id"],
        "redirect_uri": redirect_uri,
        "code_challenge": params["code_challenge"],
        "resource": params["resource"],
        "scope": scope,
        "sub": "duamorpaper-owner",
        "exp": time.time() + AUTH_CODE_LIFETIME,
    }
    return RedirectResponse(redirect_uri + "?" + urlencode({"code": code, "state": state, "iss": ISSUER}), status_code=303)

@app.post("/token")
async def token_endpoint(request: Request):
    try:
        form = {k: str(v) for k, v in (await request.form()).items()}
    except Exception:
        return _error_response("invalid_request", "Expected application/x-www-form-urlencoded.", 400)
    client_id = form.get("client_id", "")
    try:
        metadata = await _get_client_metadata(client_id)
        await _authenticate_oauth_client(form, client_id, metadata)
    except (ValueError, httpx.HTTPError, RuntimeError, jwt.PyJWTError) as exc:
        return _error_response("invalid_client", str(exc), 401)
    grant_type = form.get("grant_type")
    resource = form.get("resource", "")
    if resource != RESOURCE:
        return _error_response("invalid_target", "The resource parameter must exactly match this MCP server.")
    if grant_type == "authorization_code":
        code = form.get("code", "")
        record = AUTH_CODES.get(code)
        if not record or float(record.get("exp", 0)) < time.time():
            AUTH_CODES.pop(code, None)
            return _error_response("invalid_grant", "Authorization code is invalid or expired.")
        if (record["client_id"] != client_id or record["redirect_uri"] != form.get("redirect_uri")
                or record["resource"] != resource):
            return _error_response("invalid_grant", "Authorization code binding does not match.")
        verifier = form.get("code_verifier", "")
        if not re.fullmatch(r"[A-Za-z0-9._~-]{43,128}", verifier):
            return _error_response("invalid_grant", "A valid PKCE verifier is required.")
        actual_challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
        if not hmac.compare_digest(actual_challenge, record["code_challenge"]):
            return _error_response("invalid_grant", "PKCE verification failed.")
        AUTH_CODES.pop(code, None)
        return JSONResponse(_token_response(client_id, record["scope"], resource), headers={"Cache-Control": "no-store", "Pragma": "no-cache"})
    if grant_type == "refresh_token":
        raw_refresh = form.get("refresh_token", "")
        try:
            claims = jwt.decode(raw_refresh, _secret(), algorithms=["HS256"], issuer=ISSUER, audience=RESOURCE,
                                options={"require": ["exp", "iat", "iss", "sub", "aud", "scope", "jti", "token_use", "client_id", "resource"]})
        except (jwt.PyJWTError, RuntimeError) as exc:
            return _error_response("invalid_grant", "Refresh token is invalid or expired.")
        old_jti = str(claims.get("jti", ""))
        if claims.get("token_use") != "refresh" or claims.get("client_id") != client_id or claims.get("resource") != resource:
            return _error_response("invalid_grant", "Refresh token client or resource binding mismatch.")
        if old_jti in USED_REFRESH_JTIS:
            return _error_response("invalid_grant", "Refresh token has already been rotated.")
        USED_REFRESH_JTIS.add(old_jti)
        return JSONResponse(_token_response(client_id, _clean_scope(str(claims["scope"])), resource), headers={"Cache-Control": "no-store", "Pragma": "no-cache"})
    return _error_response("unsupported_grant_type", "Only authorization_code and refresh_token are supported.")

def handle_rpc(msg: dict[str, Any], scopes: set[str]):
    method, ident = msg.get("method"), msg.get("id")
    if method == "notifications/initialized":
        return None
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": ident, "result": {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}, "serverInfo": {"name": "duamorpaper-button-renderer", "version": "1.2.0"}}}
    if method == "ping":
        return {"jsonrpc": "2.0", "id": ident, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": ident, "result": {"tools": TOOLS}}
    if method == "tools/call":
        params = msg.get("params") or {}
        name, args = params.get("name"), params.get("arguments") or {}
        needed = PROFILE_SCOPE if name == "get_gabarito_profile" else RENDER_SCOPE if name == "render_button_from_layers" else None
        if not needed:
            return {"jsonrpc": "2.0", "id": ident, "error": {"code": -32601, "message": "Unknown tool"}}
        if needed not in scopes:
            challenge = f'Bearer resource_metadata="{RESOURCE_METADATA_URL}", error="insufficient_scope", error_description="Missing required scope"'
            return {"jsonrpc": "2.0", "id": ident, "result": {"content": [{"type": "text", "text": "The OAuth access token does not include the required permission."}], "_meta": {"mcp/www_authenticate": [challenge]}, "isError": True}}
        try:
            if name == "get_gabarito_profile":
                size = args.get("size_mm")
                if isinstance(size, bool) or size not in PROFILES:
                    raise ValueError("size_mm deve ser 32, 44 ou 58")
                profile = PROFILES[size]
                result = {"size_mm": size, "visible_mm": profile["visible"], "fold_mm": profile["fold"], "total_cut_mm": profile["total"], "rule": "Texto e conteúdo principal dentro da área visível; decoração secundária somente até a dobra; dobra ao corte apenas fundo contínuo."}
                return {"jsonrpc": "2.0", "id": ident, "result": {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]}}
            raw, summary = render_layers(args)
            return {"jsonrpc": "2.0", "id": ident, "result": {"content": [
                {"type": "text", "text": json.dumps(summary, ensure_ascii=False)},
                {"type": "image", "mimeType": "image/png", "data": base64.b64encode(raw).decode("ascii")},
            ]}}
        except Exception as exc:
            return {"jsonrpc": "2.0", "id": ident, "error": {"code": -32602, "message": str(exc)}}
    if ident is None:
        return None
    return {"jsonrpc": "2.0", "id": ident, "error": {"code": -32601, "message": "Method not found"}}

@app.get("/healthz")
async def healthz():
    return {"status": "ok", "service": "duamorpaper-button-renderer-mcp", "auth": "oauth2"}

@app.get("/.well-known/oauth-protected-resource")
async def protected_resource_metadata():
    return {"resource": RESOURCE, "authorization_servers": [ISSUER], "scopes_supported": sorted(SUPPORTED_SCOPES),
            "resource_documentation": "https://github.com/dimitrirj2005-a11y/duamorpaper-button-renderer"}

@app.get("/.well-known/oauth-authorization-server")
async def authorization_server_metadata():
    return {
        "issuer": ISSUER, "authorization_endpoint": AUTH_URL, "token_endpoint": TOKEN_URL,
        "registration_endpoint": REGISTER_URL, "response_types_supported": ["code"],
        "response_modes_supported": ["query"], "grant_types_supported": ["authorization_code", "refresh_token"],
        "token_endpoint_auth_methods_supported": ["none", "private_key_jwt"],
        "code_challenge_methods_supported": ["S256"], "scopes_supported": sorted(SUPPORTED_SCOPES),
        "client_id_metadata_document_supported": True, "authorization_response_iss_parameter_supported": True,
    }

@app.post("/register")
async def register_oauth_client(request: Request):
    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError("Client metadata must be a JSON object")
        metadata = _validate_client_metadata(body)
        client_id = _dcr_client_id(metadata)
        methods = _client_auth_methods(metadata)
        return JSONResponse({
            "client_id": client_id, "client_id_issued_at": int(time.time()),
            "client_name": metadata.get("client_name", "ChatGPT MCP client"),
            "redirect_uris": metadata["redirect_uris"],
            "token_endpoint_auth_method": "none" if "none" in methods else "private_key_jwt",
            "grant_types": ["authorization_code", "refresh_token"], "response_types": ["code"],
            "scope": " ".join(sorted(SUPPORTED_SCOPES)),
        }, status_code=201, headers={"Cache-Control": "no-store"})
    except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
        return _error_response("invalid_client_metadata", str(exc))

@app.get("/authorize")
async def authorize(request: Request):
    params = {key: value for key, value in request.query_params.items()}
    try:
        metadata, _ = await _validate_authorize_params(params)
        if not os.getenv("OAUTH_LOGIN_PASSWORD"):
            return HTMLResponse("OAuth login is not configured. Set OAUTH_LOGIN_PASSWORD in Render.", status_code=503)
        _secret()
        return _oauth_login_page(params, str(metadata.get("client_name") or "ChatGPT"))
    except (ValueError, httpx.HTTPError, RuntimeError) as exc:
        return HTMLResponse(f"OAuth request rejected: {html.escape(str(exc))}", status_code=400, headers={"Cache-Control": "no-store", "X-Frame-Options": "DENY"})

@app.post("/authorize/decision")
async def authorize_decision(request: Request):
    try:
        form = {key: str(value) for key, value in (await request.form()).items()}
    except Exception:
        return _error_response("invalid_request", "Expected form-encoded authorization response.")
    params = {key: form.get(key, "") for key in ("client_id", "redirect_uri", "response_type", "scope", "state", "code_challenge", "code_challenge_method", "resource")}
    try:
        metadata, scope = await _validate_authorize_params(params)
    except (ValueError, httpx.HTTPError, RuntimeError) as exc:
        return HTMLResponse(f"OAuth request rejected: {html.escape(str(exc))}", status_code=400, headers={"Cache-Control": "no-store"})
    redirect_uri, state = params["redirect_uri"], params["state"]
    if form.get("decision", "deny") != "approve":
        return RedirectResponse(redirect_uri + "?" + urlencode({"error": "access_denied", "state": state, "iss": ISSUER}), status_code=302)
    username = os.getenv("OAUTH_LOGIN_USERNAME", "duamorpaper")
    password = os.getenv("OAUTH_LOGIN_PASSWORD", "")
    remote = request.client.host if request.client else "unknown"
    attempts = [stamp for stamp in LOGIN_FAILURES.get(remote, []) if stamp > time.time() - LOGIN_FAILURE_WINDOW]
    LOGIN_FAILURES[remote] = attempts
    client_name = str(metadata.get("client_name") or "ChatGPT")
    if len(attempts) >= MAX_LOGIN_FAILURES:
        return _oauth_login_page(params, client_name, "Muitas tentativas. Aguarde 15 minutos.")
    if not password or not hmac.compare_digest(form.get("username", ""), username) or not hmac.compare_digest(form.get("password", ""), password):
        attempts.append(time.time())
        LOGIN_FAILURES[remote] = attempts
        return _oauth_login_page(params, client_name, "Usuário ou senha incorretos.")
    LOGIN_FAILURES.pop(remote, None)
    code = secrets.token_urlsafe(32)
    AUTH_CODES[code] = {
        "client_id": params["client_id"], "redirect_uri": redirect_uri,
        "code_challenge": params["code_challenge"], "resource": params["resource"],
        "scope": scope, "sub": "duamorpaper-owner", "exp": time.time() + AUTH_CODE_LIFETIME,
    }
    return RedirectResponse(redirect_uri + "?" + urlencode({"code": code, "state": state, "iss": ISSUER}), status_code=302)

@app.post("/token")
async def token_endpoint(request: Request):
    try:
        form = {key: str(value) for key, value in (await request.form()).items()}
    except Exception:
        return _error_response("invalid_request", "Expected application/x-www-form-urlencoded.")
    client_id = form.get("client_id", "")
    try:
        metadata = await _get_client_metadata(client_id)
        await _authenticate_oauth_client(form, client_id, metadata)
    except (ValueError, httpx.HTTPError, RuntimeError, jwt.PyJWTError) as exc:
        return _error_response("invalid_client", str(exc), 401)
    grant_type = form.get("grant_type")
    resource = form.get("resource", "")
    if resource != RESOURCE:
        return _error_response("invalid_target", "The resource parameter must exactly match this MCP server.")
    if grant_type == "authorization_code":
        code = form.get("code", "")
        record = AUTH_CODES.get(code)
        if not record or float(record.get("exp", 0)) < time.time():
            AUTH_CODES.pop(code, None)
            return _error_response("invalid_grant", "Authorization code is invalid or expired.")
        if record["client_id"] != client_id or record["redirect_uri"] != form.get("redirect_uri") or record["resource"] != resource:
            return _error_response("invalid_grant", "Authorization code binding does not match.")
        verifier = form.get("code_verifier", "")
        if not re.fullmatch(r"[A-Za-z0-9._~-]{43,128}", verifier):
            return _error_response("invalid_grant", "A valid PKCE verifier is required.")
        if not hmac.compare_digest(_b64url(hashlib.sha256(verifier.encode("ascii")).digest()), record["code_challenge"]):
            return _error_response("invalid_grant", "PKCE verification failed.")
        AUTH_CODES.pop(code, None)
        return JSONResponse(_token_response(client_id, record["scope"], resource), headers={"Cache-Control": "no-store", "Pragma": "no-cache"})
    if grant_type == "refresh_token":
        raw_refresh = form.get("refresh_token", "")
        try:
            claims = jwt.decode(raw_refresh, _secret(), algorithms=["HS256"], issuer=ISSUER, audience=RESOURCE,
                                options={"require": ["exp", "iat", "iss", "sub", "aud", "scope", "jti", "token_use", "client_id", "resource"]})
        except (jwt.PyJWTError, RuntimeError):
            return _error_response("invalid_grant", "Refresh token is invalid or expired.")
        old_jti = str(claims.get("jti", ""))
        if claims.get("token_use") != "refresh" or claims.get("client_id") != client_id or claims.get("resource") != resource:
            return _error_response("invalid_grant", "Refresh token client or resource binding mismatch.")
        if old_jti in USED_REFRESH_JTIS:
            return _error_response("invalid_grant", "Refresh token has already been rotated.")
        USED_REFRESH_JTIS.add(old_jti)
        return JSONResponse(_token_response(client_id, _clean_scope(str(claims["scope"])), resource), headers={"Cache-Control": "no-store", "Pragma": "no-cache"})
    return _error_response("unsupported_grant_type", "Only authorization_code and refresh_token are supported.")

def handle_rpc(msg: dict[str, Any], scopes: set[str]):
    method, ident = msg.get("method"), msg.get("id")
    if method == "notifications/initialized":
        return None
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": ident, "result": {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}, "serverInfo": {"name": "duamorpaper-button-renderer", "version": "1.2.0"}}}
    if method == "ping":
        return {"jsonrpc": "2.0", "id": ident, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": ident, "result": {"tools": TOOLS}}
    if method == "tools/call":
        params = msg.get("params") or {}
        name, args = params.get("name"), params.get("arguments") or {}
        needed = PROFILE_SCOPE if name == "get_gabarito_profile" else RENDER_SCOPE if name == "render_button_from_layers" else None
        if not needed:
            return {"jsonrpc": "2.0", "id": ident, "error": {"code": -32601, "message": "Unknown tool"}}
        if needed not in scopes:
            challenge = f'Bearer resource_metadata="{RESOURCE_METADATA_URL}", error="insufficient_scope", error_description="Missing required scope"'
            return {"jsonrpc": "2.0", "id": ident, "result": {"content": [{"type": "text", "text": "OAuth token is missing the required permission."}], "_meta": {"mcp/www_authenticate": [challenge]}, "isError": True}}
        try:
            if name == "get_gabarito_profile":
                size = args.get("size_mm")
                if isinstance(size, bool) or size not in PROFILES:
                    raise ValueError("size_mm deve ser 32, 44 ou 58")
                p = PROFILES[size]
                result = {"size_mm": size, "visible_mm": p["visible"], "fold_mm": p["fold"], "total_cut_mm": p["total"], "rule": "Texto e conteúdo principal dentro da área visível; decoração secundária somente até a dobra; dobra ao corte apenas fundo contínuo."}
                return {"jsonrpc": "2.0", "id": ident, "result": {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]}}
            raw, summary = render_layers(args)
            return {"jsonrpc": "2.0", "id": ident, "result": {"content": [
                {"type": "text", "text": json.dumps(summary, ensure_ascii=False)},
                {"type": "image", "mimeType": "image/png", "data": base64.b64encode(raw).decode("ascii")},
            ]}}
        except Exception as exc:
            return {"jsonrpc": "2.0", "id": ident, "error": {"code": -32602, "message": str(exc)}}
    if ident is None:
        return None
    return {"jsonrpc": "2.0", "id": ident, "error": {"code": -32601, "message": "Method not found"}}

@app.get("/healthz")
async def healthz():
    return {"status": "ok", "service": "duamorpaper-button-renderer-mcp", "auth": "oauth2"}

@app.post("/mcp")
async def mcp_post(request: Request):
    origin = request.headers.get("origin")
    if origin and origin not in ALLOWED_ORIGINS:
        return JSONResponse({"error": "Origin not allowed"}, status_code=403)
    bearer = request.headers.get("authorization", "")
    if not bearer.lower().startswith("bearer "):
        return JSONResponse({"error": "Unauthorized"}, status_code=401, headers={"WWW-Authenticate": f'Bearer resource_metadata="{RESOURCE_METADATA_URL}", scope="{PROFILE_SCOPE} {RENDER_SCOPE}"'})
    token = bearer[7:].strip()
    try:
        claims = jwt.decode(token, _secret(), algorithms=["HS256"], issuer=ISSUER, audience=RESOURCE,
                            options={"require": ["exp", "iat", "iss", "sub", "aud", "scope", "jti", "token_use", "resource"]}, leeway=10)
        if claims.get("token_use") != "access" or claims.get("resource") != RESOURCE:
            raise jwt.InvalidTokenError("Wrong token type or resource")
        scopes = set(str(claims.get("scope", "")).split())
        if not scopes.issubset(SUPPORTED_SCOPES):
            raise jwt.InvalidTokenError("Unsupported scope")
    except (jwt.PyJWTError, RuntimeError) as exc:
        return JSONResponse({"error": "Invalid or expired OAuth access token", "detail": str(exc)}, status_code=401,
                            headers={"WWW-Authenticate": f'Bearer resource_metadata="{RESOURCE_METADATA_URL}", error="invalid_token"'})
    raw = await request.body()
    if len(raw) > MAX_BODY:
        return JSONResponse({"error": "Request body too large"}, status_code=413)
    try:
        msg = json.loads(raw)
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0" or not isinstance(msg.get("method"), str):
            raise ValueError
    except Exception:
        return JSONResponse({"error": "Invalid JSON-RPC request"}, status_code=400)
    response = handle_rpc(msg, scopes)
    if response is None:
        return Response(status_code=202)
    headers = {"Cache-Control": "no-store"}
    if msg.get("method") == "initialize":
        headers["MCP-Protocol-Version"] = response["result"]["protocolVersion"]
    return JSONResponse(response, headers=headers)

@app.get("/mcp")
async def mcp_get():
    return Response(status_code=405, headers={"Allow": "POST, OPTIONS"})
