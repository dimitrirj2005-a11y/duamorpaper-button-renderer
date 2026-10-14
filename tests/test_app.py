import base64
import hashlib
import io
from urllib.parse import parse_qs, urlparse

import jwt
from fastapi.testclient import TestClient
from PIL import Image

import app as service

client = TestClient(service.app)
CLIENT_ID = "https://chatgpt.com/oauth/client.json"
REDIRECT = "https://chatgpt.com/connector_platform_oauth_redirect"
RESOURCE = service.RESOURCE
CLIENT_METADATA = {
    "client_id": CLIENT_ID,
    "client_name": "ChatGPT",
    "redirect_uris": [REDIRECT],
    "token_endpoint_auth_method": "none",
    "token_endpoint_auth_methods_supported": ["none", "private_key_jwt"],
}

def png_b64(color=(80, 50, 130, 255), size=(128, 128)):
    im = Image.new("RGBA", size, color)
    stream = io.BytesIO()
    im.save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode("ascii")

def oauth_headers(monkeypatch):
    monkeypatch.setenv("OAUTH_SIGNING_SECRET", "test-secret-that-is-long-enough-for-hs256")
    monkeypatch.setenv("OAUTH_LOGIN_USERNAME", "duamorpaper")
    monkeypatch.setenv("OAUTH_LOGIN_PASSWORD", "test-login-password")
    monkeypatch.setenv("PUBLIC_BASE_URL", RESOURCE)

def mock_cimd(monkeypatch):
    async def get_client(client_id):
        assert client_id == CLIENT_ID or client_id.startswith("dcr.")
        if client_id.startswith("dcr."):
            return service._decode_dcr_client_id(client_id)
        return CLIENT_METADATA.copy()
    monkeypatch.setattr(service, "_get_client_metadata", get_client)

def test_profiles_are_official():
    assert service.PROFILES == {
        32: {"visible": 32.0, "fold": 41.0, "total": 48.0},
        44: {"visible": 44.0, "fold": 53.0, "total": 60.0},
        58: {"visible": 58.0, "fold": 67.0, "total": 74.0},
    }

def test_output_has_correct_dpi_and_transparent_outside():
    result, summary = service.render_layers({
        "size_mm": 32,
        "background_png_base64": png_b64((50, 30, 90, 255)),
        "subject_png_base64": png_b64((0, 0, 0, 0)),
        "exact_text": "SEM TEXTO",
    })
    im = Image.open(io.BytesIO(result))
    assert im.format == "PNG"
    assert round(im.info["dpi"][0]) == 300
    assert im.size == (int(round(48 / 25.4 * 300)),) * 2
    assert im.getpixel((0, 0))[3] == 0
    assert summary["geometry_masks_applied"] is True

def test_healthcheck_and_oauth_discovery():
    health = client.get("/healthz")
    assert health.status_code == 200 and health.json()["auth"] == "oauth2"
    protected = client.get("/.well-known/oauth-protected-resource").json()
    assert protected["resource"] == RESOURCE
    assert protected["authorization_servers"] == [service.ISSUER]
    metadata = client.get("/.well-known/oauth-authorization-server").json()
    assert metadata["issuer"] == service.ISSUER
    assert metadata["code_challenge_methods_supported"] == ["S256"]
    assert metadata["client_id_metadata_document_supported"] is True
    assert metadata["authorization_response_iss_parameter_supported"] is True
    assert "private_key_jwt" in metadata["token_endpoint_auth_methods_supported"]

def test_mcp_requires_oauth_not_old_api_key(monkeypatch):
    oauth_headers(monkeypatch)
    body = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}}
    r = client.post("/mcp", json=body, headers={"Authorization": "Bearer legacy-api-key"})
    assert r.status_code == 401
    assert "oauth-protected-resource" in r.headers["www-authenticate"]

def test_dcr_registration_and_signed_client_id(monkeypatch):
    oauth_headers(monkeypatch)
    body = {
        "client_name": "Test ChatGPT",
        "redirect_uris": [REDIRECT],
        "token_endpoint_auth_method": "none",
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
    }
    r = client.post("/register", json=body)
    assert r.status_code == 201
    dcr_id = r.json()["client_id"]
    decoded = service._decode_dcr_client_id(dcr_id)
    assert decoded["client_name"] == "Test ChatGPT"
    assert decoded["redirect_uris"] == [REDIRECT]

def test_authorization_code_pkce_and_refresh(monkeypatch):
    oauth_headers(monkeypatch)
    mock_cimd(monkeypatch)
    verifier = "a" * 43
    challenge = service._b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT,
        "scope": "buttons:profile buttons:render",
        "state": "test-state",
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "resource": RESOURCE,
    }
    auth = client.get("/authorize", params=params)
    assert auth.status_code == 200
    assert "Autorizar acesso" in auth.text
    denied = client.post("/authorize/decision", data={**params, "username":"duamorpaper", "password":"wrong", "decision":"approve"})
    assert denied.status_code == 200 and "incorretos" in denied.text
    grant = client.post("/authorize/decision", data={**params, "username":"duamorpaper", "password":"test-login-password", "decision":"approve"}, follow_redirects=False)
    assert grant.status_code == 302
    callback = parse_qs(urlparse(grant.headers["location"]).query)
    assert callback["state"] == ["test-state"]
    assert callback["iss"] == [service.ISSUER]
    code = callback["code"][0]
    exchange = client.post("/token", data={
        "grant_type":"authorization_code", "client_id":CLIENT_ID, "redirect_uri":REDIRECT,
        "code":code, "code_verifier":verifier, "resource":RESOURCE,
    })
    assert exchange.status_code == 200, exchange.text
    tokens = exchange.json()
    assert tokens["token_type"] == "Bearer"
    assert tokens["scope"] == "buttons:profile buttons:render"
    access = jwt.decode(tokens["access_token"], "test-secret-that-is-long-enough-for-hs256", algorithms=["HS256"], audience=RESOURCE, issuer=service.ISSUER)
    assert access["token_use"] == "access" and access["resource"] == RESOURCE
    mcp = client.post("/mcp", json={"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}},
                      headers={"Authorization":"Bearer " + tokens["access_token"]})
    assert mcp.status_code == 200 and len(mcp.json()["result"]["tools"]) == 2
    refresh = client.post("/token", data={
        "grant_type":"refresh_token", "client_id":CLIENT_ID, "refresh_token":tokens["refresh_token"],
        "resource":RESOURCE,
    })
    assert refresh.status_code == 200, refresh.text
    replay = client.post("/token", data={
        "grant_type":"refresh_token", "client_id":CLIENT_ID, "refresh_token":tokens["refresh_token"],
        "resource":RESOURCE,
    })
    assert replay.status_code == 400 and replay.json()["error"] == "invalid_grant"

def test_bad_pkce_cannot_exchange_code(monkeypatch):
    oauth_headers(monkeypatch)
    mock_cimd(monkeypatch)
    verifier = "b" * 43
    challenge = service._b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    params = {
        "response_type":"code", "client_id":CLIENT_ID, "redirect_uri":REDIRECT,
        "scope":"buttons:profile buttons:render", "state":"s", "code_challenge":challenge,
        "code_challenge_method":"S256", "resource":RESOURCE,
    }
    r = client.post("/authorize/decision", data={**params, "username":"duamorpaper", "password":"test-login-password", "decision":"approve"}, follow_redirects=False)
    code = parse_qs(urlparse(r.headers["location"]).query)["code"][0]
    bad = client.post("/token", data={
        "grant_type":"authorization_code", "client_id":CLIENT_ID, "redirect_uri":REDIRECT,
        "code":code, "code_verifier":"c"*43, "resource":RESOURCE,
    })
    assert bad.status_code == 400 and bad.json()["error"] == "invalid_grant"
