import base64
import io
import os
from fastapi.testclient import TestClient
from PIL import Image
from app import app, PROFILES, render_layers

client = TestClient(app)

def png_b64(color=(80, 50, 130, 255), size=(128, 128)):
    im = Image.new('RGBA', size, color)
    stream = io.BytesIO()
    im.save(stream, format='PNG')
    return base64.b64encode(stream.getvalue()).decode('ascii')

def test_profiles_are_official():
    assert PROFILES == {
        32: {'visible':32.0,'fold':41.0,'total':48.0},
        44: {'visible':44.0,'fold':53.0,'total':60.0},
        58: {'visible':58.0,'fold':67.0,'total':74.0},
    }

def test_output_has_correct_dpi_and_transparent_outside():
    result, summary = render_layers({
        'size_mm':32,
        'background_png_base64':png_b64((50,30,90,255)),
        'subject_png_base64':png_b64((0,0,0,0)),
        'exact_text':'SEM TEXTO'
    })
    im = Image.open(io.BytesIO(result))
    assert im.format == 'PNG'
    assert im.info['dpi'][0] == 300
    assert im.size == (int(round(48/25.4*300)),)*2
    assert im.getpixel((0,0))[3] == 0
    assert summary['geometry_masks_applied'] is True

def test_healthcheck_does_not_require_secret():
    r = client.get('/healthz')
    assert r.status_code == 200 and r.json()['status'] == 'ok'

def test_http_endpoint_requires_auth(monkeypatch):
    monkeypatch.setenv('MCP_API_KEY','unit-secret')
    body = {'jsonrpc':'2.0','id':1,'method':'initialize','params':{'protocolVersion':'2024-11-05'}}
    assert client.post('/mcp',json=body).status_code == 401
    r = client.post('/mcp',json=body,headers={'Authorization':'Bearer unit-secret'})
    assert r.status_code == 200
    assert r.json()['result']['serverInfo']['name'] == 'duamorpaper-button-renderer'

def test_tools_list_and_profile(monkeypatch):
    monkeypatch.setenv('MCP_API_KEY','unit-secret')
    h={'Authorization':'Bearer unit-secret'}
    body={'jsonrpc':'2.0','id':3,'method':'tools/list','params':{}}
    r=client.post('/mcp',json=body,headers=h)
    assert {x['name'] for x in r.json()['result']['tools']} == {'get_gabarito_profile','render_button_from_layers'}
    body={'jsonrpc':'2.0','id':4,'method':'tools/call','params':{'name':'get_gabarito_profile','arguments':{'size_mm':44}}}
    r=client.post('/mcp',json=body,headers=h)
    assert r.status_code==200 and '53.0' in r.text
