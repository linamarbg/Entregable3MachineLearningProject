"""Capturas del dashboard para el capítulo 7 del libro (book/figuras/).

Abre Chrome sin interfaz con el protocolo DevTools, recorre las tres páginas, hace clic en las pestañas de
Modelos (incluida una predicción en vivo con un corredor real) y guarda nueve capturas en JPG. El mapa usa
WebGL, por eso Chrome se lanza con SwiftShader (WebGL por software).

Requisitos: el dashboard corriendo (python app.py) y `websocket-client`.
Uso: python scripts/27_capturas_dashboard.py [url_base] [carpeta_salida]
     por defecto http://127.0.0.1:8050 y book/figuras
"""
import base64
import io
import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import websocket
from PIL import Image

RAIZ = Path(__file__).resolve().parents[1]
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8050").rstrip("/")
SALIDA = Path(sys.argv[2]) if len(sys.argv) > 2 else RAIZ / "book" / "figuras"
PUERTO = 9333


class Pestana:
    """Cliente mínimo del protocolo DevTools para una pestaña."""

    def __init__(self, ws_url):
        self.ws = websocket.create_connection(ws_url, timeout=120)
        self.n = 0

    def cmd(self, metodo, **params):
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": metodo, "params": params}))
        while True:
            r = json.loads(self.ws.recv())
            if r.get("id") == self.n:
                if "error" in r:
                    raise RuntimeError(r["error"])
                return r.get("result", {})

    def js(self, expr):
        r = self.cmd("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        return r.get("result", {}).get("value")

    def tamano(self, ancho, alto, movil=False):
        self.cmd("Emulation.setDeviceMetricsOverride", width=ancho, height=alto, deviceScaleFactor=1, mobile=movil)

    def ir(self, url, espera=6):
        self.cmd("Page.navigate", url=url)
        time.sleep(espera)

    def clic_texto(self, texto, selector="a, button", espera=4):
        ok = self.js(f"""(() => {{ const el = [...document.querySelectorAll('{selector}')]
                                   .find(e => e.innerText.trim() === {json.dumps(texto)});
                          if (!el) return false; el.click(); return true; }})()""")
        if not ok:
            raise RuntimeError(f"no encontré «{texto}»")
        time.sleep(espera)

    def bajar_a(self, texto, desfase=80):
        self.js(f"""(() => {{ const el = [...document.querySelectorAll('.card-header, h2, h3')]
                                   .find(e => e.innerText.includes({json.dumps(texto)}));
                    if (el) window.scrollTo(0, el.getBoundingClientRect().top + window.scrollY - {desfase}); }})()""")
        time.sleep(1.5)

    def captura(self, nombre, ancho, alto, calidad=85):
        datos = self.cmd("Page.captureScreenshot", format="png", captureBeyondViewport=False,
                         clip={"x": 0, "y": self.js("window.scrollY"), "width": ancho, "height": alto, "scale": 1})["data"]
        im = Image.open(io.BytesIO(base64.b64decode(datos))).convert("RGB")
        im.save(SALIDA / f"{nombre}.jpg", quality=calidad, optimize=True)
        print("ok", nombre, im.size)


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    perfil = tempfile.mkdtemp(prefix="chrome-capturas-")
    proc = subprocess.Popen([CHROME, "--headless=new", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                             "--ignore-gpu-blocklist", "--no-first-run", "--hide-scrollbars", f"--user-data-dir={perfil}",
                             f"--remote-debugging-port={PUERTO}", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        lista = []
        for _ in range(40):
            try:
                lista = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PUERTO}/json").read())
                break
            except OSError:
                time.sleep(0.5)
        p = Pestana([t for t in lista if t["type"] == "page"][0]["webSocketDebuggerUrl"])
        p.cmd("Page.enable")
        W, H = 1440, 900
        p.tamano(W, H)
        # Contexto: portada y mapa de los corredores
        p.ir(BASE + "/", 9)
        p.captura("dash_contexto", W, H)
        p.bajar_a("El mapa de los corredores")
        time.sleep(4)
        p.captura("dash_mapa", W, H)
        # EDA
        p.ir(BASE + "/eda", 9)
        p.captura("dash_eda", W, H)
        # Modelos: líderes, predicción en vivo, diagnóstico, comparación y modelo original
        p.ir(BASE + "/modelos", 8)
        p.bajar_a("Mejor combinación", 160)
        p.captura("dash_lideres", W, H)
        p.js("window.scrollTo(0, 0)")
        p.clic_texto("Predicción en vivo")
        p.js("[...document.querySelectorAll('button')].find(b => b.innerText.includes('Cargar')).click()")
        time.sleep(3)
        p.js("document.getElementById('b-predecir').click()")
        time.sleep(10)
        p.bajar_a("Datos del corredor", 150)
        p.captura("dash_prediccion", W, H)
        p.js("window.scrollTo(0, 0)")
        p.clic_texto("Diagnóstico", espera=8)
        p.bajar_a("Curvas ROC", 150)
        p.captura("dash_diagnostico", W, H)
        p.js("window.scrollTo(0, 0)")
        p.clic_texto("Comparación estadística", espera=6)
        p.js("window.scrollTo(0, 330)")
        time.sleep(1)
        p.captura("dash_comparacion", W, H)
        p.js("window.scrollTo(0, 0)")
        p.clic_texto("Modelo original (AJ-KL)", espera=6)
        p.js("window.scrollTo(0, 330)")
        time.sleep(1)
        p.captura("dash_original", W, H)
        # tableta (768 × 1024)
        p.tamano(768, 1024, movil=True)
        p.ir(BASE + "/eda", 9)
        p.captura("dash_tableta", 768, 1024)
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
