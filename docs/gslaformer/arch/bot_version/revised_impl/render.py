"""用隔离的本机 Chrome 渲染同源 SVG；不接触用户浏览器配置。"""
from pathlib import Path
import base64
import hashlib
import json

import subprocess
import tempfile
import time
import urllib.request
import xml.etree.ElementTree as ET

import websocket
from PIL import Image

OUT = Path(__file__).resolve().parent
CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
SCRATCH = Path.home() / '.hermes/profiles/coder/cache/scratch'


class CDP:
    def __init__(self, url):
        self.socket = websocket.create_connection(url, timeout=30, suppress_origin=True)
        self.number = 0

    def call(self, method, **params):
        self.number += 1
        self.socket.send(json.dumps({'id': self.number, 'method': method, 'params': params}))
        while True:
            value = json.loads(self.socket.recv())
            if value.get('id') == self.number:
                if 'error' in value:
                    raise RuntimeError(value)
                return value.get('result', {})

    def evaluate(self, expression):
        result = self.call('Runtime.evaluate', expression=expression, returnByValue=True, awaitPromise=True)
        if 'exceptionDetails' in result:
            raise RuntimeError(result)
        return result['result'].get('value')


def segment_enters(a, b, node):
    x0, y0, x1, y1 = node['x'], node['y'], node['x'] + node['w'], node['y'] + node['h']
    if a[0] == b[0]:
        return x0 < a[0] < x1 and max(min(a[1], b[1]), y0) < min(max(a[1], b[1]), y1)
    return y0 < a[1] < y1 and max(min(a[0], b[0]), x0) < min(max(a[0], b[0]), x1)


def main():
    layout = json.loads((OUT / 'layout.json').read_text())
    manifest = json.loads((OUT / 'manifest.json').read_text())
    for name, expected in manifest['original_assets_sha256'].items():
        assert hashlib.sha256((OUT.parent / name).read_bytes()).hexdigest() == expected
    profile = Path(tempfile.mkdtemp(prefix='gsla-diagram-chrome-', dir=SCRATCH))
    log = (profile / 'chrome.log').open('w')
    process = subprocess.Popen([CHROME, '--headless=new', '--no-first-run', '--no-default-browser-check',
                                '--disable-background-networking', '--remote-debugging-port=0',
                                f'--user-data-dir={profile}', 'about:blank'], stdout=log, stderr=log)
    pages = []
    report = {'renderer': 'isolated local headless Chrome / CDP', 'profile': str(profile), 'diagrams': []}
    try:
        port_file = profile / 'DevToolsActivePort'
        for _ in range(100):
            if port_file.exists():
                break
            if process.poll() is not None:
                raise RuntimeError(f'Chrome failed; inspect {profile / "chrome.log"}')
            time.sleep(.1)
        port = int(port_file.read_text().splitlines()[0])
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/json/list') as response:
            targets = json.load(response)
        client = CDP(next(t['webSocketDebuggerUrl'] for t in targets if t['type'] == 'page'))
        client.call('Page.enable')
        for item in layout:
            name, width, height = item['name'], item['width'], item['height']
            svg = (OUT / (name + '.svg')).read_text()
            page = '<!doctype html><meta charset="utf-8"><title>' + name + '</title>'
            page += f'<style>@page{{size:{width}px {height}px;margin:0}}html,body{{margin:0;padding:0;width:{width}px;height:{height}px;background:white}}svg{{display:block}}</style>' + svg
            html_path = OUT / (name + '.html')
            html_path.write_text(page)
            client.call('Emulation.setDeviceMetricsOverride', width=width, height=height, deviceScaleFactor=2, mobile=False)
            client.call('Page.navigate', url=html_path.as_uri())
            for _ in range(100):
                ready = client.evaluate('document.readyState === "complete" && !!document.querySelector("svg")')
                if ready:
                    break
                time.sleep(.05)
            client.evaluate('document.fonts.ready.then(() => true)')
            boxes = client.evaluate('Array.from(document.querySelectorAll("svg text")).map(e=>{const b=e.getBBox();return {text:e.textContent,x:b.x,y:b.y,w:b.width,h:b.height}})')
            overflows = []
            for text in boxes:
                owner = next((n for n in item['nodes'] if abs(n['x'] + 18 - text['x']) < 2 and n['y'] < text['y'] < n['y'] + n['h']), None)
                if owner and (text['x'] + text['w'] > owner['x'] + owner['w'] - 10 or text['y'] + text['h'] > owner['y'] + owner['h'] - 5):
                    overflows.append({'node': owner['id'], **text})
                if text['x'] + text['w'] > width or text['y'] + text['h'] > height:
                    overflows.append({'canvas': True, **text})
            crossings = []
            for edge in item['edges']:
                for a, b in zip(edge['points'], edge['points'][1:]):
                    for node in item['nodes']:
                        if node['id'] not in (edge['source'], edge['target']) and segment_enters(a, b, node):
                            crossings.append([edge['source'], edge['target'], node['id']])
            if overflows or crossings:
                raise ValueError(json.dumps({'diagram': name, 'text_overflow': overflows, 'edges_through_boxes': crossings}, ensure_ascii=False))
            image = client.call('Page.captureScreenshot', format='png', captureBeyondViewport=True)
            (OUT / (name + '.png')).write_bytes(base64.b64decode(image['data']))
            pdf = client.call('Page.printToPDF', printBackground=True, preferCSSPageSize=True, displayHeaderFooter=False, marginTop=0, marginBottom=0, marginLeft=0, marginRight=0)
            (OUT / (name + '.pdf')).write_bytes(base64.b64decode(pdf['data']))
            with Image.open(OUT / (name + '.png')) as img:
                assert img.size == (width * 2, height * 2)
            # 只解析本目录 generate.py 自产 XML，不接收外部不可信 XML。
            ET.parse(OUT / (name + '.drawio'))
            ET.parse(OUT / (name + '.svg'))
            report['diagrams'].append({'name': name, 'png_size': [width * 2, height * 2], 'text_overflow': overflows, 'edges_through_boxes': crossings, 'xml_valid': True})
            pages.append(svg)
            print('Rendered and checked', name, flush=True)
        report_path = OUT / 'index.html'
        report_path.write_text('<!doctype html><meta charset="utf-8"><title>GSLAformer 修订图册</title><style>body{margin:0;background:#e9eef4;font-family:sans-serif}nav{position:sticky;top:0;background:#203348;color:white;padding:16px}a{color:white;margin-right:24px}section{margin:24px;background:white}svg{display:block;width:100%;height:auto}</style><nav><a href="#overview">01 实体与空间</a><a href="#inference">02 预测与回灌</a><a href="#training">03 两阶段训练</a></nav>' + ''.join(f'<section id="{anchor}">{svg}</section>' for anchor, svg in zip(('overview','inference','training'), pages)))
        report['original_assets_unchanged'] = all(hashlib.sha256((OUT.parent / name).read_bytes()).hexdigest() == expected for name, expected in manifest['original_assets_sha256'].items())
        (OUT / 'validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
        client.socket.close()
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        log.close()


if __name__ == '__main__':
    main()
