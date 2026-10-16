"""真实浏览器导出及精修图验收；只写当前新增目录。"""
from pathlib import Path
import base64
import hashlib
import json
import re
import subprocess
import tempfile
import time
import urllib.request
import xml.etree.ElementTree as ET

import websocket
from PIL import Image

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
SCRATCH = Path.home() / '.hermes/profiles/coder/cache/scratch'
CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'


class CDP:
    def __init__(self, url):
        self.socket = websocket.create_connection(url, timeout=40, suppress_origin=True)
        self.number = 0

    def call(self, method, **params):
        self.number += 1
        self.socket.send(json.dumps({'id':self.number,'method':method,'params':params}))
        while True:
            result = json.loads(self.socket.recv())
            if result.get('id') == self.number:
                if 'error' in result:
                    raise RuntimeError(result)
                return result.get('result', {})

    def evaluate(self, expression):
        result = self.call('Runtime.evaluate',expression=expression,returnByValue=True,awaitPromise=True)
        if 'exceptionDetails' in result:
            raise RuntimeError(result)
        return result['result'].get('value')


def intersects_box(a,b,n):
    x,y,w,h = (n[k] for k in ('x','y','w','h'))
    if a[0] == b[0]:
        return x+.1<a[0]<x+w-.1 and max(min(a[1],b[1]),y+.1)<min(max(a[1],b[1]),y+h-.1)
    return y+.1<a[1]<y+h-.1 and max(min(a[0],b[0]),x+.1)<min(max(a[0],b[0]),x+w-.1)


def main():
    data = json.loads((OUT/'layout.json').read_text())
    name,w,h = data['name'],data['width'],data['height']
    svg = (OUT/(name+'.svg')).read_text()
    page = f'<!doctype html><meta charset="utf-8"><title>GSLAformer 原图精修版</title><style>@page{{size:{w}px {h}px;margin:0}}html,body{{margin:0;background:white;width:{w}px;height:{h}px}}svg{{display:block}}</style>'+svg
    html_path=OUT/(name+'.html')
    html_path.write_text(page)
    profile=Path(tempfile.mkdtemp(prefix='gsla-refined-chrome-',dir=SCRATCH))
    log=(profile/'chrome.log').open('w')
    process=subprocess.Popen([CHROME,'--headless=new','--no-first-run','--no-default-browser-check',
                              '--disable-background-networking','--remote-debugging-port=0',
                              f'--user-data-dir={profile}','about:blank'],stdout=log,stderr=log)
    client=None
    try:
        port_file=profile/'DevToolsActivePort'
        for _ in range(100):
            if port_file.exists():
                break
            if process.poll() is not None:
                raise RuntimeError(f'Chrome exited; log: {profile}')
            time.sleep(.1)
        port=int(port_file.read_text().splitlines()[0])
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/json/list') as response:
            targets=json.load(response)
        client=CDP(next(t['webSocketDebuggerUrl'] for t in targets if t['type']=='page'))
        client.call('Page.enable')
        client.call('Emulation.setDeviceMetricsOverride',width=w,height=h,deviceScaleFactor=2,mobile=False)
        client.call('Page.navigate',url=html_path.as_uri())
        for _ in range(100):
            if client.evaluate('document.readyState==="complete" && !!document.querySelector("svg")'):
                break
            time.sleep(.05)
        client.evaluate('document.fonts.ready.then(()=>true)')
        bounds=client.evaluate('Array.from(document.querySelectorAll("text[data-owner]")).map(e=>{const b=e.getBBox();return {id:e.dataset.owner,text:e.textContent,x:b.x,y:b.y,w:b.width,h:b.height}})')
        node_map={n['id']:n for n in data['nodes']}
        overflows=[]
        for t in bounds:
            n=node_map[t['id']]
            if t['x']<n['x']-1 or t['y']<n['y']-1 or t['x']+t['w']>n['x']+n['w']+1 or t['y']+t['h']>n['y']+n['h']+1:
                overflows.append(t)
        collisions=[]
        for edge in data['edges']:
            for a,b in zip(edge['points'],edge['points'][1:]):
                for n in data['nodes']:
                    if n['container'] or n['id'] in (edge['source'],edge['target']):
                        continue
                    if intersects_box(a,b,n):
                        collisions.append([edge['id'],n['id']])
        (OUT/(name+'.png')).write_bytes(base64.b64decode(client.call('Page.captureScreenshot',format='png',captureBeyondViewport=True)['data']))
        (OUT/(name+'.pdf')).write_bytes(base64.b64decode(client.call('Page.printToPDF',printBackground=True,preferCSSPageSize=True,displayHeaderFooter=False,marginTop=0,marginBottom=0,marginLeft=0,marginRight=0)['data']))
        # 以下 XML 均来自本任务自产或已经读取审阅的本地原图。
        original=ET.parse(BASE/'gslaformer_impl.drawio')
        refined=ET.parse(OUT/(name+'.drawio'))
        ET.parse(OUT/(name+'.svg'))
        old={c.get('id'):c for c in original.iter('mxCell')}
        new={c.get('id'):c for c in refined.iter('mxCell')}
        kept=set(old)<=set(new)
        geometry_same=0
        normalized_geometry_same=0
        for key, cell in old.items():
            if cell.get('vertex')!='1':
                continue
            old_geometry=cell.find('mxGeometry')
            new_geometry=new[key].find('mxGeometry')
            assert old_geometry is not None and new_geometry is not None
            geometry_same += ET.tostring(old_geometry)==ET.tostring(new_geometry)
            normalized_geometry_same += all(abs(float(old_geometry.get(k,'0'))-float(new_geometry.get(k,'0'))/2)<.001 for k in ('x','y','width','height'))
        required_modules={'c5':'GroupAdapter','c6':'GRUCell','c13':'GroupSpatialEncoder',
                          'c54':'RelationAttention','c60':'GroupReadout','c92':'StaticConditionEncoder',
                          'c177':'TemporalEncoder','c182':'FuturePathEncoder','c184':'ConditionFusion',
                          'c189':'load_head','c190':'latent_heads','history_scope':'EntityHistoryEncoder'}
        bilingual_modules=all(module in new[key].get('value','') and re.search(r'[\u4e00-\u9fff]',new[key].get('value','')) for key,module in required_modules.items())
        dashed=all('dashed=1;' in new[k].get('style','') for k in ('c3','c94'))
        baseline=json.loads((OUT/'preserved_files.json').read_text())
        unchanged=all(hashlib.sha256((BASE/p).read_bytes()).hexdigest()==value for p,value in baseline.items())
        pdf_pages=len(re.findall(rb'/Type\s*/Page\b',(OUT/(name+'.pdf')).read_bytes()))
        with Image.open(OUT/(name+'.png')) as img:
            png_size=list(img.size)
        report={'status':'passed' if not overflows and not collisions and kept and dashed and unchanged and pdf_pages==1 and bilingual_modules and png_size==[w*2,h*2] else 'failed',
                'text_overflows':overflows,'edges_through_nodes':collisions,'original_ids_retained':kept,
                'original_vertex_geometry_unchanged':geometry_same,'original_vertex_count':sum(c.get('vertex')=='1' for c in old.values()),
                'original_vertex_geometry_retained_before_2x_scaling':normalized_geometry_same,
                'bilingual_module_labels_verified':bool(bilingual_modules),'required_module_count':len(required_modules),
                'logical_canvas':[w,h],
                'left_right_frames_dashed':dashed,'preserved_files_count':len(baseline),'preserved_files_unchanged':unchanged,
                'png_size':png_size,'pdf_pages':pdf_pages,'renderer':'isolated local Chrome from same-layout SVG',
                'drawio_desktop_export':'not_run: unavailable','model_tests_and_training':'not_run'}
        (OUT/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print(json.dumps(report,ensure_ascii=False,indent=2))
        assert report['status']=='passed'
    finally:
        if client:
            client.socket.close()
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        log.close()


if __name__=='__main__':
    main()
