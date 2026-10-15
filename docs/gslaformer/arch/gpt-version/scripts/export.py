"""用隔离 Chrome/CDP 导出矢量 PDF、PNG 并检查文本边界；不访问用户浏览器配置。"""
import base64
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
from urllib.request import urlopen
import xml.etree.ElementTree as ET

import websocket

HERE = Path(__file__).resolve().parents[1]
EVIDENCE = HERE / "evidence"
CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")


class Browser:
    def __init__(self, url):
        self.socket = websocket.create_connection(url, suppress_origin=True, timeout=30)
        self.sequence = 0

    def call(self, method, **params):
        self.sequence += 1
        self.socket.send(json.dumps({"id": self.sequence, "method": method, "params": params}))
        while True:
            response = json.loads(self.socket.recv())
            if response.get("id") == self.sequence:
                if "error" in response:
                    raise RuntimeError(response["error"])
                return response.get("result", {})

    def evaluate(self, expression):
        response = self.call("Runtime.evaluate", expression=expression,
                             returnByValue=True, awaitPromise=True)
        if "exceptionDetails" in response:
            raise RuntimeError(response["exceptionDetails"])
        return response["result"].get("value")

    def navigate(self, path):
        response = self.call("Page.navigate", url=path.as_uri())
        assert "errorText" not in response, response
        for _ in range(100):
            if self.evaluate("document.readyState") == "complete":
                break
            time.sleep(0.05)
        else:
            raise TimeoutError(path)
        self.evaluate("document.fonts.ready.then(()=>true)")

    def viewport(self, width, height, scale=1):
        self.call("Emulation.setDeviceMetricsOverride", width=width, height=height,
                  deviceScaleFactor=scale, mobile=False)

    def screenshot(self, path, width, height):
        response = self.call("Page.captureScreenshot", format="png", captureBeyondViewport=True,
                             clip={"x": 0, "y": 0, "width": width, "height": height, "scale": 1})
        path.write_bytes(base64.b64decode(response["data"]))


def line_crossings(svg):
    # 只解析本目录 render.py 生成的受控 SVG，不接收网络或用户上传 XML。
    root = ET.fromstring(svg)
    ns = {"s": "http://www.w3.org/2000/svg"}
    rectangles = []
    for group in root.findall("s:g", ns):
        if group.get("class") == "diagram-node":
            rect = group.find("s:rect", ns)
            text = group.find("s:text", ns)
            assert rect is not None and text is not None
            label = text.text
            rectangles.append((float(rect.attrib["x"]), float(rect.attrib["y"]),
                               float(rect.attrib["width"]), float(rect.attrib["height"]), label))
    problems = []
    for edge in root.findall("s:path", ns):
        if edge.get("class") not in {"edge", "bus"}:
            continue
        points = [(float(x), float(y)) for x, y in re.findall(r"([\d.]+),([\d.]+)", edge.attrib["d"])]
        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            for x, y, w, h, label in rectangles:
                vertical = x1 == x2 and x + .1 < x1 < x + w - .1 and max(min(y1,y2), y+.1) < min(max(y1,y2), y+h-.1)
                horizontal = y1 == y2 and y + .1 < y1 < y + h - .1 and max(min(x1,x2), x+.1) < min(max(x1,x2), x+w-.1)
                if vertical or horizontal:
                    problems.append({"edge": edge.get("d"), "node": label})
    return problems


def main():
    assert CHROME.is_file(), "Chrome not found; set CHROME to an available executable"
    scratch = Path(os.environ["TMPDIR"])
    profile = Path(tempfile.mkdtemp(prefix="gsla-arch-", dir=scratch))
    process = subprocess.Popen([str(CHROME), "--headless=new", "--no-first-run",
                                "--no-default-browser-check", "--disable-background-networking",
                                "--disable-extensions", "--remote-debugging-port=0",
                                f"--user-data-dir={profile}", "about:blank"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    browser = None
    try:
        for _ in range(200):
            port_file = profile / "DevToolsActivePort"
            if port_file.exists():
                port = int(port_file.read_text().splitlines()[0])
                break
            if process.poll() is not None:
                raise RuntimeError("Chrome exited before CDP started")
            time.sleep(.05)
        else:
            raise TimeoutError("Chrome CDP startup")
        with urlopen(f"http://127.0.0.1:{port}/json/list", timeout=10) as response:
            targets = json.load(response)
        browser = Browser(next(t["webSocketDebuggerUrl"] for t in targets if t["type"] == "page"))
        browser.call("Page.enable")
        report = {"timestamp": datetime.now().astimezone().isoformat(),
                  "browser": browser.call("Browser.getVersion"), "figures": {},
                  "isolation": "dedicated scratch profile; no personal browser login/cookie access"}
        for name in ("current-architecture", "mechanisms", "comparison"):
            path = HERE / f"{name}.svg"
            root = ET.fromstring(path.read_text())
            width, height = int(root.attrib["width"]), int(root.attrib["height"])
            browser.viewport(width, height, 2)
            browser.navigate(path)
            geometry = browser.evaluate('''(()=>{
                const svg=document.documentElement, view=svg.viewBox.baseVal;
                const outside=[]; const nodeOverflow=[];
                for(const t of svg.querySelectorAll('text')){
                    const b=t.getBBox();
                    if(b.x<0||b.y<0||b.x+b.width>view.width||b.y+b.height>view.height)outside.push(t.textContent);
                    if(t.parentElement.classList.contains('diagram-node')){
                        const r=t.parentElement.querySelector('rect').getBBox();
                        if(b.x<r.x+3||b.y<r.y+2||b.x+b.width>r.x+r.width-3||b.y+b.height>r.y+r.height-2)
                            nodeOverflow.push({text:t.textContent,textBounds:{x:b.x,y:b.y,w:b.width,h:b.height},nodeBounds:{x:r.x,y:r.y,w:r.width,h:r.height}});
                    }
                }
                return {outside,nodeOverflow,textCount:svg.querySelectorAll('text').length};
            })()''')
            geometry["edgeThroughNode"] = line_crossings(path.read_text())
            browser.screenshot(HERE / f"{name}.png", width, height)
            pdf = browser.call("Page.printToPDF", printBackground=True, paperWidth=width/96,
                               paperHeight=height/96, marginTop=0, marginBottom=0,
                               marginLeft=0, marginRight=0, scale=1)
            payload = base64.b64decode(pdf["data"])
            (HERE / f"{name}.pdf").write_bytes(payload)
            geometry["pdfPages"] = len(re.findall(rb"/Type /Page\b", payload))
            geometry["svgSize"] = [width, height]
            report["figures"][name] = geometry
        browser.viewport(1440, 1000)
        browser.navigate(HERE / "index.html")
        browser.screenshot(EVIDENCE / "report-desktop.png", 1440, 1000)
        report["desktop"] = browser.evaluate('''(()=>{const ids=[...document.querySelectorAll('[id]')].map(n=>n.id);return {bodyWidth:document.documentElement.scrollWidth,viewport:innerWidth,duplicateIds:ids.filter((x,i)=>ids.indexOf(x)!==i),images:[...document.images].map(i=>({loaded:i.complete&&i.naturalWidth>0})),sections:document.querySelectorAll('main section').length}})()''')
        browser.viewport(390, 844)
        browser.screenshot(EVIDENCE / "report-mobile.png", 390, 844)
        report["mobile"] = browser.evaluate("({bodyWidth:document.documentElement.scrollWidth,viewport:innerWidth,overflow:[...document.querySelectorAll('main>header *,main>section>p,main>section>table')].filter(e=>e.getBoundingClientRect().right>innerWidth).map(e=>({tag:e.tagName,text:e.textContent.slice(0,80),right:e.getBoundingClientRect().right}))})")
        browser.call("Emulation.setScriptExecutionDisabled", value=True)
        browser.navigate(HERE / "index.html")
        # DOM inspection via CDP still works; page scripts are disabled.
        report["noJavascript"] = browser.evaluate("({headings:document.querySelectorAll('h2').length,textLength:document.body.innerText.length,svgCount:document.querySelectorAll('svg').length})")
        report["passed"] = all(not row["outside"] and not row["nodeOverflow"] and not row["edgeThroughNode"] and row["pdfPages"] == 1 for row in report["figures"].values()) and not report["desktop"]["duplicateIds"] and all(i["loaded"] for i in report["desktop"]["images"]) and report["desktop"]["bodyWidth"] <= report["desktop"]["viewport"] and report["mobile"]["bodyWidth"] <= report["mobile"]["viewport"] and report["noJavascript"]["svgCount"] == 3
        (EVIDENCE / "visual-check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report["passed"]:
            raise SystemExit(1)
    finally:
        if browser is not None:
            browser.socket.close()
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


if __name__ == "__main__":
    main()
