"""审计交付的链接、内嵌图、源码快照与产物一致性检查；不执行模型训练。"""
import base64
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit

from PIL import Image

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[3]


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.ids = set()
        self.embedded = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag == "a" and "href" in attrs:
            self.links.append(attrs["href"])
        if tag == "img":
            self.embedded.append(attrs["src"])


def check_link(source, link):
    parsed = urlsplit(link)
    if parsed.scheme or parsed.netloc:
        return
    target = (source.parent / unquote(parsed.path)).resolve() if parsed.path else source
    assert target.exists(), f"{source}: missing {link}"
    anchor = unquote(parsed.fragment)
    if not anchor:
        return
    text = target.read_text()
    if re.fullmatch(r"L\d+", anchor):
        assert 1 <= int(anchor[1:]) <= len(text.splitlines()), link
        return
    anchors = set(re.findall(r'(?:id|name)=["\']([^"\']+)["\']', text))
    for heading in re.findall(r"^#{1,6}\s+(.+)$", text, flags=re.M):
        anchors.add(re.sub(r"[^\w\s-]", "", heading.lower()).replace(" ", "-"))
    assert anchor in anchors, f"{source}: missing anchor {link}"


def main():
    runtime = json.loads((HERE / "evidence/runtime.json").read_text())
    visual = json.loads((HERE / "evidence/visual-check.json").read_text())
    assert runtime["tests"]["success"] and runtime["tests"]["skipped"] == 0
    assert all(runtime["checks"].values()) and visual["passed"]
    # 兼容迁移前原图的审计路径，历史回执本身保持原样。
    assert all(hashlib.sha256((ROOT / ("docs/gslaformer/arch/model-v4.0.png"
               if name == "docs/gslaformer/model-v4.0.png" else name)).read_bytes()).hexdigest() == value
               for name, value in runtime["source_sha256"].items())
    html = (HERE / "index.html").read_text()
    parser = Links()
    parser.feed(html)
    for link in parser.links:
        check_link(HERE / "index.html", link)
    # 不把同目录中其他会话新增的图稿子目录混入本次验收范围。
    markdown = list((ROOT / "docs/gslaformer/v1").glob("*.md")) + list(HERE.glob("*.md"))
    links_checked = len(parser.links)
    for path in markdown:
        text = path.read_text()
        assert text.count("```") % 2 == 0, path
        for link in re.findall(r"\[[^\]]*\]\(([^\s)]+)(?:\s+[^)]*)?\)", text):
            check_link(path, link)
            links_checked += 1
    original = (ROOT / "docs/gslaformer/arch/model-v4.0.png").read_bytes()
    assert len(parser.embedded) == 1
    assert base64.b64decode(parser.embedded[0].split(",", 1)[1]) == original
    assert not re.search(r'<(?:link|script)[^>]+(?:src|href)=["\']https?://', html)
    inline_svgs = re.findall(r"<svg\b.*?</svg>", html, flags=re.S)
    names = ["current-architecture", "mechanisms", "comparison"]
    assert len(inline_svgs) == len(names)
    for name, embedded in zip(names, inline_svgs):
        assert embedded == (HERE / f"{name}.svg").read_text().strip()
        expected_size = visual["figures"][name]["svgSize"]
        with Image.open(HERE / f"{name}.png") as image:
            assert image.size == tuple(value * 2 for value in expected_size)
            image.verify()
        pdf = (HERE / f"{name}.pdf").read_bytes()
        assert pdf.startswith(b"%PDF-") and len(re.findall(rb"/Type /Page\b", pdf)) == 1
    for script in (HERE / "scripts").glob("*.py"):
        compile(script.read_text(), str(script), "exec")
    subprocess.run(["git", "diff", "--check"], cwd=ROOT, check=True)
    artifacts = {}
    delivered = ["README.md", "comparison.json", "comparison.md", "index.html",
                 "scripts/probe.py", "scripts/render.py", "scripts/export.py", "scripts/verify.py",
                 "evidence/runtime.json", "evidence/targeted-tests.log",
                 "evidence/visual-check.json", "evidence/report-desktop.png",
                 "evidence/report-mobile.png"]
    delivered += [f"{name}.{extension}" for name in names for extension in ("svg", "png", "pdf")]
    for name in sorted(delivered):
        payload = (HERE / name).read_bytes()
        artifacts[name] = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    report = {"passed": True, "local_links_checked": links_checked,
              "markdown_files_checked": len(markdown), "figure_sets": len(names),
              "original_embedded_unchanged": True, "inline_svgs_match_sources": True,
              "runtime_source_hashes_match": True, "git_diff_check": "passed",
              "artifacts": artifacts}
    (HERE / "evidence/delivery-check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k:v for k,v in report.items() if k != "artifacts"}, ensure_ascii=False, indent=2))
    print(f"Artifact manifest entries: {len(artifacts)}")


if __name__ == "__main__":
    main()
