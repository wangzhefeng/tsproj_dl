"""GSLAformer 审计图的确定性 SVG/HTML 生成器；语义来自人工核实的 comparison.json。"""
import base64
from collections import Counter
import hashlib
from html import escape
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[3]
COLORS = {"ink": "#141413", "muted": "#5E5E57", "line": "#87867F", "border": "#D1CFC5",
          "paper": "#FAF9F5", "white": "#FFFFFF", "neutral": "#F0EEE6", "oat": "#E3DACC",
          "added": "#EFF3E9", "green": "#788C5D", "changed": "#FBEDE7", "clay": "#D97757"}


class Figure:
    def __init__(self, width, height, title, subtitle):
        self.width, self.height = width, height
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
            '<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M1 1 L9 5 L1 9 Z" fill="#87867F"/></marker></defs>',
            '<style>text{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;fill:#141413}.title{font-size:36px;font-weight:650}.sub{font-size:18px;fill:#5E5E57}.panel-title{font-size:23px;font-weight:600}.node-title{font-size:21px;font-weight:600}.node-line{font-size:18px;fill:#3D3D3A}.small{font-size:17px;fill:#5E5E57}.mono{font-family:Menlo,Consolas,"PingFang SC",monospace;font-size:18px}.edge{stroke:#87867F;stroke-width:2;fill:none;marker-end:url(#arr);stroke-linejoin:round}.bus{stroke:#87867F;stroke-width:2;fill:none}</style>',
            f'<rect width="{width}" height="{height}" fill="#FAF9F5"/>']
        self.text(42, 58, title, "title")
        self.text(44, 96, subtitle, "sub")

    def text(self, x, y, text, cls="node-line", anchor="start"):
        self.parts.append(f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{escape(text)}</text>')

    def panel(self, x, y, w, h, title, subtitle=""):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="#FFFFFF" stroke="#D1CFC5" stroke-width="1.5"/>')
        self.text(x + 22, y + 34, title, "panel-title")
        if subtitle:
            self.text(x + 22, y + 63, subtitle, "small")

    def box(self, x, y, w, h, title, lines=(), kind="neutral"):
        fill = COLORS.get(kind, COLORS["neutral"])
        stroke = COLORS["green"] if kind == "added" else COLORS["clay"] if kind == "changed" else COLORS["border"]
        self.parts.append(f'<g class="diagram-node"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        self.text(x + w / 2, y + 29, title, "node-title", "middle")
        for i, line in enumerate(lines):
            self.text(x + w / 2, y + 56 + 25 * i, line, "node-line", "middle")
        self.parts.append('</g>')

    def edge(self, points, arrow=True, dashed=False):
        d = 'M' + ' L'.join(f'{x},{y}' for x, y in points)
        dash = ' stroke-dasharray="7 5"' if dashed else ''
        self.parts.append(f'<path class="{"edge" if arrow else "bus"}" d="{d}"{dash}/>')

    def tokens(self, x, y, n, w=30, h=24):
        for i in range(n):
            fill = ("#E3DACC", "#EFF3E9", "#FBEDE7")[i % 3]
            self.parts.append(f'<rect x="{x+i*(w+4)}" y="{y}" width="{w}" height="{h}" rx="2" fill="{fill}" stroke="#87867F"/>')

    def finish(self):
        marker = f'arr-{self.width}-{self.height}'
        return '\n'.join(self.parts + ['</svg>']).replace('id="arr"', f'id="{marker}"').replace('url(#arr)', f'url(#{marker})')


def architecture():
    f = Figure(1800, 1370, "GSLAformer · 当前实现架构", "full 变体 / AIDC day profile · 工作区计算版本 2 · 按真实源码绘制，不是原始需求的复刻")
    f.panel(40, 140, 1720, 490, "A  实体历史 → 空间交互 → 分组状态", "每个请求只编码真实历史一次；不同组独立参数，同组实体与时刻共享参数")
    groups = [("算力", "compute"), ("IT 电力", "it_power"), ("配电", "distribution"),
              ("暖通", "hvac"), ("空间", "space · 机房 / 楼宇"), ("环境", "environment")]
    centers = []
    for i, (label, code) in enumerate(groups):
        x = 80 + 280 * i
        f.box(x, 220, 245, 72, label, [code], "changed")
        centers.append(x + 122.5)
        f.edge([(x + 122.5, 292), (x + 122.5, 322)], arrow=False)
    f.edge([(centers[0], 322), (centers[-1], 322)], arrow=False)
    f.edge([(260, 322), (260, 365)])
    f.box(85, 365, 350, 116, "实体属性适配 + 因果 GRU", ["值 / mask / 静态属性 → 32 维", "每实体沿 L 步积累历史"], "added")
    f.box(495, 365, 350, 116, "组内空间 Transformer", ["摘要 token + 实体 tokens", "1 层 · 4 heads · Post-LN"])
    f.box(905, 365, 350, 116, "类型关系注意力", ["实体级跨组消息 · 2 层", "类型 / 方向 bias + 合法边 mask"], "added")
    f.box(1315, 365, 400, 116, "每动态组查询池化", ["query attention + 数量 / 覆盖率", "Linear → LayerNorm → 32 维"], "changed")
    for left, right in [(435, 495), (845, 905), (1255, 1315)]:
        f.edge([(left, 423), (right, 423)])
    f.edge([(1515, 481), (1515, 525), (900, 525), (900, 550)])
    f.box(480, 550, 840, 65, "规范状态 z[B, L, 6, 32]", ["动态组横向拼接 · 时间纵向堆叠 · 每时刻 192 维"])
    f.text(85, 550, "不加实体行号位置编码", "small")
    f.text(85, 579, "ID 只用于对齐，不做可学习查表", "small")
    f.panel(40, 650, 1720, 660, "B  系统时间建模 → 条件融合 → 双头自由递推", "空间状态与总负荷共同回灌；输出状态不反解实体、不重新运行实体 GRU")
    f.edge([(480, 582), (450, 582), (450, 638), (55, 638), (55, 725), (275, 725), (275, 735)])
    f.box(100, 735, 350, 135, "长度固定的系统状态窗口", ["z + y + 质量 / 来源标记", "+ 四维日历 → token[B,L,211]", "L=168；生成态与观测态分开"], "changed")
    f.box(550, 735, 330, 135, "因果时间 Transformer", ["Linear(211→64) + 正弦位置", "2 层 · 4 heads · Post-LN", "仅取最后位置 h[B,64]"])
    f.edge([(450, 802), (550, 802)])
    f.box(970, 735, 360, 135, "逐步条件融合", ["拼接 h、静态 c、未来 pₖ", "+ 原起点 lead(k)：16 维", "MLP(208→64→64)"], "added")
    f.edge([(880, 802), (970, 802)])
    f.box(1440, 715, 260, 70, "总负荷头", ["Linear(64→1) → ŷₖ"])
    f.box(1440, 815, 260, 70, "分组状态头 ×6", ["Linear(64→32) → ẑₖ"])
    f.edge([(1330, 802), (1380, 802), (1380, 750), (1440, 750)])
    f.edge([(1380, 802), (1380, 850), (1440, 850)])
    f.edge([(1700, 750), (1730, 750), (1730, 910), (1570, 910), (1570, 930)])
    f.edge([(1700, 850), (1730, 850)], arrow=False)
    f.box(1440, 930, 260, 75, "同一步 ŷₖ + ẑₖ", ["存入结果，并回填状态窗口"], "changed")
    f.box(700, 1040, 390, 110, "起点静态条件 c[B,64]", ["属性统计 / 类别 / 关系度数", "逐组投影 → 拼接 → 单层 Linear"], "added")
    f.box(1190, 1040, 500, 110, "完整已知未来路径 p[B,24,64]", ["日历 + 计划 / 天气值 + mask + 位置", "1 层双向 Transformer；原起点固定"], "added")
    f.edge([(895, 1040), (895, 970), (1080, 970), (1080, 870)])
    f.edge([(1440, 1040), (1440, 1018), (1220, 1018), (1220, 870)])
    f.edge([(1700, 968), (1745, 968), (1745, 1240), (65, 1240), (65, 802), (100, 802)])
    f.text(115, 1188, "回灌规则：丢弃最早 token；追加生成 z / y；observed=false，generated=true，coverage=0", "node-line")
    f.text(115, 1220, "重复 H=24 步；真实历史编码一次，未来路径编码一次，时间窗口完整重算 H 次；无 KV cache", "small")
    f.text(80, 1345, "浅灰：保留主干 / 具体算子     浅绿：原图之外新增     浅橙：语义替代或扩展     输出：y[B,24,1] 与 z[B,24,6,32]", "small")
    return f.finish()


def mechanisms():
    f = Figure(1800, 1260, "GSLAformer · 关键机制展开", "区分三种注意力的可见范围；规范状态 z、时间 hidden h 与条件融合结果 a 不是同一个坐标")
    f.panel(40, 140, 540, 880, "A  残差与归一化骨架", "组内 / 时间 / 关系块均为 Post-LN")
    x, w, mid = 95, 360, 275
    f.box(x, 235, w, 65, "输入 tokens", ["空间宽度 32 / 时间宽度 64"])
    f.box(x, 370, w, 70, "Multi-Head Attention", ["注意力可见域由各分支决定"])
    f.box(x, 500, w, 55, "Add & LayerNorm")
    f.box(x, 625, w, 95, "Feed Forward", ["Linear(d→4d) → GELU", "→ Dropout → Linear(4d→d)"])
    f.box(x, 780, w, 55, "Add & LayerNorm")
    f.box(x, 900, w, 65, "输出 tokens", ["不是直接得到一个组向量"])
    for a, b in [(300, 370), (440, 500), (555, 625), (720, 780), (835, 900)]:
        f.edge([(mid, a), (mid, b)])
    f.edge([(mid, 332), (515, 332), (515, 527), (455, 527)])
    f.edge([(mid, 587), (515, 587), (515, 807), (455, 807)])
    f.text(75, 995, "残差支路为原输入；attention / FFN 输出另有 dropout", "small")
    f.panel(630, 140, 540, 880, "B  空间信息如何变成状态", "组内全连接与关系受限通信是两步，不是同一层")
    f.box(690, 235, 420, 95, "组内注意力 + 摘要 token", ["同组有效实体可相互读取", "摘要参与本层计算，之后剥离"])
    f.box(690, 395, 420, 120, "实体拼接 → 关系注意力 ×2", ["Q 行=接收方；K/V 列=发送方", "self + 正向边 + 声明的反向边", "每类型 / 方向 / head 可学习 bias"], "added")
    f.box(690, 580, 420, 95, "按组切回增强实体", ["无效实体输出置零", "仍保留实体轴 N_g，不提前压成组向量"])
    f.box(690, 740, 420, 120, "分组 query cross-attention", ["Q：每组可学习 query", "K/V：空组 token + 增强实体", "各动态组独立查询与参数"], "changed")
    f.box(690, 915, 420, 75, "规范状态 zᵍ", ["拼数量 / 覆盖率 → Linear → LayerNorm"])
    for a, b in [(330, 395), (515, 580), (675, 740), (860, 915)]:
        f.edge([(900, a), (900, b)])
    f.panel(1220, 140, 540, 880, "C  时间与预测坐标", "这里展示实际计算，不按过时文字补不存在的层")
    f.box(1280, 235, 420, 120, "系统历史 token", ["z：192；y：1；总表标记：2", "组 coverage / generated：12", "日历：4；合计 211 维"], "changed")
    f.box(1280, 415, 420, 95, "联合 Linear(211→64) + 位置", ["因果时间 attention ×2", "h = 最后 token；不另设日历投影"])
    f.box(1280, 570, 420, 120, "拼 h / c / pₖ / lead(k)", ["64 + 64 + 64 + 16 = 208", "Linear → GELU → Dropout → Linear", "a[B,64]：双头共享的融合结果"], "added")
    f.box(1280, 750, 420, 95, "两个并列预测目标", ["ŷₖ：1 维；ẑₖ：6×32 维", "状态头末端无额外 LayerNorm"])
    f.box(1280, 905, 420, 95, "回灌的是规范 ẑₖ 和同一 ŷₖ", ["不是 h，不是原始实体特征", "不更新实体 memory / 静态 c"], "changed")
    for a, b in [(355, 415), (510, 570), (690, 750), (845, 905)]:
        f.edge([(1490, a), (1490, b)])
    f.panel(40, 1050, 1720, 170, "D  训练与推理分开", "原图没有规定训练协议；以下是当前实现的补充，不是原图中的隐含要求")
    f.box(70, 1126, 450, 70, "阶段 A：H 步直接负荷监督", ["表示链可训练；latent 头不进入 loss"])
    f.box(655, 1126, 450, 70, "选择 best_A → 冻结表示", ["实体 / 空间 / 关系 / 读出 / 静态链"])
    f.box(1240, 1126, 470, 70, "阶段 B：自由递推 + latent 辅助", ["1 → min(3,H) → H；teacher 只用于标签"])
    f.edge([(520, 1161), (655, 1161)])
    f.edge([(1105, 1161), (1240, 1161)])
    return f.finish()


def comparison_figure():
    f = Figure(1800, 1160, "GSLAformer · 原始 v4.0 与当前实现对照", "左列只提炼原图明确内容；右列为源码事实。新增不等于偏离错误，替代项必须显式说明。")
    f.panel(40, 145, 790, 890, "原始需求图", "三组示意 + 空间 / 时间两级建模 + next embedding 回灌")
    f.panel(970, 145, 790, 890, "当前 full 实现", "保留主闭环，扩展实体历史、关系、条件及训练协议")
    entries = [
        ("用户层 / 机房层 / 楼宇层", ["每组先加 Space Positional Encoding"], "六组业务对象 + 属性编码", ["不加实体行号位置；按实体轨迹先过 GRU"], "changed"),
        ("各组空间 Transformer ×N", ["每组输出空间向量"], "组内一层 → 关系两层 → 查询读出", ["跨组消息先发生在实体轴，再汇总成组向量"], "added"),
        ("空间向量横拼，沿时间纵拼", ["time PE + Masked MHA + FFN"], "z + 总表历史 + 标记 + 日历", ["联合投影 + 位置 → 因果时间 Transformer"], "changed"),
        ("Linear → 总负荷", ["multi Linear → next embedding"], "条件融合 → 总负荷头 / 分组状态头", ["额外融合静态条件、整段未来已知路径、lead"], "added"),
        ("next embedding 返回 time series", ["未定义总负荷回灌、裁窗和训练损失"], "预测 z 与同一预测 y 同时回灌", ["固定 L 窗口；A 学表示 / B 冻结后自由递推"], "changed"),
    ]
    for i, (a, al, b, bl, kind) in enumerate(entries):
        y = 250 + 150 * i
        f.box(85, y, 700, 95, a, al)
        f.box(1015, y, 700, 95, b, bl, kind)
        f.edge([(785, y + 47), (1015, y + 47)])
    f.text(65, 1002, "原图“任何时序结构都可适用”：未指定后端接口。", "small")
    f.text(995, 1002, "当前时间后端固定 Transformer，尚非任意结构可插拔。", "small")
    f.box(40, 1060, 1720, 72, "单独登记的现行规格差异（不与原图演进混算）", ["D01：静态链末端是单 Linear，而不是规格所述 MLP；D02：日历拼接后联合投影，而不是独立投影相加。"], "changed")
    return f.finish()


def source_link(source, html=False):
    path, span = source.rsplit(":", 1)
    first = span.split("-")[0]
    url = "../../../../" + path
    if html:
        return f'<a href="{escape(url)}#L{first}"><code>{escape(source)}</code></a>'
    return f'[{source}]({url}#L{first})'


def main():
    data = json.loads((HERE / "comparison.json").read_text())
    runtime = json.loads((HERE / "evidence/runtime.json").read_text())
    assert runtime["tests"]["success"]
    # 历史回执保留原路径；仅对已确认移动的原图解析新位置，不改写历史哈希。
    assert all(hashlib.sha256((ROOT / ("docs/gslaformer/arch/model-v4.0.png"
               if p == "docs/gslaformer/model-v4.0.png" else p)).read_bytes()).hexdigest() == v
               for p, v in runtime["source_sha256"].items()), "Source changed: re-audit before rendering"
    ids = [r["id"] for r in data["rows"] + data["spec_drift"]]
    assert len(ids) == len(set(ids))
    for row in data["rows"] + data["spec_drift"]:
        for source in row["sources"]:
            path, span = source.rsplit(":", 1)
            total = len((ROOT / path).read_text().splitlines())
            start, end = (int(x) for x in span.split("-"))
            assert 1 <= start <= end <= total, source
    figures = {"current-architecture": architecture(), "mechanisms": mechanisms(), "comparison": comparison_figure()}
    for name, svg in figures.items():
        (HERE / f"{name}.svg").write_text(svg + "\n")
    md = ["# GSLAformer：当前架构与原始 v4.0 需求对照", "",
          f'> 审计时间：{runtime["timestamp"]}。分支 `{runtime["branch"]}`，基准提交 `{runtime["base_commit"]}`，**包含未提交工作区实现**；精确源码快照见 [运行证据](evidence/runtime.json)。',
          "", "## 1. 结论与边界", "",
          "原始闭环（分组空间编码→空间横拼／时间纵拼→因果时间模型→负荷／下一状态双头→回灌）已实现。当前不是原图逐字复刻：增加实体 GRU、类型关系、查询池化、总表历史、静态条件、已知未来路径与 A/B 训练；独立 Space PE 被属性／关系编码替代。",
          "", data["interpretation"], "",
          "“任何时序结构”尚未成为可配置后端，空间 Nx 首版固定为一层。不能以结构核验、随机初始化前向或定向测试通过推断完整模拟验收、真实精度或相对对照收益。", "",
          "## 2. 图形", "",
          "- [离线完整报告](index.html)",
          "- [当前架构 SVG](current-architecture.svg) / [PNG](current-architecture.png) / [PDF](current-architecture.pdf)",
          "- [机制展开 SVG](mechanisms.svg) / [PNG](mechanisms.png) / [PDF](mechanisms.pdf)",
          "- [原始与当前对照 SVG](comparison.svg) / [PNG](comparison.png) / [PDF](comparison.pdf)",
          "", "## 3. 逐项需求映射", ""]
    cards = []
    for row in data["rows"]:
        md += [f'### {row["id"]} {row["topic"]}｜{row["status"]}', "",
               f'- **原图**：{row["original"]}', f'- **实现**：{row["current"]}',
               f'- **判断**：{row["meaning"]}',
               '- **证据**：' + '；'.join(source_link(s) for s in row["sources"]), ""]
        cards.append(f'<article class="item" id="{row["id"]}"><header><span class="pill">{escape(row["status"])}</span><h3>{row["id"]} · {escape(row["topic"])}</h3></header><dl><dt>原始需求</dt><dd>{escape(row["original"])}</dd><dt>当前实现</dt><dd>{escape(row["current"])}</dd><dt>判断</dt><dd>{escape(row["meaning"])}</dd></dl><p class="sources">' + ' · '.join(source_link(s, True) for s in row["sources"]) + '</p></article>')
    md += ["## 4. 现行规格与代码差异（待裁决，不擅改）", ""]
    drift = []
    for row in data["spec_drift"]:
        md += [f'### {row["id"]} {row["topic"]}', "", f'**{row["severity"]}**。', "",
               f'- 规格：{row["spec"]}', f'- 代码：{row["code"]}', f'- 判断：{row["assessment"]}',
               '- 证据：' + '；'.join(source_link(s) for s in row["sources"]), ""]
        drift.append(f'<article class="item drift"><h3>{row["id"]} · {escape(row["topic"])}</h3><p><strong>{escape(row["severity"])}</strong></p><dl><dt>规格</dt><dd>{escape(row["spec"])}</dd><dt>代码</dt><dd>{escape(row["code"])}</dd><dt>判断</dt><dd>{escape(row["assessment"])}</dd></dl><p class="sources">' + ' · '.join(source_link(s, True) for s in row["sources"]) + '</p></article>')
    md += ["## 5. 真实运行证据", "",
           "使用既有 AIDC synthetic 包，正式 day profile 的 L=168、H=24、六组与模型宽度不变；取一个实际训练样本，临时统计只拟合前两个训练窗口，模型随机初始化。只验证结构，不产出训练模型或业务预测结果。", "",
           f'- 历史状态：`{runtime["encoded_history"]["z"]}`；静态条件：`{runtime["encoded_history"]["c"]}`。',
           f'- 时间输入宽度：`{runtime["temporal_input_width"]}`；输出 y：`{runtime["rollout"]["y"]}`；输出 z：`{runtime["rollout"]["z"]}`。',
           '- 实际捕获第二步窗口末 token，逐值验证它等于第一步生成的 z 与 y，并核对 observed/generated/coverage。',
           '- 验证历史编码与未来路径各调用一次，时间链调用 H 次，direct 与 rollout 首步相等。',
           '- 定向模型／契约／训练支持测试真实执行，命令与计数唯一记账于 [实施规格 §12](../../v1/implementation.md#work-packages)；机器结果与日志见 [runtime.json](evidence/runtime.json)、[targeted-tests.log](evidence/targeted-tests.log)。',
           '- 图形浏览器渲染与文本边界检查见 [visual-check.json](evidence/visual-check.json)。',
           "", "## 6. 后续判断", "",
           "- 若以原图主旨为标准：主干与双头回灌已经成立，新增部分是后续 R4 设计扩展。",
           "- 若要求原图逐字一致：需重新裁定 Space PE、空间重复层数、任意时间后端，以及保留哪些新增路径，不能默认为删减授权。",
           "- 优先裁定 D01：接受当前单 Linear 并改文字规格，或补跨组 MLP 并按计算版本升级与重训处理。D02 可优先按等价线性分解澄清文档。",
           "- 本次不修改模型、不改配置／依赖、不启动长训练、不恢复旧 architecture、不提交或推送。", ""]
    (HERE / "comparison.md").write_text('\n'.join(md))
    original = base64.b64encode((ROOT / data["baseline"]).read_bytes()).decode()
    section_svgs = ''.join(f'<section id="{name}"><h2>{title}</h2><p>{desc}</p><div class="figure">{figures[name]}</div><p class="downloads"><a href="{name}.svg">SVG 源图</a> · <a href="{name}.png">PNG 预览</a> · <a href="{name}.pdf">PDF 矢量版</a></p></section>' for name, title, desc in [
        ("current-architecture", "02 / 当前实现 · 完整计算路径", "主图展示 full 变体。主干之外的已知未来／静态条件不是可随意忽略的装饰分支。"),
        ("mechanisms", "03 / 关键机制 · 注意力与状态坐标", "图中宽度来自 AIDC day profile；一般形状使用 B、L、H、Gd、d_s、d_z、d_t 表示。"),
        ("comparison", "04 / 原始与当前 · 看清变化的位置", "此图为人工核实后的语义对照，不补写原图未明确的要求。")])
    body = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GSLAformer — 当前架构与 v4.0 需求对照</title>
<style>
main{{min-width:0;overflow-wrap:anywhere}}
:root{{--ivory:#FAF9F5;--white:#FFFFFF;--slate:#141413;--clay:#D97757;--olive:#788C5D;--rust:#B04A3F;--oat:#E3DACC;--gray-150:#F0EEE6;--gray-300:#D1CFC5;--gray-500:#87867F;--gray-700:#3D3D3A;--serif:Georgia,"Songti SC",serif;--sans:system-ui,-apple-system,"PingFang SC",sans-serif;--mono:Menlo,Consolas,monospace}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:var(--ivory);color:var(--gray-700);font:16px/1.7 var(--sans)}}.layout{{display:grid;grid-template-columns:230px minmax(0,1fr);gap:38px;max-width:1920px;margin:auto;padding:36px 38px 100px}}nav{{position:sticky;top:36px;align-self:start}}nav a{{display:block;padding:9px 12px;color:var(--gray-700);text-decoration:none;border-left:3px solid transparent}}nav a.active{{border-color:var(--clay);background:var(--oat);color:var(--slate)}}.eyebrow{{font:12px var(--mono);letter-spacing:.08em;color:var(--gray-500)}}h1,h2,h3{{color:var(--slate);font-family:var(--serif);font-weight:500}}h1{{font-size:40px;line-height:1.3;margin:16px 0}}h2{{font-size:28px}}h3{{font-size:22px;margin:0}}section{{margin:56px 0;scroll-margin-top:24px}}.callout{{background:var(--white);border:1px solid var(--gray-300);border-left:4px solid var(--clay);border-radius:10px;padding:20px 24px}}.chips{{display:flex;gap:14px;flex-wrap:wrap;margin:20px 0}}.chip{{background:var(--oat);padding:7px 12px;font:13px var(--mono);border-radius:5px}}.figure{{background:var(--white);border:1px solid var(--gray-300);border-radius:12px;overflow:auto}}.figure svg{{display:block;width:100%;height:auto;min-width:1000px}}.original{{max-width:1064px;display:block;width:100%;height:auto;margin:auto}}a{{color:var(--clay)}}code{{font-family:var(--mono);font-size:13px;overflow-wrap:anywhere}}.item{{background:var(--white);border:1px solid var(--gray-300);border-radius:10px;padding:24px;margin:20px 0}}.item header{{display:flex;gap:16px;align-items:center}}.pill{{background:var(--oat);padding:3px 9px;border-radius:5px;font-size:13px;white-space:nowrap}}dl{{display:grid;grid-template-columns:100px 1fr;gap:12px 18px}}dt{{font-weight:600}}dd{{margin:0}}.sources{{font-size:13px;color:var(--gray-500)}}.drift{{border-left:4px solid var(--clay)}}.downloads{{font-size:14px}}table{{border-collapse:collapse;width:100%;background:var(--white)}}td,th{{padding:12px 16px;text-align:left;border-bottom:1px solid var(--gray-300)}}th{{background:var(--gray-150)}}.muted{{color:var(--gray-500)}}@media(max-width:900px){{.layout{{grid-template-columns:1fr;padding:20px}}nav{{position:static}}nav a{{display:inline-block}}h1{{font-size:30px}}dl{{grid-template-columns:1fr;gap:5px}}dd{{margin-bottom:15px}}}}@media print{{nav,.downloads{{display:none}}.layout{{display:block;padding:0}}.figure svg{{min-width:0}}section{{break-before:page}}.item{{break-inside:avoid}}}}
</style></head><body><div class="layout"><nav><div class="eyebrow">GSLAformer / ARCH AUDIT</div><h3>阅读导航</h3><a href="#summary">结论与审计范围</a><a href="#original">原始 v4.0 需求</a><a href="#current-architecture">当前实现架构</a><a href="#mechanisms">关键机制展开</a><a href="#comparison">原始与当前对照</a><a href="#mapping">逐项需求映射</a><a href="#drift">规格差异 · 待裁决</a><a href="#evidence">运行证据与边界</a><a href="comparison.md">Markdown 版</a></nav><main>
<header id="summary"><div class="eyebrow">SOURCE-GROUNDED MODEL DIAGRAM · COMPUTATION VERSION 2</div><h1>GSLAformer<br>从原始需求到当前实现</h1><p>保留空间—时间双层主干与双头回灌；新增实体记忆、类型关系及预测条件。</p><div class="chips"><span class="chip">{escape(runtime['timestamp'])}</span><span class="chip">branch: {escape(runtime['branch'])}</span><span class="chip">L=168 · H=24 · Gd=6</span><span class="chip">未提交工作区快照</span></div><div class="callout"><strong>结论：</strong>原始闭环已经进入真实计算路径，不是只有类名或占位模块。但当前版本不是原图的逐字复刻：Space PE 被属性／关系编码替代，空间 Nx 首版固定一层，“任何时序结构”尚未成为可配置后端。<br><strong>另有两处现行文字规格差异：</strong>静态链末端层数、日历注入方式。它们独立于原图演进，见下方 D01 / D02。<br><strong>本报告不证明：</strong>完整模拟矩阵收口、真实业务精度、优于对照或生产可用性。</div></header>
<section id="original"><h2>01 / 原始需求 · model-v4.0.png</h2><p>{escape(data['interpretation'])}</p><div class="figure"><img class="original" src="data:image/png;base64,{original}" alt="用户提供的 GSLAformer 原始 v4.0 需求架构图"></div><p class="muted">原图按字节内嵌，未改写、未重绘；不能据图推断作者使用的绘图软件。</p></section>
{section_svgs}
<section id="mapping"><h2>05 / 逐项需求映射</h2><p>“具体化”表示原图没有给出精确算子；“新增”表示后续设计扩展，不自动等于实现错误。</p>{''.join(cards)}</section>
<section id="drift"><h2>06 / 现行规格与代码差异</h2><p>本次不擅自修改模型或改写权威设计来消除差异，先记录事实和真实代价。</p>{''.join(drift)}</section>
<section id="evidence"><h2>07 / 真实运行证据与适用边界</h2><p>复用现有 AIDC synthetic 标准包；正式 day profile 的 L/H 和架构不变，随机初始化模型，仅取一个实际训练样本运行。临时 scaler 只以两个训练窗口拟合，不保存模型或业务预测。</p><table><tr><th>检查对象</th><th>实际结果</th></tr><tr><td>历史状态 z / 静态条件 c</td><td><code>{runtime['encoded_history']['z']} / {runtime['encoded_history']['c']}</code></td></tr><tr><td>总负荷 y / 预测状态 z</td><td><code>{runtime['rollout']['y']} / {runtime['rollout']['z']}</code></td></tr><tr><td>真实回灌</td><td>第二步输入逐值等于第一步生成 z / y；质量标记符合生成态</td></tr><tr><td>调用边界</td><td>历史表示与未来路径各一次，时间链逐预测步重算</td></tr><tr><td>direct / rollout 与冻结</td><td>同权重首步相等；B 阶段表示链冻结且保持 eval</td></tr></table><p>定向测试命令与计数唯一记录于 <a href="../../v1/implementation.md#work-packages">实施规格 §12</a>。源码 SHA-256、形状、调用次数和机器测试结果见 <a href="evidence/runtime.json">runtime.json</a>；运行日志见 <a href="evidence/targeted-tests.log">targeted-tests.log</a>；图形检查见 <a href="evidence/visual-check.json">visual-check.json</a>。</p><div class="callout">建议先裁定 D01 的跨组静态非线性层是否确实需要，再决定修改规格还是升级模型。本次仅交付架构审计，不改变模型、配置或依赖，不启动长训练，不恢复旧 architecture，不提交推送。</div><p class="muted">代码链接的 #L 行号用于审计定位，本地文件查看器未必自动跳转。当前分支基准提交 {escape(runtime['base_commit'])}；未提交内容以证据中的文件哈希为准。</p></section>
</main></div><script>(()=>{{const links=[...document.querySelectorAll('nav a[href^="#"]')];const observer=new IntersectionObserver(entries=>{{for(const e of entries)if(e.isIntersecting){{links.forEach(a=>a.classList.toggle('active',a.hash==='#'+e.target.id));}}}},{{rootMargin:'-10% 0px -70% 0px'}});document.querySelectorAll('main section,main>header').forEach(s=>observer.observe(s));}})();</script></body></html>'''
    (HERE / "index.html").write_text(body)
    counts = Counter(row["status"] for row in data["rows"])
    print(json.dumps({"figures": list(figures), "requirements": len(data["rows"]),
                      "spec_drift": len(data["spec_drift"]), "statuses": counts,
                      "source_hashes_match": True}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
