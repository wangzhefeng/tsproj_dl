"""生成与源码一致的三幅架构图；仅写入本目录，不覆盖旧图。"""
from pathlib import Path
import hashlib
import html
import json
import subprocess
import unicodedata
import xml.etree.ElementTree as ET

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[4]
PALETTE = {
    'input': ('#f1f5f9', '#64748b'), 'entity': ('#ede9fe', '#8b74b6'),
    'spatial': ('#fff0d9', '#cd9450'), 'state': ('#e3f0ff', '#5486b9'),
    'condition': ('#ddf2ec', '#589a8b'), 'output': ('#e8f3de', '#7d9d60'),
    'loss': ('#fbe6ea', '#b97586'), 'note': ('#ffffff', '#c4cfdb'),
}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wrap(text, width, size):
    result = []
    for part in text.split('\n'):
        row, used = '', 0
        for char in part:
            advance = size if unicodedata.east_asian_width(char) in 'WF' else size * .56
            if used + advance > width and row:
                result.append(row)
                row, used = '', 0
            row += char
            used += advance
        result.append(row)
    return result


class Diagram:
    def __init__(self, name, title, subtitle, width=1800, height=1440):
        self.name, self.width, self.height = name, width, height
        self.nodes, self.edges = [], []
        self.title, self.subtitle = title, subtitle

    def box(self, key, title, body, x, y, w, h, kind='input'):
        node = dict(id=key, title=title, body=body, x=x, y=y, w=w, h=h, kind=kind)
        lines = wrap(body, w - 36, 21)
        needed = 54 + len(lines) * 28
        if needed > h:
            raise ValueError(f'{self.name}/{key}: text needs {needed}, box has {h}: {lines}')
        self.nodes.append(node)
        return node

    def edge(self, source, target, points, dashed=False):
        assert all(a[0] == b[0] or a[1] == b[1] for a, b in zip(points, points[1:]))
        self.edges.append(dict(source=source, target=target, points=points, dashed=dashed))

    def save(self):
        esc = html.escape
        svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" viewBox="0 0 {self.width} {self.height}">',
               '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 Z" fill="#4a5e74"/></marker></defs>',
               f'<rect width="{self.width}" height="{self.height}" fill="#ffffff"/>',
               '<g font-family="PingFang SC,Microsoft YaHei,Noto Sans CJK SC,sans-serif" fill="#203348">',
               f'<text x="60" y="61" font-size="34" font-weight="700">{esc(self.title)}</text>',
               f'<text x="60" y="104" font-size="21" fill="#5c6f83">{esc(self.subtitle)}</text>']
        graph = ET.Element('mxGraphModel', dx='1800', dy='1440', grid='1', gridSize='10', page='1', pageScale='1', pageWidth=str(self.width), pageHeight=str(self.height))
        root = ET.SubElement(graph, 'root')
        ET.SubElement(root, 'mxCell', id='0')
        ET.SubElement(root, 'mxCell', id='1', attrib={'parent': '0'})
        for key, text, y, size, height in [('title', self.title, 22, 34, 50), ('subtitle', self.subtitle, 76, 21, 45)]:
            cell = ET.SubElement(root, 'mxCell', id=key, value=text, vertex='1', attrib={'parent': '1'}, style=f'text;html=0;align=left;fontSize={size};fontColor=#203348;whiteSpace=wrap;')
            ET.SubElement(cell, 'mxGeometry', x='60', y=str(y), width=str(self.width - 120), height=str(height), attrib={'as': 'geometry'})
        for index, edge in enumerate(self.edges):
            points = edge['points']
            dashed = ' stroke-dasharray="9 7"' if edge['dashed'] else ''
            path = 'M ' + ' L '.join(f'{x} {y}' for x, y in points)
            svg.append(f'<path d="{path}" fill="none" stroke="#4a5e74" stroke-width="2.5"{dashed} marker-end="url(#arrow)"/>')
            source = next(n for n in self.nodes if n['id'] == edge['source'])
            target = next(n for n in self.nodes if n['id'] == edge['target'])
            exit_x = (points[0][0] - source['x']) / source['w']
            exit_y = (points[0][1] - source['y']) / source['h']
            entry_x = (points[-1][0] - target['x']) / target['w']
            entry_y = (points[-1][1] - target['y']) / target['h']
            anchors = f'exitX={exit_x};exitY={exit_y};entryX={entry_x};entryY={entry_y};exitPerimeter=0;entryPerimeter=0;'
            cell = ET.SubElement(root, 'mxCell', id=f'edge{index}', edge='1', source=edge['source'], target=edge['target'], attrib={'parent': '1'}, style='edgeStyle=orthogonalEdgeStyle;rounded=0;html=0;endArrow=block;strokeColor=#4a5e74;strokeWidth=2;' + anchors + ('dashed=1;' if edge['dashed'] else ''))
            # 保留连接对象、固定锚点与正交拐点，支持在 draw.io 中继续编辑。
            geo = ET.SubElement(cell, 'mxGeometry', relative='1', attrib={'as': 'geometry'})
            ET.SubElement(geo, 'mxPoint', x=str(points[0][0]), y=str(points[0][1]), attrib={'as': 'sourcePoint'})
            ET.SubElement(geo, 'mxPoint', x=str(points[-1][0]), y=str(points[-1][1]), attrib={'as': 'targetPoint'})
            array = ET.SubElement(geo, 'Array', attrib={'as': 'points'})
            for x, y in points[1:-1]:
                ET.SubElement(array, 'mxPoint', x=str(x), y=str(y))
        for n in self.nodes:
            fill, stroke = PALETTE[n['kind']]
            x, y, w, h = (n[k] for k in ('x', 'y', 'w', 'h'))
            svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>')
            svg.append(f'<text x="{x+18}" y="{y+33}" font-size="25" font-weight="650">{esc(n["title"])}</text>')
            for i, line in enumerate(wrap(n['body'], w-36, 21)):
                svg.append(f'<text x="{x+18}" y="{y+65+i*28}" font-size="21">{esc(line)}</text>')
            value = '<div style="font-size:25px;font-weight:bold;margin-bottom:10px">' + esc(n['title']) + '</div><div style="font-size:21px;line-height:28px">' + '<br>'.join(esc(line) for line in wrap(n['body'], w-36, 21)) + '</div>'
            cell = ET.SubElement(root, 'mxCell', id=n['id'], value=value, vertex='1', attrib={'parent': '1'}, style=f'rounded=1;arcSize=8;html=1;whiteSpace=wrap;align=left;verticalAlign=top;spacing=18;fontSize=21;fontColor=#203348;fillColor={fill};strokeColor={stroke};')
            ET.SubElement(cell, 'mxGeometry', x=str(x), y=str(y), width=str(w), height=str(h), attrib={'as': 'geometry'})
        for index, (kind, label) in enumerate([('input','输入 / 证据'),('entity','实体 / 特征投影'),('spatial','空间 / 关系'),('state','状态 / 时间链'),('condition','条件 / teacher'),('output','预测 / 权重'),('loss','监督损失')]):
            x, y = 60 + index * 240, self.height - 44
            fill, stroke = PALETTE[kind]
            svg.append(f'<rect x="{x}" y="{y-17}" width="20" height="20" fill="{fill}" stroke="{stroke}"/><text x="{x+30}" y="{y}" font-size="19">{label}</text>')
            cell = ET.SubElement(root, 'mxCell', id='legend_'+kind, value=label, vertex='1', attrib={'parent':'1'}, style=f'rounded=0;html=0;fontSize=18;fillColor={fill};strokeColor={stroke};')
            ET.SubElement(cell, 'mxGeometry', x=str(x), y=str(y-24), width='210', height='35', attrib={'as':'geometry'})
        svg.append('</g></svg>')
        (OUT / f'{self.name}.svg').write_text('\n'.join(svg))
        mx = ET.Element('mxfile', host='app.diagrams.net', version='24.7.17')
        ET.SubElement(mx, 'diagram', name=self.title, id=self.name).append(graph)
        ET.ElementTree(mx).write(OUT / f'{self.name}.drawio', encoding='utf-8', xml_declaration=True)
        return dict(name=self.name, width=self.width, height=self.height, nodes=self.nodes, edges=self.edges)


def overview():
    d = Diagram('gslaformer_impl_revised_01', 'GSLAformer · 实体历史与空间表示', '01 / 通用 full 结构：总组数 G 与动态组数 Gd 分开；AIDC 六动态组仅是当前配置实例', height=1400)
    d.box('observations', '历史实体观测与属性', '逐组 [B,L,Ng,Fg]；实体身份只用于对齐\n数值、观测、适用性、静态属性及有效性', 60,180,480,140)
    d.box('adapter', '逐组共享实体适配', '动态值、掩码与属性拼接\nLinear → GELU → Linear；输出宽度 ds',60,370,480,140,'entity')
    d.box('dynamic', '动态组', '同组共享 GRU\n逐实体沿时间更新',60,560,225,140,'entity')
    d.box('static', '纯静态组', '无 GRU\n适配结果直接传递',315,560,225,140,'entity')
    d.box('within', '逐组、逐时刻的实体交互', '实体 + summary token → 单层空间注意力\n去掉 summary 输出；保留实体 [B,L,Ng,ds]',60,760,480,145,'spatial')
    d.box('edges', '历史关系与有效实体', '每个历史时刻使用对应关系\n有向类型、允许反向通信、padding / present',660,560,500,140)
    d.box('relation', '跨组类型关系交互', '在实体轴拼接全部 G 组；按帧独立计算\n加性通信掩码与方向偏置；层数来自配置',660,760,500,145,'spatial')
    d.box('readout', '仅对动态组进行独立查询读出', 'Q：可学习查询；K/V：实体 + 常驻空组 token\n池化结果拼接 log1p(N)、coverage → Linear → LN',660,965,500,150,'spatial')
    d.box('z', '动态组历史状态 z', '全窗口 [B,L,Gd,dz]；单时刻 [B,Gd,dz]\n按组轴堆叠，进入时间链时才展平特征轴',660,1170,500,140,'state')
    d.box('origin', '预测起点的静态证据', '有效实体的属性、类别与数量\n起点关系的逐类型出入度；不读动态值',1260,180,480,140)
    d.box('group_static', '逐组静态统计与投影', '均值、总和、可用率、数量及度数\n类别嵌入均值；每组 Linear → GELU',1260,370,480,140,'condition')
    d.box('c', '固定静态条件 c', '拼接全部组统计表示 → 单 Linear\n[B,dt]；同一起点的整段预测共用',1260,560,480,140,'condition')
    d.box('scope', '配置边界', 'G：全部组；Gd：动态组\n所有组可参与关系；只有动态组产生 z\n关系层通常为 2 层，但不是硬编码常量',1260,790,480,170,'note')
    d.box('next', '后续使用', 'z 与真实负荷历史用于时间预测\nc 独立进入每一步条件融合\n推理路径与训练路径分别见图 02、03',1260,1060,480,170,'note')
    d.edge('observations','adapter',[(300,320),(300,370)])
    d.edge('adapter','dynamic',[(300,510),(300,535),(172,535),(172,560)])
    d.edge('adapter','static',[(300,510),(300,535),(428,535),(428,560)])
    d.edge('dynamic','within',[(172,700),(172,730),(300,730),(300,760)])
    d.edge('static','within',[(428,700),(428,730),(300,730),(300,760)])
    d.edge('within','relation',[(540,832),(660,832)])
    d.edge('edges','relation',[(910,700),(910,760)])
    d.edge('relation','readout',[(910,905),(910,965)])
    d.edge('readout','z',[(910,1115),(910,1170)])
    d.edge('origin','group_static',[(1500,320),(1500,370)])
    d.edge('group_static','c',[(1500,510),(1500,560)])
    return d


def inference():
    d = Diagram('gslaformer_impl_revised_02', 'GSLAformer · 直接预测与自由递推', '02 / 两条路径分开：模型内部负荷为标准化坐标；对外结果另行反标准化',height=1600)
    d.box('shared', '同一起点只准备一次', '历史编码得到 z、c；完整起点已知未来路径经双向注意力得到 p[1:H]；lead 相对原起点，不随滑窗重置。',60,150,1680,115,'condition')
    d.box('direct_start', 'A 阶段 / direct：真实历史窗口', 'z 展平 + 真实负荷 + 观测/生成标记\n逐组 coverage/生成标记 + 日历四维',80,330,700,140,'state')
    d.box('direct_project', '全部 token 特征联合投影', '单 Linear → 加窗口内正弦位置\n日历已在 token 内，不再独立投影相加',80,520,700,140,'entity')
    d.box('direct_time', '因果时间编码只运行一次', '沿历史时间轴计算；取末位表示 h\n所有未来步共享这个历史表示',80,710,700,140,'state')
    d.box('direct_fusion', '按未来步条件融合', '拼接 h、固定 c、p[k]、lead(k)\n共享 Linear → GELU → Dropout → Linear',80,900,700,140,'condition')
    d.box('direct_load', '仅负荷头；没有 latent 预测和回灌', '各步共享 Linear(dt,1)\n收集 H 个预测，得到 [B,H,1]',80,1090,700,140,'output')
    d.box('direct_output', '对外负荷预测', '预测完成后反标准化\n物理单位遵循目标合同；不是 latent 的单位',80,1310,700,140,'output')
    d.box('ar_state', '自由递推：内部滚动状态窗口', '由真实历史初始化；窗口长度始终为 L\n预测写入内部状态，不改原始 target_history',980,330,700,140,'state')
    d.box('ar_project', '全部 token 特征联合投影', '单 Linear → 加当前窗口内正弦位置\n每步重算；不是 KV cache 或跨起点记忆',980,520,700,140,'entity')
    d.box('ar_time', '每步重新计算因果时间编码', '从当前窗口取末位表示 h[k]\n空间历史只编码一次，不随滑窗重新运行',980,710,700,140,'state')
    d.box('ar_fusion', '按未来步条件融合', '拼接当前 h[k]、固定 c、p[k]、lead(k)\np 来自完整起点已知路径，不刷新晚到条件',980,900,700,140,'condition')
    d.box('latent_head', '动态组 latent 头', '每组 Linear(dt,dz)\n预测 [B,Gd,dz]',980,1090,325,140,'output')
    d.box('load_head', '总负荷头', '共享 Linear(dt,1)\n预测 [B,1]',1355,1090,325,140,'output')
    d.box('append', '追加预测状态，下一步取新的 L 长窗口', '同时追加预测 z、y 与未来日历；丢弃最旧位置\nobserved=false，generated=true\ncoverage=0，group_generated=1；收集 H 步输出',980,1310,700,170,'state')
    for prefix in ('direct','ar'):
        first='direct_start' if prefix=='direct' else 'ar_state'
        x=430 if prefix=='direct' else 1330
        for a,b,y0,y1 in [(first,prefix+'_project',470,520),(prefix+'_project',prefix+'_time',660,710),(prefix+'_time',prefix+'_fusion',850,900)]:
            d.edge(a,b,[(x,y0),(x,y1)])
    d.edge('direct_fusion','direct_load',[(430,1040),(430,1090)])
    d.edge('direct_load','direct_output',[(430,1230),(430,1310)])
    d.edge('ar_fusion','latent_head',[(1330,1040),(1330,1065),(1142,1065),(1142,1090)])
    d.edge('ar_fusion','load_head',[(1330,1040),(1330,1065),(1518,1065),(1518,1090)])
    d.edge('latent_head','append',[(1142,1230),(1142,1310)])
    d.edge('load_head','append',[(1518,1230),(1518,1310)])
    d.edge('append','ar_state',[(1680,1395),(1740,1395),(1740,400),(1680,400)],True)
    return d


def training():
    d = Diagram('gslaformer_impl_revised_03', 'GSLAformer · 两阶段训练与监督隔离', '03 / full 的训练机制：teacher 仅产生监督标签，不进入生产预测状态；所有图示均非新增算法',height=1540)
    d.box('phase_a','A：负荷监督学习表示','latent 头冻结，不参与此阶段损失',60,160,470,115,'entity')
    d.box('freeze','阶段切换：加载 best_A 并冻结表示链','冻结实体历史、组内空间、关系、读出、静态链\n冻结模块保持 eval；建立 latent 训练统计',600,160,600,145,'condition')
    d.box('phase_b','B：自由递推与辅助监督','时间链、未来路径、融合层、两头继续训练',1270,160,470,145,'state')
    d.box('a_input','历史与起点已知未来条件','真实未来负荷仅作为损失标签\n不输入预测分支',60,370,650,140)
    d.box('a_direct','直接多步负荷预测','完整空间/关系编码 + 时间链 + 条件融合\n调用负荷头，不调用 latent 头；不回灌',60,570,650,140,'entity')
    d.box('a_loss','主损失：预测负荷对真实标签','只在有效标签上计算\n先按预测步分别平均，再等权汇总',60,770,650,140,'loss')
    d.box('best','验证集选择 best_A','B 从选中的 A 权重开始，而非最后一次更新\nA 阶段不以 latent 辅助目标预训练 latent 头',60,970,650,140,'output')
    d.box('curriculum','B 阶段课程与冻结边界','监督递推长度：1 → min(3,H) → H\n未来条件仍编码完整 H；不截断回灌梯度\nB 的主预测路径不读取未来真实负荷或实体值',60,1200,650,170,'note')
    d.box('b_input','生产预测分支输入','真实历史 + 起点已知条件\n不接收未来真值',850,370,390,140)
    d.box('b_roll','双头自由递推','历史编码一次并冻结\n预测 z、y 回灌内部窗口',850,570,390,140,'state')
    d.box('b_y','预测负荷与预测 latent','主损失读取预测负荷\n辅助损失读取预测 latent',850,770,390,140,'output')
    d.box('loss_y','主负荷损失','预测负荷对真实未来负荷\n遵守有效标签掩码',850,970,390,140,'loss')
    d.box('teacher_input','仅训练可用的监督数据','未来实体观测及掩码\n不是生产模型输入',1350,370,390,140)
    d.box('teacher','冻结 teacher：no_grad','复制历史末端实体 memory\n按起点上下文顺序延续编码',1350,570,390,140,'condition')
    d.box('labels','规范 latent 标签与质量 q','标签不写回生产窗口\n质量及维度权重仅用于损失',1350,770,390,140,'condition')
    d.box('loss_z','latent 辅助损失','预测 latent 对 teacher 标签\n按质量及训练方差加权',1350,970,390,140,'loss')
    d.box('combined','B 总损失与可训练参数更新','主负荷损失 + λ × latent 辅助损失\n标准配置 λ=0.1；no_latent_loss 消融为 0\n冻结表示链不更新；teacher 不替代任何一步预测输入',850,1200,890,170,'loss')
    d.edge('phase_a','freeze',[(530,215),(600,215)])
    d.edge('freeze','phase_b',[(1200,230),(1270,230)])
    for col,ids in [(385,['a_input','a_direct','a_loss','best']),(1045,['b_input','b_roll','b_y','loss_y']),(1545,['teacher_input','teacher','labels','loss_z'])]:
        for a,b,y in zip(ids,ids[1:],[510,710,910]):
            d.edge(a,b,[(col,y),(col,y+60)])
    d.edge('b_y','loss_z',[(1240,840),(1295,840),(1295,1040),(1350,1040)],True)
    d.edge('loss_y','combined',[(1045,1110),(1045,1200)])
    d.edge('loss_z','combined',[(1545,1110),(1545,1200)])
    return d


if __name__ == '__main__':
    original = {p.name: digest(p) for p in OUT.parent.iterdir() if p.is_file()}
    sources = ['models/transformer/GSLAformer.py', 'layers/gslaformer/entity_history.py',
               'layers/gslaformer/spatial.py', 'layers/gslaformer/relations.py',
               'layers/gslaformer/temporal.py', 'exp/exp_gslaformer_forecasting.py',
               'exp/gslaformer_support.py', 'exp/gslaformer_inference.py']
    layouts = [diagram.save() for diagram in (overview(), inference(), training())]
    (OUT / 'layout.json').write_text(json.dumps(layouts, ensure_ascii=False, indent=2))
    manifest = {'git_head': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                'source_sha256': {p: digest(ROOT / p) for p in sources},
                'original_assets_sha256': original,
                'scope': 'diagram-only; no training, tests, or source changes',
                'drawio_export': 'native editable geometry generated from same layout as SVG; desktop exporter unavailable'}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    assert original == {p.name: digest(p) for p in OUT.parent.iterdir() if p.is_file()}
    print('Generated three SVG/draw.io diagrams; original assets unchanged.')
