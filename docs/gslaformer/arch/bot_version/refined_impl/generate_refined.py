"""在原 draw.io 副本上精修单页图；不覆盖原图或三图修订版。"""
from pathlib import Path
import copy
import hashlib
import html
import json
import re
import unicodedata
import xml.etree.ElementTree as ET

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
NAME = 'gslaformer_impl_refined'
W, H = 3080, 2800
GEOMETRY_SCALE, FONT_SCALE = 2.0, 1.8
FONT = 'PingFang SC,Microsoft YaHei,Noto Sans CJK SC,sans-serif'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def style_dict(cell):
    return dict(piece.split('=', 1) for piece in cell.get('style', '').split(';') if '=' in piece)


def set_style(cell, **changes):
    values = style_dict(cell)
    values.update({k: str(v) for k, v in changes.items()})
    cell.set('style', ';'.join(f'{k}={v}' for k, v in values.items()) + ';')


def geom(cell):
    g = cell.find('mxGeometry')
    return [float(g.get(k, '0')) for k in ('x', 'y', 'width', 'height')]


def wrap(text, width, size):
    lines = []
    for line in text.split('\n'):
        current, used = '', 0
        for char in line:
            advance = size if unicodedata.east_asian_width(char) in 'WF' else size * .56
            if current and used + advance > width:
                lines.append(current)
                current, used = '', 0
            current += char
            used += advance
        lines.append(current)
    return lines


def main():
    baseline_path = OUT / 'preserved_files.json'
    if not baseline_path.exists():
        preserved = {str(p.relative_to(BASE)): sha(p) for p in BASE.rglob('*')
                     if p.is_file() and OUT not in p.parents}
        baseline_path.write_text(json.dumps(preserved, ensure_ascii=False, indent=2))
    else:
        preserved = json.loads(baseline_path.read_text())
    assert all(sha(BASE / p) == h for p, h in preserved.items()), 'Existing asset changed'
    # 仅解析仓库中已审阅的原图 XML，不接收外部不可信 XML。
    tree = copy.deepcopy(ET.parse(BASE / 'gslaformer_impl.drawio'))
    graph = next(tree.iter('mxGraphModel'))
    graph.set('pageWidth', str(W))
    graph.set('pageHeight', str(H))
    root = graph.find('root')
    assert root is not None
    cells = {c.get('id'): c for c in root.findall('mxCell')}
    original_ids = set(cells)

    def update(key: str, text=None, xywh=None, size=None, **style):
        c = cells[key]
        if text is not None:
            c.set('value', text)
        if xywh:
            geometry = c.find('mxGeometry')
            assert geometry is not None
            for field, value in zip(('x', 'y', 'width', 'height'), xywh):
                geometry.set(field, str(value))
        if size is not None:
            style['fontSize'] = size
        set_style(c, html=0, fontFamily='PingFang SC', **style)
        return c

    def add(key: str, text, xywh, fill='#f5f5f5', stroke='#8795a3', size: float=11, **style):
        c = ET.SubElement(root, 'mxCell', id=key, value=text, vertex='1', attrib={'parent':'1'})
        ET.SubElement(c, 'mxGeometry', attrib={'as':'geometry'})
        cells[key] = c
        defaults = dict(rounded=1, arcSize=8, whiteSpace='wrap', fillColor=fill,
                        strokeColor=stroke, align='center', verticalAlign='middle')
        defaults.update(style)
        return update(key, text, xywh, size, **defaults)

    paths = []

    def route(key: str, source: str, target: str, points, color='#435465', dashed=False):
        if key in cells:
            c = cells[key]
            c.clear()
            c.attrib.update(id=key, edge='1', parent='1', source=source, target=target, value='')
        else:
            c = ET.SubElement(root, 'mxCell', id=key, edge='1', source=source, target=target, attrib={'parent':'1'})
            cells[key] = c
        sx, sy, sw, sh = geom(cells[source])
        tx, ty, tw, th = geom(cells[target])
        set_style(c, edgeStyle='orthogonalEdgeStyle', rounded=0, html=0, endArrow='block',
                  endSize=5, strokeColor=color, strokeWidth=1.2, dashed=int(dashed),
                  exitX=(points[0][0]-sx)/sw, exitY=(points[0][1]-sy)/sh,
                  entryX=(points[-1][0]-tx)/tw, entryY=(points[-1][1]-ty)/th,
                  exitPerimeter=0, entryPerimeter=0)
        geo = ET.SubElement(c, 'mxGeometry', relative='1', attrib={'as':'geometry'})
        arr = ET.SubElement(geo, 'Array', attrib={'as':'points'})
        for x, y in points[1:-1]:
            ET.SubElement(arr, 'mxPoint', x=str(x), y=str(y))
        assert all(a[0] == b[0] or a[1] == b[1] for a,b in zip(points,points[1:])), key
        paths.append(dict(id=key, source=source, target=target, points=points, color=color, dashed=dashed))

    update('c2', 'GSLAformer 实现架构 · AIDC 六动态组实例', (20,10,1500,32), 21, fontColor='#233547')
    update('c3', '(a) encode_history · 实体历史与空间表示', (20,55,880,985), 14, dashed=1, dashPattern='7 5', strokeWidth=1.3, fillColor='#fcfcfc')
    update('c94', '(b) rollout · 时间建模与双头自由递推', (920,55,600,985), 14, dashed=1, dashPattern='7 5', strokeWidth=1.3, fillColor='#fcfcfc')
    add('origin_input', '预测起点的静态证据\n属性 / 类别 / 有效实体\n逐类型、逐方向的关系度数', (50,100,270,80), fill='#f5f5f5', size=12)
    update('c92', 'StaticConditionEncoder · 独立静态条件 c\n属性均值/总和、可用率、类别均值、数量与关系度数\n逐组 Linear → GELU；跨组拼接 → 单 Linear；起点内固定', (360,100,500,80), 12)
    update('c93', '空间表示按历史时刻计算；一次预测只编码一次历史 z 与静态条件 c。', (50,185,830,24), 11, align='left')
    update('c67', '动态组状态：单时刻 [B,6,dz]；全窗口 [B,L,6,dz]', (140,212,540,22), size=12)
    for i, offset in enumerate((0,16,32)):
        names = ['compute · 算力实体','it_power · IT 电力实体','hvac · 暖通实体']
        update(f'c{4+offset}', names[i]+'\n动态值 / 观测掩码 / 适用性\n静态数值与类别属性', (50+250*i,800,200,65), 11)
        update(f'c{5+offset}', 'GroupAdapter · 实体适配\n数值 / 掩码 / 类别嵌入\nLinear → GELU → Linear', (70+250*i,724,160,50), 11)
        update(f'c{6+offset}', 'GRUCell · 实体历史\n同组共享，逐实体沿时间更新', size=11)
        update(f'c{9+offset}', 'Multi-Head Attention\n组内注意力 · 摘要与实体', size=10.5)
        update(f'c{10+offset}', 'Add & Norm · 残差归一化', size=10.5)
        update(f'c{11+offset}', 'FFN · 前馈（4ds / GELU）', size=11)
        update(f'c{12+offset}', 'Add & Norm · 残差归一化', size=10.5)
        update(f'c{13+offset}', 'GroupSpatialEncoder\n组内空间编码 · 单层', (55+250*i,472,190,28), size=10.5)
    update('c52', '其余三组\n配电 / 空间 / 环境\n共六个动态组', (765,790,120,78), 11)
    update('c54', 'RelationAttention · 类型关系交互（2 × RelationBlock，当前配置）\n逐时刻拼接六组实体；按关系类型、方向及注意力头添加可学习偏置\n允许自连接、已注册关系及反向通信；MHA → Add & Norm → FFN → Add & Norm', size=11.5)
    update('c55', 'relations_history\n历史关系 Rₜ · 按时刻对齐', size=10.5)
    for key, x in [('c60',50),('c62',300),('c64',550)]:
        update(key, 'GroupReadout · 查询读出\nQ：可学习查询\nK/V：实体 + 常驻空组 token\n拼接 log1p(N)、coverage\nLinear → LayerNorm', (x,280,200,96), 11)
    add('history_scope', 'EntityHistoryEncoder · 实体历史编码：GroupAdapter + 同组共享 GRUCell；缺席或填充实体的状态归零。', (50,878,830,30), fill='none', stroke='none', size=11.5)
    add('spatial_note', '配置与表示边界\n本图为六动态组实例；其他配置可含纯静态组（无 GRU、不产生动态 latent）。\n实体身份只用于对齐；摘要 token 输出被丢弃，分组状态由独立查询读出。\n消融示例：no_spatial → GroupMeanReadout；no_entity_history → 无 GRU；total_only → 仅负荷。', (50,925,830,85), fill='#ffffff', stroke='#c4cbd2', size=12)
    update('c95', '内部滚动状态 · 按时间排列的 L 个位置', (950,640,540,140), 13, dashed=0)
    add('state_note','保留最近 L 个位置\n追加预测 z、y\n不覆盖真实历史', (1290,669,185,68), fill='none', stroke='none', size=12)
    update('c165', '每步特征：z 展平 / 负荷 y / 观测与生成标记 / 逐组覆盖率 / 日历四维', (960,746,520,28), 10.5)
    update('c167', 'target_history · 真实负荷历史\n仅用于初始化内部状态', (1240,835,250,50), 12)
    update('c169', '全部特征联合 Linear → dt；再加窗口内正弦位置（不另设日历投影）', size=12)
    update('c172', '×Nₜ', size=12)
    update('c173', 'Masked Multi-Head Attention · 因果注意力\n屏蔽未来窗口位置', size=12)
    update('c174', 'Add & Norm · 残差相加与归一化', size=12)
    update('c175', 'Feed Forward · 前馈层（4dt / GELU）', size=12)
    update('c176', 'Add & Norm · 残差相加与归一化', size=12)
    update('c177', 'TemporalEncoder · 因果时间编码\nPost-LN；每步取当前窗口末位 h', size=11.5)
    update('c182', 'FuturePathEncoder\n已知未来路径编码\n值 / mask / 日历\n完整 H 步双向注意力\npₖ：起点一次编码', (1355,455,135,90), 11)
    update('c183', 'lead(k) · 预测步长编码\n相对原起点的正弦编码', (1355,395,135,45), 11)
    update('c184', 'ConditionFusion · 条件融合\n历史 h / 静态 c / 未来 pₖ / 步长 lead(k)\n拼接 → Linear → GELU → Dropout → Linear', (1030,282,460,54), size=12)
    update('c189', 'load_head · 总负荷头\nLinear(dt,1)', size=12)
    update('c190', 'latent_heads · 分组状态头\n每组 Linear(dt,dz)', size=12)
    update('c193', '下一步总负荷 ŷ\n[B,1] · 标准化坐标', size=11)
    update('c194', '下一步动态组状态 ẑ\n[B,6,dz]', size=12)
    update('c198', '预测位置的状态标记\nobserved=false；generated=true\ncoverage=0；group_generated=1', (950,805,270,90), 11, fontColor='#a4524e', align='left')
    update('c200', 'direct 对照 / A 阶段\n历史时间编码仅运行一次；各未来步使用自身条件\n只调用负荷头，不预测 latent、不回灌', (950,925,540,85), 12, fillColor='#eef3fb', strokeColor='#7f96b8', rounded=1, fontColor='#33465a')
    add('feedback_join','',(1502,137,6,6),fill='#b85450',stroke='#b85450',rounded=0)
    add('training_panel','(c) 两阶段训练 · 先学习表示，再训练自由递推',(20,1080,1500,230),fill='#f8fafc',stroke='#aeb9c5',size=14,align='left',verticalAlign='top')
    add('stage_a','A · 直接多步负荷监督\n训练表示链及负荷预测路径\nlatent 头冻结，不参与损失\n按验证集选出 best_A',(45,1140,310,110),fill='#e1d5e7',stroke='#9673a6',size=12)
    add('stage_freeze','加载 best_A · 冻结完整表示链\n实体历史 / 组内空间 / 关系 / 读出 / 静态链\n冻结模块保持 eval\n建立 latent 训练统计',(405,1140,360,110),fill='#d0ece7',stroke='#3f8f80',size=12)
    add('stage_b','B · 双头自由递推\n预测负荷主损失 + latent 辅助损失\n时间链、未来路径、融合层与两头继续训练\n预测 z、y 回灌，不使用 teacher 真值回灌',(815,1140,360,110),fill='#dae8fc',stroke='#7f96b8',size=12)
    add('teacher','仅训练使用的 teacher\n冻结表示链 + 未来实体监督\n复制历史末端 memory；no_grad\nlatent 标签仅进入辅助损失',(1225,1140,270,110),fill='#fff2cc',stroke='#d6b656',size=12)
    add('training_note','B 课程：1 → min(3,H) → H；完整已知未来路径保持 H 步。预测结束后负荷反标准化；latent 不具有负荷物理单位。',(45,1265,1450,28),fill='none',stroke='none',size=11.5)
    update('c201','B：批大小；L：历史窗口；H：预测长度；ds：实体表示宽度；dz：组状态宽度；dt：时间表示宽度。图示为 full 主路径，direct 单独说明。',(20,1320,1500,26),11.5)
    colors=[('#e1d5e7','特征适配 / 读出'),('#eef3fb','历史 / 前馈层'),('#f5c28e','注意力 / 关系'),('#fff2cc','残差归一化'),('#d0ece7','静态 / 未来条件'),('#d5e8d4','预测头')]
    for i,(color,text) in enumerate(colors):
        add(f'legend{i}',text,(30+i*250,1360,225,25),fill=color,stroke='#b5bec7',size=11)

    # 重复实体分支保留原图位置，只有适配器和读出框局部加高。
    for i,o in enumerate((0,16,32)):
        x=150+250*i
        for key,s,t,y0,y1 in [(14+o,4+o,5+o,800,774),(15+o,5+o,6+o,724,712),(16+o,6+o,9+o,672,639),(17+o,9+o,10+o,605,592),(18+o,10+o,11+o,572,564),(19+o,11+o,12+o,542,532)]:
            route(f'c{key}',f'c{s}',f'c{t}',[(x,y0),(x,y1)])
        route(f'c{57+i}',f'c{12+o}','c54',[(70+250*i,522),(45+250*i,522),(45+250*i,470),(x,470),(x,452)])
        readout=['c60','c62','c64'][i]
        route(f'c{61+i*2}','c54',readout,[(x,390),(x,376)])
        endpoint=[208,378,548][i]
        route(f'c{89+i}',readout,'c66',[(x,280),(x,275),(endpoint,275),(endpoint,270)])
    route('c56','c55','c54',[(770,421),(750,421)])
    route('static_input','origin_input','c92',[(320,140),(360,140)])
    route('c188','c92','c184',[(860,140),(912,140),(912,313),(1030,313)])
    route('c166','c66','c95',[(690,241),(895,241),(895,700),(950,700)])
    route('c168','c167','c95',[(1365,835),(1365,780)])
    route('c170','c95','c169',[(1220,640),(1220,619)])
    route('c178','c169','c173',[(1175,585),(1175,551)])
    route('c179','c173','c174',[(1175,515),(1175,503)])
    route('c180','c174','c175',[(1175,483),(1175,474)])
    route('c181','c175','c176',[(1175,450),(1175,438)])
    route('c185','c171','c184',[(1175,370),(1175,350),(1154,350),(1154,336)])
    route('c186','c182','c184',[(1490,500),(1497,500),(1497,350),(1460,350),(1460,336)])
    route('c187','c183','c184',[(1355,417),(1348,417),(1348,350),(1375,350),(1375,336)])
    route('c191','c184','c189',[(1415,282),(1415,240)])
    route('c192','c184','c190',[(1120,282),(1120,240)])
    route('c195','c189','c193',[(1415,200),(1415,160)])
    route('c196','c190','c194',[(1120,200),(1120,160)])
    route('c197','c194','feedback_join',[(1120,120),(1120,90),(1505,90),(1505,137)],'#b85450',True)
    route('c199','c193','feedback_join',[(1490,140),(1502,140)],'#b85450',True)
    route('feedback_state','feedback_join','c95',[(1505,143),(1505,700),(1490,700)],'#b85450',True)
    route('train_a','stage_a','stage_freeze',[(355,1195),(405,1195)])
    route('train_b','stage_freeze','stage_b',[(765,1195),(815,1195)])
    route('teacher_loss','teacher','stage_b',[(1225,1195),(1175,1195)],'#a68a37',True)
    assert {c.get('id') for c in root if c.get('edge')=='1'} == {p['id'] for p in paths}
    assert original_ids <= set(cells), 'All original node/edge identifiers must remain'
    # 实际增大逻辑画布、节点与留白，不以单纯提高 PNG 像素代替排版放大。
    for cell in cells.values():
        geometry = cell.find('mxGeometry')
        if geometry is not None:
            for field in ('x','y','width','height'):
                if field in geometry.attrib:
                    geometry.set(field,str(float(geometry.get(field,'0'))*GEOMETRY_SCALE))
            for point in geometry.iter('mxPoint'):
                for field in ('x','y'):
                    point.set(field,str(float(point.get(field,'0'))*GEOMETRY_SCALE))
        values = style_dict(cell)
        if cell.get('vertex')=='1':
            set_style(cell,fontSize=float(values.get('fontSize',11))*FONT_SCALE,
                      spacing=2*GEOMETRY_SCALE)
        set_style(cell,strokeWidth=float(values.get('strokeWidth',1))*GEOMETRY_SCALE)
    for edge in paths:
        edge['points']=[(x*GEOMETRY_SCALE,y*GEOMETRY_SCALE) for x,y in edge['points']]
    tree.write(OUT / (NAME+'.drawio'),encoding='utf-8',xml_declaration=True)

    containers={'c3','c94','c7','c23','c39','c66','c74','c81','c88','c95','c118','c141','c164','c171','training_panel'}
    nodes=[]
    for cell in cells.values():
        if cell.get('vertex')!='1':
            continue
        values=style_dict(cell)
        text=cell.get('value','')
        if '<' in text:
            text=html.unescape(re.sub('<[^>]+>','',text))
        x,y,w,h=geom(cell)
        nodes.append(dict(id=cell.get('id'),x=x,y=y,w=w,h=h,text=text,style=values,container=cell.get('id') in containers))
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><defs>']
    for color in ('#435465','#b85450','#a68a37'):
        svg.append(f'<marker id="a{color[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10Z" fill="{color}"/></marker>')
    svg.append(f'</defs><rect width="{W}" height="{H}" fill="white"/><g font-family="{FONT}">')

    def draw_node(n):
        s=n['style']; x,y,w,h=[n[k] for k in ('x','y','w','h')]
        fill=s.get('fillColor','none'); stroke=s.get('strokeColor','none')
        is_text='text;' in cells[n['id']].get('style','') or (fill=='none' and stroke=='none')
        if not is_text:
            dash=' stroke-dasharray="14 10"' if s.get('dashed')=='1' else ''
            radius=24 if n['id'] in ('c3','c94') else 8 if s.get('rounded')=='1' else 0
            svg.append(f'<rect data-node="{n["id"]}" x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{s.get("strokeWidth",1)}"{dash}/>')
        if not n['text'].strip():
            return
        size=float(s.get('fontSize',11)); lines=wrap(n['text'],w-28,size); line_height=size*1.35
        top=s.get('verticalAlign')=='top'
        start=y+size+16 if top else y+(h-len(lines)*line_height)/2+size
        if size + (len(lines)-1)*line_height > h-2:
            raise ValueError(f'Text too tall: {n["id"]}: {lines}')
        align=s.get('align','center'); tx=x+16 if align=='left' else x+w/2
        anchor='start' if align=='left' else 'middle'
        weight='700' if s.get('fontStyle')=='1' or top else '400'
        for i,line in enumerate(lines):
            svg.append(f'<text data-owner="{n["id"]}" x="{tx}" y="{start+i*line_height}" text-anchor="{anchor}" font-size="{size}" font-weight="{weight}" fill="{s.get("fontColor","#263747")}">{html.escape(line)}</text>')
    for n in nodes:
        if n['container']:
            draw_node(n)
    for e in paths:
        d='M '+' L '.join(f'{x} {y}' for x,y in e['points'])
        dash=' stroke-dasharray="10 8"' if e['dashed'] else ''
        svg.append(f'<path data-edge="{e["id"]}" d="{d}" fill="none" stroke="{e["color"]}" stroke-width="2.4"{dash} marker-end="url(#a{e["color"][1:]})"/>')
    for n in nodes:
        if not n['container']:
            draw_node(n)
    svg.append('</g></svg>')
    (OUT/(NAME+'.svg')).write_text('\n'.join(svg))
    (OUT/'layout.json').write_text(json.dumps(dict(name=NAME,width=W,height=H,nodes=nodes,edges=paths),ensure_ascii=False,indent=2))
    assert all(sha(BASE/p)==h for p,h in preserved.items())
    print('Original XML copied and refined; all original IDs retained; preserved files:',len(preserved))


if __name__=='__main__':
    main()
