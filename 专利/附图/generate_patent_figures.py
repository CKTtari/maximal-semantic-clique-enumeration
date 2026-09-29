"""Regenerate the four Chinese patent figures for the MSC disclosure.

The figures use the running example and algorithms in the companion paper:
SHBE is the structural-host bounded enumeration method and MCRS is the
minimum-contribution canonical reverse search method.
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
FONT = Path(r"C:\Windows\Fonts\simhei.ttf")
BOLD_FONT = Path(r"C:\Windows\Fonts\simhei.ttf")
INK = "#000000"


def font(size: int, bold: bool = False):
    return ImageFont.truetype(str(BOLD_FONT if bold else FONT), size=size, index=0)


def canvas(size):
    return Image.new("RGB", size, "white"), size


def centered(draw, box, text, fnt, spacing=5):
    x0, y0, x1, y1 = box
    bbox = draw.multiline_textbbox((0, 0), text, font=fnt, spacing=spacing, align="center")
    draw.multiline_text(((x0 + x1) / 2 - (bbox[2] - bbox[0]) / 2,
                         (y0 + y1) / 2 - (bbox[3] - bbox[1]) / 2),
                        text, font=fnt, fill=INK, spacing=spacing, align="center")


def title(draw, width, text):
    draw.text((width / 2, 50), text, font=font(45, True), fill=INK, anchor="ma")


def box(draw, xy, text, size=29):
    draw.rounded_rectangle(xy, radius=14, fill="white", outline=INK, width=3)
    centered(draw, xy, text, font(size))


def dashed_rectangle(draw, xy, dash=14, gap=8, width=3):
    x0, y0, x1, y1 = xy
    segments = [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)),
                ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]
    for (sx, sy), (ex, ey) in segments:
        length = abs(ex - sx) + abs(ey - sy)
        step = dash + gap
        for offset in range(0, length, step):
            end = min(offset + dash, length)
            if sx == ex:
                sign = 1 if ey >= sy else -1
                draw.line([(sx, sy + sign * offset), (sx, sy + sign * end)], fill=INK, width=width)
            else:
                sign = 1 if ex >= sx else -1
                draw.line([(sx + sign * offset, sy), (sx + sign * end, sy)], fill=INK, width=width)


def dashed_line(draw, start, end, dash=12, gap=8, width=3):
    x0, y0 = start
    x1, y1 = end
    if x0 == x1:
        sign = 1 if y1 >= y0 else -1
        for offset in range(0, abs(y1 - y0), dash + gap):
            draw.line([(x0, y0 + sign * offset),
                       (x0, y0 + sign * min(offset + dash, abs(y1 - y0)))], fill=INK, width=width)
    elif y0 == y1:
        sign = 1 if x1 >= x0 else -1
        for offset in range(0, abs(x1 - x0), dash + gap):
            draw.line([(x0 + sign * offset, y0),
                       (x0 + sign * min(offset + dash, abs(x1 - x0)), y0)], fill=INK, width=width)


def arrow(draw, points, label=None):
    """Draw an orthogonal polyline with an arrow at its final point."""
    draw.line(points, fill=INK, width=4, joint="curve")
    (x0, y0), (x1, y1) = points[-2], points[-1]
    if x1 > x0:
        head = [(x1, y1), (x1 - 20, y1 - 11), (x1 - 20, y1 + 11)]
    elif x1 < x0:
        head = [(x1, y1), (x1 + 20, y1 - 11), (x1 + 20, y1 + 11)]
    elif y1 > y0:
        head = [(x1, y1), (x1 - 11, y1 - 20), (x1 + 11, y1 - 20)]
    else:
        head = [(x1, y1), (x1 - 11, y1 + 20), (x1 + 11, y1 + 20)]
    draw.polygon(head, fill=INK)
    if label:
        lengths = [abs(b[0] - a[0]) + abs(b[1] - a[1]) for a, b in zip(points, points[1:])]
        target = sum(lengths) / 2
        passed = 0
        mx, my = points[0]
        for (ax, ay), (bx, by), length in zip(points, points[1:], lengths):
            if passed + length >= target:
                ratio = (target - passed) / length if length else 0
                mx = ax + (bx - ax) * ratio
                my = ay + (by - ay) * ratio
                break
            passed += length
        draw.text((mx, my - 24), label, font=font(24), fill=INK, anchor="mm")


def node(draw, pos, label):
    x, y = pos
    draw.ellipse((x - 45, y - 45, x + 45, y + 45), fill="white", outline=INK, width=4)
    draw.text((x, y), label, font=font(29), fill=INK, anchor="mm")


def figure1(path):
    im, (w, h) = canvas((2000, 1380)); d = ImageDraw.Draw(im)
    title(d, w, "图1  运行示例：结构极大团与极大语义团的差异")
    d.text((80, 145), "（a）原始向量属性图与结构极大团", font=font(31, True), fill=INK)
    pos = {1: (320, 350), 2: (320, 560), 3: (550, 350), 4: (550, 560), 5: (780, 350), 6: (780, 560)}
    d.ellipse((180, 240, 660, 670), outline=INK, width=3)
    d.ellipse((440, 240, 920, 670), outline=INK, width=3)
    d.text((420, 215), "M1={v1,v2,v3,v4}", font=font(26), fill=INK, anchor="mm")
    d.text((680, 215), "M2={v3,v4,v5,v6}", font=font(26), fill=INK, anchor="mm")
    edges = {(1, 2): "0.80", (1, 3): "0.60", (1, 4): "0.80", (2, 3): "0.48", (2, 4): "1.00", (3, 4): "0.48", (3, 5): "0.64", (3, 6): "0.48", (4, 5): "0.00", (4, 6): "0.00", (5, 6): "0.96"}
    offsets = {(1, 2): (-38, 0), (1, 3): (0, -31), (1, 4): (-70, -45), (2, 3): (-65, 45), (2, 4): (0, 34), (3, 4): (32, 0), (3, 5): (0, -31), (3, 6): (-75, -38), (4, 5): (75, 38), (4, 6): (0, 34), (5, 6): (32, 0)}
    for (u, v), value in edges.items():
        d.line([pos[u], pos[v]], fill=INK, width=3)
        mx, my = (pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2
        ox, oy = offsets[(u, v)]
        d.text((mx + ox, my + oy), value, font=font(25), fill=INK, anchor="mm")
    for index, point in pos.items():
        node(d, point, f"v{index}")
    d.text((420, 755), "M1：0.693 ≥ τ", font=font(26), fill=INK, anchor="mm")
    d.text((680, 755), "M2：0.427 < τ", font=font(26), fill=INK, anchor="mm")
    d.text((1100, 145), "（b）节点向量", font=font(31, True), fill=INK)
    box(d, (1100, 190, 1820, 765), "v1=(1.0, 0.0, 0.0, 0.0)\n\nv2=(0.8, 0.6, 0.0, 0.0)\n\nv3=(0.6, 0.0, 0.8, 0.0)\n\nv4=(0.8, 0.6, 0.0, 0.0)\n\nv5=(0.0, 0.0, 0.8, 0.6)\n\nv6=(0.0, 0.0, 0.6, 0.8)", 35)
    d.line([(80, 860), (1920, 860)], fill=INK, width=2)
    d.text((80, 905), "（c）平均语义约束下的极大语义团输出（τ=0.6）", font=font(31, True), fill=INK)
    box(d, (180, 970, 900, 1200), "C1=M1={v1,v2,v3,v4}\n\nS(C1)=4.16，平均相似度=4.16/6=0.693≥0.6\n\nC1为极大语义团", 28)
    box(d, (1100, 970, 1820, 1200), "C2={v3,v5,v6}\n\nS(C2)=2.08，平均相似度=2.08/3=0.693≥0.6\n\nC2严格包含于M2，且C2为极大语义团", 28)
    im.save(path)


def figure2(path):
    im, (w, h) = canvas((1800, 1260)); d = ImageDraw.Draw(im)
    title(d, w, "图2  结构约束语义群组检索系统架构图")
    box(d, (110, 160, 510, 290), "图与向量接入模块\n图快照和节点向量", 27)
    box(d, (700, 160, 1100, 290), "预处理模块\n图规范化、向量归一化\n边相似度缓存", 24)
    box(d, (1290, 160, 1690, 290), "算法选择模块\n阈值配置\n选择 SHBE 或 MCRS", 24)
    arrow(d, [(510, 225), (700, 225)])
    arrow(d, [(1100, 225), (1290, 225)])
    box(d, (650, 410, 1150, 510), "搜索任务分发", 27)
    arrow(d, [(1490, 290), (1490, 350), (900, 350), (900, 410)])
    box(d, (520, 620, 1280, 760), "自适应搜索引擎\nSHBE：结构宿主和有界语义子集枚举\nMCRS：最小贡献节点和规范反向搜索", 25)
    arrow(d, [(900, 510), (900, 620)])
    box(d, (550, 870, 1250, 1010), "统一结果模块\n去重、严格包含过滤、群组索引和解释元数据", 26)
    arrow(d, [(900, 760), (900, 870)])
    box(d, (550, 1120, 1250, 1230), "GraphRAG、文献发现、风控图谱和图数据库接口", 24)
    arrow(d, [(900, 1010), (900, 1120)])
    im.save(path)


def _complete_graph(draw, positions, labels, edges=None):
    keys = list(positions)
    pairs = edges if edges is not None else [(keys[i], keys[j]) for i in range(len(keys)) for j in range(i + 1, len(keys))]
    for u, v in pairs:
        draw.line([positions[u], positions[v]], fill=INK, width=3)
    for key, pos in positions.items():
        node(draw, pos, labels[key])


def figure3(path):
    """Chinese patent adaptation of the paper's StrSub running-example figure."""
    im, (w, h) = canvas((1800, 1280)); d = ImageDraw.Draw(im)
    title(d, w, "图3  结构宿主有界枚举方法（SHBE）运行示例")
    d.text((80, 120), "（a）结构极大团宿主", font=font(30, True), fill=INK)

    pos = {1: (600, 300), 2: (600, 500), 3: (900, 300), 4: (900, 500), 5: (1200, 300), 6: (1200, 500)}
    d.ellipse((460, 205, 1040, 610), outline=INK, width=3)
    d.ellipse((760, 205, 1340, 610), outline=INK, width=3)
    d.text((750, 180), "M1={v1,v2,v3,v4}", font=font(25), fill=INK, anchor="mm")
    d.text((1050, 180), "M2={v3,v4,v5,v6}", font=font(25), fill=INK, anchor="mm")
    _complete_graph(draw=d, positions={k: pos[k] for k in (1, 2, 3, 4)}, labels={k: f"v{k}" for k in (1, 2, 3, 4)})
    _complete_graph(draw=d, positions={k: pos[k] for k in (3, 4, 5, 6)}, labels={k: f"v{k}" for k in (3, 4, 5, 6)})
    d.text((750, 670), "M1：0.693≥τ", font=font(24), fill=INK, anchor="mm")
    d.text((1050, 670), "M2：0.427<τ", font=font(24), fill=INK, anchor="mm")

    d.line([(80, 755), (1720, 755)], fill=INK, width=2)
    d.text((80, 795), "（b）宿主内语义处理与结果归并", font=font(30, True), fill=INK)
    box(d, (150, 860, 510, 970), "M1 可行\n形成候选 C1=M1", 27)
    box(d, (150, 1035, 510, 1145), "M2 不可行\n进入宿主内有界搜索", 27)
    box(d, (650, 860, 1040, 970), "候选 C1\n平均相似度=0.693≥τ", 26)
    box(d, (650, 1035, 1040, 1145), "排序边上界、节点贡献上界\n得到候选 C2={v3,v5,v6}", 25)
    box(d, (1210, 860, 1670, 1145), "全局去重与\n严格包含过滤\n\n输出 C1 和 C2", 29)
    arrow(d, [(510, 915), (650, 915)])
    arrow(d, [(510, 1090), (650, 1090)])
    arrow(d, [(1040, 915), (1210, 915)])
    arrow(d, [(1040, 1090), (1210, 1090)])
    im.save(path)


def figure4(path):
    """Chinese patent adaptation of the paper's canonical-ownership example."""
    im, (w, h) = canvas((1800, 1340)); d = ImageDraw.Draw(im)
    title(d, w, "图4  最小贡献规范反向搜索方法（MCRS）运行示例")
    box(d, (360, 115, 1440, 215), "对可行子团 D，仅当插入节点等于 D 的最小贡献节点 d(D) 时，接受该插入")

    d.text((90, 280), "（a）C1 的规范构造路径", font=font(30, True), fill=INK)
    edge1 = {2: (220, 460), 4: (400, 460)}
    tri1 = {1: (760, 344), 2: (670, 500), 4: (850, 500)}
    clique1 = {1: (1240, 330), 2: (1240, 510), 3: (1420, 330), 4: (1420, 510)}
    _complete_graph(d, edge1, {2: "v2", 4: "v4"}, [(2, 4)])
    _complete_graph(d, tri1, {1: "v1", 2: "v2", 4: "v4"})
    _complete_graph(d, clique1, {1: "v1", 2: "v2", 3: "v3", 4: "v4"})
    arrow(d, [(450, 460), (570, 460)], "插入 v1")
    arrow(d, [(910, 460), (1080, 460)], "插入 v3")
    d.text((320, 570), "平均相似度=1.000", font=font(25), fill=INK, anchor="mm")
    d.text((760, 610), "d({v1,v2,v4})=v1；平均相似度=0.867", font=font(22), fill=INK, anchor="mm")
    d.text((1330, 610), "d(C1)=v3", font=font(22), fill=INK, anchor="mm")
    box(d, (1130, 570, 1550, 650), "终止候选 C1；平均相似度=0.693", 22)

    d.line([(80, 675), (1720, 675)], fill=INK, width=2)
    d.text((90, 720), "（b）C2 的规范构造与重复路径", font=font(30, True), fill=INK)
    edge2 = {5: (100, 900), 6: (280, 900)}
    clique2 = {3: (620, 790), 5: (530, 946), 6: (710, 946)}
    edge3 = {3: (930, 900), 5: (1110, 900)}
    clique3 = {3: (1500, 790), 5: (1410, 946), 6: (1590, 946)}
    _complete_graph(d, edge2, {5: "v5", 6: "v6"}, [(5, 6)])
    _complete_graph(d, clique2, {3: "v3", 5: "v5", 6: "v6"})
    _complete_graph(d, edge3, {3: "v3", 5: "v5"}, [(3, 5)])
    _complete_graph(d, clique3, {3: "v3", 5: "v5", 6: "v6"})
    arrow(d, [(330, 900), (460, 900)], "插入 v3")
    arrow(d, [(1165, 900), (1315, 900)], "插入 v6")
    d.text((350, 1015), "平均相似度=0.960", font=font(25), fill=INK, anchor="mm")
    d.text((620, 1070), "d(C2)=v3；平均相似度=0.693", font=font(22), fill=INK, anchor="mm")
    box(d, (410, 1110, 830, 1205), "终止候选 C2\n平均相似度=0.693", 25)
    d.text((1020, 1015), "平均相似度=0.640", font=font(24), fill=INK, anchor="mm")
    dashed_rectangle(d, (1340, 720, 1690, 1040), width=3)
    d.text((1515, 1080), "d(C2)=v3，插入节点为v6\n重复构造在此停止", font=font(22), fill=INK, anchor="mm")
    d.text((1660, 745), "×", font=font(46, True), fill=INK, anchor="mm")
    im.save(path)


def generate_all():
    figure1(HERE / "图1_最大语义团定义示意.png")
    figure2(HERE / "图2_系统架构图.png")
    figure3(HERE / "图3_SHBE运行示例图.png")
    figure4(HERE / "图4_MCRS运行示例图.png")


if __name__ == "__main__":
    generate_all()
