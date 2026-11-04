"""Build every hero asset and the room shell for the Backrun studio, one GLB per family.

Run:  blender -b --factory-startup --python blender/build_all.py [-- name name ...]
Writes raw GLBs to blender/out/ (scripts/compress.mjs compresses them into public/models/)
and blender/build_report.json with triangle counts and empties.

Units are metres. Blender axes: X east, Y north (into the room, away from the visitor),
Z up. Every asset's front faces Blender -Y, which the writer turns into three.js +Z.
"""

import bpy
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lib  # noqa: E402
from lib import box, cyl, lathe, tube, prism, sphere, empty, group, mat, boolean  # noqa: E402

OUT = os.path.join(HERE, 'out')
os.makedirs(OUT, exist_ok=True)
SVG = os.path.normpath(os.path.join(HERE, '..', '..', 'docs', 'brand', 'sepia.svg'))
R = math.radians

PIGMENTS = [
    ('quinacridone_rose', '#B8325F'), ('cadmium_yellow', '#E8B820'), ('french_ultramarine', '#2E3C8C'),
    ('cerulean_blue', '#2F7DB0'), ('burnt_umber', '#4E3524'), ('hookers_green', '#2C5A30'),
    ('indian_red', '#8C3A30'), ('paynes_grey', '#3A4250'),
]


def palette_mats():
    M = {}
    M['wood'] = mat('Wood', '#8B7355', 0.7)
    M['wood_worn'] = mat('Wood_BenchTop', '#9A8062', 0.72)
    M['wood_dark'] = mat('Wood_Walnut', '#5A4433', 0.55)
    M['oak_floor'] = mat('Floor_Oak', '#6A5440', 0.6)
    M['beech'] = mat('Wood_Beech', '#C8A27A', 0.6)
    M['brass'] = mat('Brass', '#B5A642', 0.32, 1.0)
    M['brass_dark'] = mat('Brass_Aged', '#8E7F3C', 0.45, 1.0)
    M['iron'] = mat('Iron', '#3B3732', 0.5, 0.8)
    M['nickel'] = mat('Nickel', '#C9C5BA', 0.28, 1.0)
    M['glass'] = mat('Glass', '#E8E4D9', 0.05, 0.0, alpha=0.22)
    M['water'] = mat('Water_Clean', '#CFDCDC', 0.03, 0.0, alpha=0.38)
    M['plaster'] = mat('Plaster', '#F5F2EB', 0.95)
    M['brick'] = mat('Brick_Limewash', '#ECE4D6', 0.92)
    M['paper'] = mat('Paper', '#F4F0E6', 0.9)
    M['porcelain'] = mat('Porcelain', '#F7F5EF', 0.16)
    M['kraft'] = mat('Kraft_Tape', '#B89A6A', 0.8)
    M['carrier'] = mat('Carrier_Board', '#C9B99C', 0.8)
    M['cork'] = mat('Cork', '#A9825A', 0.95)
    M['frame_paint'] = mat('Window_Paint', '#CDC5B3', 0.6)
    M['fabric'] = mat('Blind_Fabric', '#E6DCC8', 0.95, double=True)
    M['rope'] = mat('Rope', '#B9A077', 0.9)
    M['terracotta'] = mat('Terracotta', '#A85F3C', 0.85)
    M['leaf'] = mat('Leaf', '#5D7A45', 0.6, double=True)
    M['soil'] = mat('Soil', '#3A2C22', 1.0)
    M['ceramic'] = mat('Ceramic_Sage', '#7F918A', 0.3)
    M['cloth'] = mat('Rag_Cloth', '#D8CDB8', 1.0)
    M['ink'] = mat('Book_Cover', '#4A3A30', 0.7)
    M['book_green'] = mat('Book_Cover_Green', '#3F4D3E', 0.7)
    M['amber'] = mat('Glass_Amber', '#7A4A1E', 0.08, alpha=0.75)
    M['cap'] = mat('Cap_Black', '#26231F', 0.5)
    return M


def finish(name, extras=None):
    path = os.path.join(OUT, name + '.glb')
    tris, size = lib.export_glb(path, extras)
    empties = {}
    for o in bpy.context.scene.objects:
        if o.get('attach'):
            w = o.matrix_world.translation
            empties[o.name] = [round(w.x, 4), round(w.z, 4), round(-w.y, 4)]
    REPORT[name] = dict(tris=tris, raw_bytes=size, empties=empties)


# =============================================================== room shell

def build_room():
    lib.reset()
    M = palette_mats()
    W, D, H, T = 5.0, 4.0, 2.8, 0.12   # inner width (x), depth (y), height, wall thickness
    root = group('Room')
    f = box('Floor', (W + 2 * T, D + 2 * T, 0.04), (0, 0, -0.02), M['oak_floor'], bev=0, parent=root)
    f['uv_scale'] = 1.0

    def wall_pieces(name, pieces, m, parent):
        g = group(name, parent=parent)
        for i, (sz, loc) in enumerate(pieces):
            box(f'{name}_P{i}', sz, loc, m, bev=0.004, parent=g, segs=1)
        return g

    # north wall: limewashed brick behind the board (inner face y = +2.0)
    yN = D / 2 + T / 2
    wall_pieces('Wall_N', [((W + 2 * T, T, H), (0, yN, H / 2))], M['brick'], root)
    wall_pieces('Wall_N_Stub', [((W + 2 * T, T, 0.3), (0, yN, 0.15))], M['brick'], root)
    # south and east walls: limewash plaster
    yS = -D / 2 - T / 2
    wall_pieces('Wall_S', [((W + 2 * T, T, H), (0, yS, H / 2))], M['plaster'], root)
    wall_pieces('Wall_S_Stub', [((W + 2 * T, T, 0.3), (0, yS, 0.15))], M['plaster'], root)
    xE = W / 2 + T / 2
    wall_pieces('Wall_E', [((T, D, H), (xE, 0, H / 2))], M['plaster'], root)
    wall_pieces('Wall_E_Stub', [((T, D, 0.3), (xE, 0, 0.15))], M['plaster'], root)
    # west wall with the window: 1.2 wide x 1.8 tall, sill at 0.9, centred at y = 0.8
    xW = -W / 2 - T / 2
    wy0, wy1, wz0, wz1 = 0.2, 1.4, 0.9, 2.7
    wall_pieces('Wall_W', [
        ((T, wy0 + D / 2, H), (xW, (-D / 2 + wy0) / 2, H / 2)),
        ((T, D / 2 - wy1, H), (xW, (wy1 + D / 2) / 2, H / 2)),
        ((T, wy1 - wy0, wz0), (xW, (wy0 + wy1) / 2, wz0 / 2)),
        ((T, wy1 - wy0, H - wz1), (xW, (wy0 + wy1) / 2, (wz1 + H) / 2)),
    ], M['plaster'], root)
    wall_pieces('Wall_W_Stub', [((T, D, 0.3), (xW, 0, 0.15))], M['plaster'], root)

    # skirting boards, always visible (they sit inside the stub height)
    sk = group('Skirting', parent=root)
    sh, st = 0.1, 0.016
    box('Skirting_N', (W, st, sh), (0, D / 2 - st / 2, sh / 2), M['wood_dark'], 0.003, parent=sk)
    box('Skirting_S', (W, st, sh), (0, -D / 2 + st / 2, sh / 2), M['wood_dark'], 0.003, parent=sk)
    box('Skirting_E', (st, D - 2 * st, sh), (W / 2 - st / 2, 0, sh / 2), M['wood_dark'], 0.003, parent=sk)
    box('Skirting_W', (st, D - 2 * st, sh), (-W / 2 + st / 2, 0, sh / 2), M['wood_dark'], 0.003, parent=sk)

    # window: casing, sash frame with muntins (2 x 3 panes), single glass pane, inner sill, roller blind
    win = group('Window', (xW, (wy0 + wy1) / 2, wz0), parent=root)
    ww, wh = wy1 - wy0, wz1 - wz0
    fw, fd = 0.06, 0.07
    fr = group('Window_Frame', parent=win)
    for nm, sz, loc in [
        ('Window_Frame_Head', (fd, ww, fw), (0, 0, wh - fw / 2)),
        ('Window_Frame_Foot', (fd, ww, fw), (0, 0, fw / 2)),
        ('Window_Frame_L', (fd, fw, wh - 2 * fw), (0, -ww / 2 + fw / 2, wh / 2)),
        ('Window_Frame_R', (fd, fw, wh - 2 * fw), (0, ww / 2 - fw / 2, wh / 2)),
        ('Window_Muntin_V', (fd * 0.6, 0.026, wh - 2 * fw), (0, 0, wh / 2)),
        ('Window_Muntin_H1', (fd * 0.6, ww - 2 * fw, 0.026), (0, 0, fw + (wh - 2 * fw) / 3)),
        ('Window_Muntin_H2', (fd * 0.6, ww - 2 * fw, 0.026), (0, 0, fw + 2 * (wh - 2 * fw) / 3)),
    ]:
        box(nm, sz, loc, M['frame_paint'], 0.004, parent=fr, segs=1)
    # reveal linings so the wall thickness reads at the opening
    for nm, sz, loc in [
        ('Window_Reveal_Top', (T, ww, 0.01), (0, 0, wh - 0.005)),
        ('Window_Reveal_L', (T, 0.01, wh), (0, -ww / 2 + 0.005, wh / 2)),
        ('Window_Reveal_R', (T, 0.01, wh), (0, ww / 2 - 0.005, wh / 2)),
    ]:
        box(nm, sz, loc, M['plaster'], 0, parent=fr)
    box('Window_Glass', (0.004, ww - 2 * fw, wh - 2 * fw), (0.0, 0, wh / 2), M['glass'], 0, parent=win)
    box('Window_Sill', (0.20, ww + 0.12, 0.03), (T / 2 + 0.03, 0, -0.015), M['wood'], 0.004, parent=win)
    empty('Window_SillTop', (T / 2 + 0.06, 0, 0.0), parent=win)
    # roller blind, lowered about a third, mounted inside the reveal head
    bl = group('Window_Blind', (T / 2 - 0.02, 0, wh - 0.04), parent=win)
    cyl('Window_Blind_Roller', 0.02, ww - 0.04, (0, -(ww - 0.04) / 2, 0), M['frame_paint'], n=12,
        rot=(R(-90), 0, 0), parent=bl)
    box('Window_Blind_Fabric', (0.002, ww - 0.08, 0.50), (0.02, 0, -0.27), M['fabric'], 0, parent=bl)
    box('Window_Blind_Bar', (0.012, ww - 0.07, 0.02), (0.02, 0, -0.53), M['wood'], 0.003, parent=bl)
    cyl('Window_Blind_Pull', 0.008, 0.03, (0.02, 0, -0.575), M['wood'], n=8, parent=bl)

    # ceiling light rose is omitted: the room has no ceiling in the diorama cut
    empty('Room_SunTarget', (0, 1.4, 0.76), parent=root)
    finish('room', extras=dict(inner=[W, D, H], wall=T, window=dict(y=[wy0, wy1], z=[wz0, wz1])))


# =============================================================== bench (work table) with paper drawer

def build_bench():
    lib.reset()
    M = palette_mats()
    L, Wd, Ht, top = 1.80, 0.80, 0.76, 0.045
    root = group('Bench')
    t = box('Bench_Top', (L, Wd, top), (0, 0, Ht - top / 2), M['wood_worn'], 0.004, parent=root)
    t['uv_scale'] = 1.0
    leg, inset = 0.07, 0.05
    for i, (sx, sy) in enumerate([(-1, -1), (1, -1), (1, 1), (-1, 1)]):
        box(f'Bench_Leg_{i}', (leg, leg, Ht - top), (sx * (L / 2 - inset - leg / 2), sy * (Wd / 2 - inset - leg / 2),
                                                     (Ht - top) / 2), M['wood'], 0.003, parent=root)
    ah = 0.10
    az = Ht - top - ah / 2
    box('Bench_Apron_Back', (L - 2 * inset - 2 * leg, 0.022, ah), (0, Wd / 2 - inset - 0.011, az), M['wood'], 0.002,
        parent=root)
    for sx in (-1, 1):
        box(f'Bench_Apron_Side_{sx}', (0.022, Wd - 2 * inset - 2 * leg, ah), (sx * (L / 2 - inset - 0.011), 0, az),
            M['wood'], 0.002, parent=root)
    # front apron split around the paper drawer
    fx = -Wd / 2 + inset + 0.011
    dw = 0.62
    side = (L - 2 * inset - 2 * leg - dw) / 2
    for sx in (-1, 1):
        box(f'Bench_Apron_Front_{sx}', (side, 0.022, ah), (sx * (dw / 2 + side / 2), fx, az), M['wood'], 0.002,
            parent=root)
    drawer = group('Bench_Drawer', (0, fx, az), parent=root)
    box('Bench_Drawer_Front', (dw - 0.006, 0.022, ah - 0.008), (0, -0.002, 0), M['wood'], 0.003, parent=drawer)
    box('Bench_Drawer_Pull', (0.10, 0.012, 0.014), (0, -0.02, 0), M['brass_dark'], 0.003, parent=drawer)
    for sx in (-1, 1):
        cyl(f'Bench_Drawer_PullPost_{sx}', 0.004, 0.012, (sx * 0.045, -0.013, 0), M['brass_dark'], n=8,
            rot=(R(90), 0, 0), parent=drawer)
    empty('Bench_Drawer_Paper', (0, 0.30, -0.02), parent=drawer)
    # lower shelf on stretchers, where the paper boxes live
    sz = 0.14
    for sy in (-1, 1):
        box(f'Bench_Stretcher_{sy}', (L - 2 * inset - 2 * leg, 0.04, 0.05),
            (0, sy * (Wd / 2 - inset - leg / 2), sz), M['wood'], 0.002, parent=root)
    for i in range(7):
        x = -0.72 + i * 0.24
        box(f'Bench_ShelfSlat_{i}', (0.20, Wd - 2 * inset - 0.02, 0.018), (x, 0, sz + 0.034), M['wood'], 0.002,
            parent=root)
    # wear: paint drips on the front edge, a couple of dried colour spots, water rings
    drips = [(-0.55, '#2F7DB0', 0.03), (-0.18, '#B8325F', 0.02), (0.31, '#E8B820', 0.045), (0.62, '#2C5A30', 0.025)]
    for i, (x, col, ln) in enumerate(drips):
        m = mat(f'Paint_Drip_{i}', col, 0.35)
        box(f'Bench_Drip_{i}', (0.006, 0.0012, ln), (x, -Wd / 2 - 0.0006, Ht - ln / 2), m, 0.0005, parent=root,
            segs=1)
        sphere(f'Bench_DripBead_{i}', 0.0035, (x, -Wd / 2 - 0.001, Ht - ln), m, seg=8, rings=4,
               scale=(1, 0.5, 1.2), parent=root)
    spots = [(0.48, 0.12, '#2E3C8C', 0.018), (-0.62, -0.20, '#8C3A30', 0.012), (0.70, -0.26, '#E8B820', 0.01)]
    for i, (x, y, col, r) in enumerate(spots):
        cyl(f'Bench_PaintSpot_{i}', r, 0.0004, (x, y, Ht), mat(f'Paint_Spot_{i}', col, 0.4), n=10, parent=root)
    ring = mat('Water_Ring', '#7C6650', 0.6, alpha=0.45)
    for i, (x, y, r) in enumerate([(0.74, 0.18, 0.046), (0.58, 0.27, 0.044), (-0.70, 0.26, 0.04)]):
        lathe(f'Bench_WaterRing_{i}', [(r - 0.0025, Ht + 0.0004), (r, Ht + 0.0004)], n=28, m=ring, loc=(x, y, 0),
              parent=root)
    empty('Bench_TopCentre', (0, 0, Ht), parent=root)
    finish('bench')


# =============================================================== drawing board, prop and painting panel

def build_board():
    lib.reset()
    M = palette_mats()
    root = group('Board')
    BW, BD, BT, TILT = 0.60, 0.45, 0.018, 12.0
    tilt = group('Board_Tilt', parent=root)
    tilt.rotation_euler = (R(TILT), 0, 0)
    box('Board_Slab', (BW, BD, BT), (0, BD / 2, BT / 2), M['wood'], 0.003, parent=tilt)
    box('Board_Lip', (BW, 0.018, 0.012), (0, 0.009, BT + 0.006), M['wood_dark'], 0.002, parent=tilt)
    # edge banding (breadboard ends) on left and right
    for sx in (-1, 1):
        box(f'Board_End_{sx}', (0.03, BD, BT + 0.001), (sx * (BW / 2 - 0.015), BD / 2, BT / 2), M['wood_dark'], 0.002,
            parent=tilt)
    # painting panel: carrier, sheet (0.5 mm above carrier), kraft tape (0.2 mm above sheet)
    panel = group('Panel', (0.0, BD / 2 + 0.01, BT), parent=tilt)
    CW, CD, CT = 0.36, 0.285, 0.004
    box('Panel_Carrier', (CW, CD, CT), (0, 0, 0.0005 + CT / 2), M['carrier'], 0.0012, parent=panel, segs=1)
    sz = 0.0005 + CT + 0.0005
    SW, SH = 0.305, 0.229
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, 0, sz))
    sheet = bpy.context.active_object
    sheet.name = 'Panel_Sheet'
    sheet.scale = (SW, SH, 1)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    sheet.data.materials.append(mat('Paper_Sheet', '#F4F0E6', 0.85))
    sheet['keep_uv'] = 1
    sheet.parent = panel
    tw, tt, ov = 0.019, 0.00015, 0.006
    tz = sz + 0.0002 + tt / 2
    for nm, size, loc in [
        ('Panel_Tape_Top', (SW + 2 * tw - 2 * ov, tw, tt), (0, SH / 2 - ov + tw / 2, tz)),
        ('Panel_Tape_Bottom', (SW + 2 * tw - 2 * ov, tw, tt), (0, -SH / 2 + ov - tw / 2, tz)),
        ('Panel_Tape_Left', (tw, SH - 2 * ov, tt), (-SW / 2 + ov - tw / 2, 0, tz + tt)),
        ('Panel_Tape_Right', (tw, SH - 2 * ov, tt), (SW / 2 - ov + tw / 2, 0, tz + tt)),
    ]:
        box(nm, size, loc, M['kraft'], 0, parent=panel)
    for nm, x, y in [('TL', -1, 1), ('TR', 1, 1), ('BL', -1, -1), ('BR', 1, -1)]:
        empty(f'Sheet_Corner_{nm}', (x * SW / 2, y * SH / 2, sz), parent=panel)
    empty('Sheet_Centre', (0, 0, sz), parent=panel)
    empty('Board_PanelHome', (0.0, BD / 2 + 0.01, BT), parent=tilt)
    # hinged prop under the back edge plus two brass butt hinges
    py = 0.40
    pz = py * math.sin(R(TILT))
    box('Board_Prop', (0.50, 0.022, pz), (0, py * math.cos(R(TILT)) - 0.012, pz / 2), M['wood_dark'], 0.002,
        parent=root)
    for sx in (-1, 1):
        box(f'Board_Hinge_{sx}', (0.035, 0.012, 0.002), (sx * 0.18, py * math.cos(R(TILT)) - 0.006, pz + 0.001),
            M['brass'], 0.0005, parent=root, segs=1)
    finish('board', extras=dict(tilt_deg=TILT, sheet_m=[SW, SH], canvas_px=[256, 192], px_mm=SW * 1000 / 256))


# =============================================================== Sepia enamel pin

def build_badge(parent, width=0.028, loc=(0, 0, 0)):
    """Import sepia.svg, stack its shapes as enamel layers on a brass back, scale to width."""
    before = set(bpy.data.objects)
    bpy.ops.import_curve.svg(filepath=SVG)
    curves = [o for o in bpy.data.objects if o not in before and o.type == 'CURVE']
    curves.sort(key=lambda o: o.name)
    # strokes in SVG order: blot rim, pupil, six arm strokes (width px, colour)
    strokes = [(2.4, None), (4.5, '#1F2630'), (8, '#536878'), (8, '#536878'), (7, '#3F4A5C'), (6, '#3F4A5C'),
               (8, '#536878'), (6, '#536878')]
    px = 0.0254 / 90.0
    si = 0
    parts = []
    for li, o in enumerate(curves):
        cu = o.data
        cu.resolution_u = 4
        z = 0.0003 * li
        if cu.fill_mode == 'FULL' and not cu.materials:
            w, col = strokes[si]
            si += 1
            if col is None:
                bpy.data.objects.remove(o, do_unlink=True)
                continue
            cu.bevel_depth = w / 2 * px
            cu.bevel_resolution = 1
            cu.materials.append(mat('Enamel_' + col.strip('#'), col, 0.22))
            o.location.z = z + w / 2 * px * 0.3
            o.scale.z = 0.3
        else:
            c = cu.materials[0].diffuse_color
            hexc = '#%02X%02X%02X' % tuple(int(round(255 * (v * 12.92 if v <= 0.0031308 else 1.055 * v ** (1 / 2.4) - 0.055)))
                                           for v in c[:3])
            cu.materials.clear()
            cu.materials.append(mat('Enamel_' + hexc.strip('#'), hexc, 0.22))
            cu.extrude = 0.00015
            o.location.z = z
        parts.append(o)
    # brass back plate: a copy of the blot, a bit larger and thicker
    blot = parts[0]
    back = blot.copy()
    back.data = blot.data.copy()
    bpy.context.scene.collection.objects.link(back)
    back.data.materials.clear()
    back.data.materials.append(mat('Brass', '#B5A642', 0.32, 1.0))
    back.data.extrude = 0.0006
    back.location.z = -0.0007
    parts.append(back)
    badge = lib.join(parts, 'Arm_Badge')
    # centre and scale
    bpy.context.view_layer.objects.active = badge
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    dims = badge.dimensions.copy()
    s = width / dims.x
    badge.location = (0, 0, 0)
    badge.scale = (s, s, 1.0)
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=True)
    zmin = min(v.co.z for v in badge.data.vertices)
    for v in badge.data.vertices:
        v.co.z -= zmin
    badge.location = loc
    badge.parent = parent
    lib.smooth(badge, 50)
    return badge


# =============================================================== the five-joint desk arm

def build_arm():
    lib.reset()
    M = palette_mats()
    root = group('Arm')
    base = group('Arm_Base', parent=root)
    # base plate (brass, 130 mm), on a walnut pad, clamped over the bench's back edge
    box('Arm_BasePad', (0.15, 0.15, 0.012), (0, 0.01, 0.006), M['wood_dark'], 0.003, parent=base)
    cyl('Arm_BasePlate', 0.065, 0.008, (0, 0, 0.012), M['brass'], n=40, bev=0.0015, parent=base)
    for i in range(4):
        a = R(45 + 90 * i)
        cyl(f'Arm_BaseScrew_{i}', 0.0045, 0.002, (0.052 * math.cos(a), 0.052 * math.sin(a), 0.02), M['brass_dark'],
            n=10, bev=0.0006, parent=base)
    # clamp: bar back to the edge, jaw down behind the edge, foot under the top, screw and wing nut
    edge = 0.10
    box('Arm_ClampBar', (0.04, edge - 0.05 + 0.02, 0.012), (0, 0.05 + (edge - 0.05 + 0.02) / 2 - 0.005, 0.006),
        M['iron'], 0.002, parent=base)
    box('Arm_ClampJaw', (0.04, 0.014, 0.10), (0, edge + 0.007, -0.038), M['iron'], 0.002, parent=base)
    box('Arm_ClampFoot', (0.04, 0.06, 0.012), (0, edge - 0.023, -0.082), M['iron'], 0.002, parent=base)
    cyl('Arm_ClampScrew', 0.005, 0.05, (0, edge - 0.035, -0.12), M['iron'], n=10, parent=base)
    box('Arm_ClampWing', (0.045, 0.006, 0.01), (0, edge - 0.035, -0.12), M['iron'], 0.002, parent=base)
    cyl('Arm_ClampPad', 0.011, 0.004, (0, edge - 0.035, -0.074), M['iron'], n=12, parent=base)
    build_badge(base, 0.030, (0, -0.036, 0.020))
    empty('Arm_BadgeCentre', (0, -0.036, 0.021), parent=base)

    # J1 base yaw
    yaw = group('Arm_Yaw', (0, 0, 0.020), parent=base)
    cyl('Arm_Turntable', 0.03, 0.012, (0, 0, 0), M['brass'], n=32, bev=0.0015, parent=yaw)
    cyl('Arm_Column', 0.013, 0.06, (0, 0, 0.012), M['brass'], n=20, parent=yaw)
    box('Arm_ShoulderFork', (0.066, 0.036, 0.024), (0, 0, 0.078), M['wood_dark'], 0.003, parent=yaw)
    for sx in (-1, 1):
        box(f'Arm_ShoulderCheek_{sx}', (0.008, 0.04, 0.05), (sx * 0.029, 0, 0.10), M['wood_dark'], 0.002, parent=yaw)

    # J2 shoulder pitch: upper arm, twin brass tubes, wooden knuckle, tension spring
    UP, LO = 0.32, 0.32
    sh = group('Arm_Shoulder', (0, 0, 0.105), parent=yaw)
    cyl('Arm_ShoulderHub', 0.02, 0.05, (-0.025, 0, 0), M['wood'], n=20, rot=(0, R(90), 0), parent=sh)
    cyl('Arm_ShoulderAxle', 0.005, 0.07, (-0.035, 0, 0), M['brass'], n=10, rot=(0, R(90), 0), parent=sh)
    for sx in (-1, 1):
        cyl(f'Arm_UpperTube_{sx}', 0.0055, UP - 0.035, (sx * 0.016, 0, 0.015), M['brass'], n=12, parent=sh)
    box('Arm_UpperBrace', (0.04, 0.012, 0.016), (0, 0, UP * 0.55), M['wood'], 0.002, parent=sh)
    # spring hooks on the back (+Y) side and a helical spring between them
    sy = 0.026
    cyl('Arm_SpringAnchorLo', 0.003, 0.03, (0, sy - 0.012, 0.04), M['brass_dark'], n=8, rot=(R(-90), 0, 0), parent=sh)
    cyl('Arm_SpringAnchorHi', 0.003, 0.03, (0, sy - 0.012, 0.25), M['brass_dark'], n=8, rot=(R(-90), 0, 0), parent=sh)
    pts = []
    z0, z1, turns, r = 0.06, 0.23, 26, 0.0055
    nseg = turns * 8
    for i in range(nseg + 1):
        a = 2 * math.pi * turns * i / nseg
        pts.append((r * math.cos(a), sy + r * math.sin(a), z0 + (z1 - z0) * i / nseg))
    tube('Arm_Spring', [(0, sy, 0.043)] + pts + [(0, sy, 0.247)], 0.0011, M['nickel'], res=0, parent=sh)

    # J3 elbow pitch: lower arm
    el = group('Arm_Elbow', (0, 0, UP), parent=sh)
    cyl('Arm_ElbowKnuckle', 0.019, 0.05, (-0.025, 0, 0), M['wood'], n=20, rot=(0, R(90), 0), parent=el)
    cyl('Arm_ElbowAxle', 0.0045, 0.064, (-0.032, 0, 0), M['brass'], n=10, rot=(0, R(90), 0), parent=el)
    for sx in (-1, 1):
        cyl(f'Arm_LowerTube_{sx}', 0.0048, LO - 0.034, (sx * 0.0125, 0, 0.016), M['brass'], n=12, parent=el)
    cyl('Arm_LowerRod', 0.0018, LO - 0.06, (0, 0.016, 0.03), M['brass_dark'], n=6, parent=el)
    box('Arm_LowerBrace', (0.034, 0.012, 0.014), (0, 0, LO * 0.5), M['wood'], 0.002, parent=el)

    # J4 wrist pitch
    wr = group('Arm_Wrist', (0, 0, LO), parent=el)
    cyl('Arm_WristKnuckle', 0.016, 0.042, (-0.021, 0, 0), M['wood'], n=18, rot=(0, R(90), 0), parent=wr)
    cyl('Arm_WristAxle', 0.004, 0.052, (-0.026, 0, 0), M['brass'], n=10, rot=(0, R(90), 0), parent=wr)
    box('Arm_WristLink', (0.022, 0.02, 0.04), (0, 0, 0.032), M['brass'], 0.003, parent=wr)
    # side bracket: the collet sits beside the wrist so the brush handle runs clear of the arm
    box('Arm_WristBracket', (0.022, 0.016, 0.012), (0.019, 0, 0.046), M['brass'], 0.002, parent=wr)

    # J5 collet roll, grips the brush ferrule (axis parallel to the wrist link, offset 36 mm)
    co = group('Arm_Collet', (0.036, 0, 0.034), parent=wr)
    cyl('Arm_ColletBody', 0.0095, 0.026, (0, 0, 0), M['brass'], n=20, bev=0.001, parent=co)
    cyl('Arm_ColletNut', 0.0115, 0.008, (0, 0, 0.02), M['brass_dark'], n=12, bev=0.0008, parent=co)
    cyl('Arm_ColletJaw', 0.0062, 0.008, (0, 0, 0.028), M['brass'], n=16, r2=0.0052, parent=co)
    empty('Arm_BrushMount', (0, 0, 0.036), parent=co)
    finish('arm', extras=dict(upper=UP, lower=LO, shoulder_h=0.125, wrist_to_mount=0.09))


# =============================================================== brushes and pot

def brush(name, kind, M, hair_m, handle_m):
    g = group(name)
    n = 12
    if kind == 'flat':
        # 50 mm flat wash: flat ferrule and flat hair block, flat wooden handle
        box(f'{name}_Ferrule', (0.052, 0.009, 0.024), (0, 0, 0), M['nickel'], 0.002, parent=g)
        hair = box(f'{name}_Hair', (0.050, 0.007, 0.036), (0, 0, -0.012 - 0.018), hair_m, 0.002, parent=g)
        box(f'{name}_HairEdge', (0.049, 0.0035, 0.004), (0, 0, -0.012 - 0.036 - 0.0015), hair_m, 0.001, parent=g)
        box(f'{name}_Handle', (0.022, 0.012, 0.16), (0, 0, 0.012 + 0.08), handle_m, 0.005, parent=g)
        tip = -0.012 - 0.036 - 0.0035
    else:
        if kind == 'round12':
            hl, hr, fl, fr, hdl, hdr = 0.030, 0.0042, 0.026, 0.0036, 0.15, 0.0042
        else:
            hl, hr, fl, fr, hdl, hdr = 0.016, 0.0021, 0.020, 0.0023, 0.14, 0.0032
        lathe(f'{name}_Ferrule', [(fr * 0.95, -fl / 2), (fr, -fl / 2 + 0.002), (fr, fl / 2 - 0.004),
                                   (hdr * 0.95, fl / 2), (hdr * 0.9, fl / 2 + 0.0005)], n=n, m=M['nickel'], parent=g)
        lathe(f'{name}_Hair', [(fr * 0.9, -fl / 2 + 0.001), (hr, -fl / 2 - hl * 0.35), (hr * 0.8, -fl / 2 - hl * 0.7),
                                (hr * 0.25, -fl / 2 - hl * 0.95), (0, -fl / 2 - hl)], n=n, m=hair_m, parent=g)
        lathe(f'{name}_Handle', [(hdr * 0.9, fl / 2), (hdr * 1.15, fl / 2 + hdl * 0.25), (hdr * 0.9, fl / 2 + hdl * 0.8),
                                  (hdr * 0.55, fl / 2 + hdl), (0, fl / 2 + hdl + 0.001)], n=n, m=handle_m, parent=g)
        tip = -fl / 2 - hl
    empty(f'{name}_Tip', (0, 0, tip), parent=g)
    empty(f'{name}_Grip', (0, 0, 0), parent=g)
    return g


def build_brushes():
    lib.reset()
    M = palette_mats()
    root = group('BrushSet')
    pot = group('Brush_Pot', parent=root)
    lathe('Brush_Pot_Body', [(0, 0.006), (0.030, 0.006), (0.034, 0.0), (0.036, 0.004), (0.037, 0.06),
                             (0.0355, 0.098), (0.037, 0.1), (0.0335, 0.1), (0.032, 0.012), (0, 0.012)],
          n=28, m=M['ceramic'], parent=pot)
    defs = [('Brush_Wash', 'flat', mat('Hair_Goat', '#D9CDB4', 0.8), mat('Handle_Natural', '#C49A6C', 0.5)),
            ('Brush_Round12', 'round12', mat('Hair_Sable', '#3B2A1E', 0.6), mat('Handle_Oxblood', '#6A2C22', 0.3)),
            ('Brush_Round4', 'round4', mat('Hair_Sable', '#3B2A1E', 0.6), mat('Handle_Black', '#25221F', 0.3))]
    slots = [(-0.012, 0.004, R(-8)), (0.012, 0.010, R(7)), (0.002, -0.014, R(5))]
    for i, ((nm, kind, hm, hd), (x, y, lean)) in enumerate(zip(defs, slots)):
        b = brush(nm, kind, M, hm, hd)
        # stored hair up, handle resting in the pot
        slot = empty(f'Pot_Slot_{i}', (x, y, 0.13), parent=pot)
        slot.rotation_euler = (lean, R(180), 0)
        b.parent = slot
        b.location = (0, 0, 0)
    finish('brushes', extras=dict(note='brush origin = grip point at mid-ferrule; hair along local -Y in three.js'))


# =============================================================== porcelain palette

def rounded_rect(w, d, r, n=6):
    pts = []
    for cx, cy, a0 in [(w / 2 - r, -d / 2 + r, -90), (w / 2 - r, d / 2 - r, 0), (-w / 2 + r, d / 2 - r, 90),
                       (-w / 2 + r, -d / 2 + r, 180)]:
        for i in range(n + 1):
            a = R(a0 + 90 * i / n)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def build_palette():
    lib.reset()
    M = palette_mats()
    root = group('Palette')
    W, D, H = 0.24, 0.15, 0.018
    body = prism('Palette_Body', rounded_rect(W, D, 0.014), H, M['porcelain'])
    body.parent = root
    xs = [-0.081, -0.027, 0.027, 0.081]
    ys = [0.046, 0.012]
    wells = []
    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            wells.append((x, y))
    for i, (x, y) in enumerate(wells):
        cut = lathe(f'cut{i}', [(0, H - 0.0095), (0.008, H - 0.009), (0.0125, H - 0.0065), (0.0142, H - 0.002),
                                (0.0148, H + 0.01), (0, H + 0.01)], n=28, loc=(x, y, 0))
        boolean(body, cut)
    for i, x in enumerate((-0.056, 0.056)):
        cut = box(f'slant{i}', (0.100, 0.046, 0.03), (x, -0.040, H + 0.0105), bev=0)
        cut.rotation_euler = (R(-6), 0, 0)
        boolean(body, cut)
    # a chipped rim on the Payne's grey well
    chip = sphere('chip', 0.004, (wells[7][0] + 0.0105, wells[7][1] - 0.0105, H + 0.0012), seg=10, rings=6)
    boolean(body, chip)
    lib.bevel(body, 0.0016, 2, 35)
    for i, ((x, y), (pname, col)) in enumerate(zip(wells, PIGMENTS)):
        cake = group(f'Palette_Cake_{i}', (x, y, H - 0.0088), parent=root)
        cake['pigment'] = pname
        lathe(f'Palette_Cake_{i}_Mesh', [(0, 0), (0.0095, 0), (0.0118, 0.0025), (0.0116, 0.0052), (0.0095, 0.0058),
                                          (0, 0.006)], n=20, m=mat(f'Pigment_{pname}', col, 0.35), parent=cake)
        empty(f'Palette_Well_{i}', (x, y, H), parent=root)
        empty(f'Palette_CakeTop_{i}', (x, y, H - 0.0088 + 0.006), parent=root)
    for i, x in enumerate((-0.056, 0.056)):
        empty(f'Palette_Slant_{i}', (x, -0.040, H - 0.0045), parent=root)
    finish('palette', extras=dict(pigments=[p for p, _ in PIGMENTS]))


# =============================================================== water jar

def build_jar():
    lib.reset()
    M = palette_mats()
    root = group('Jar')
    R0, Hh, wall, bot = 0.045, 0.110, 0.003, 0.008
    lathe('Jar_Glass', [(0, 0), (R0 - 0.004, 0), (R0 - 0.0005, 0.002), (R0, 0.006), (R0, Hh - 0.002),
                        (R0 - 0.0012, Hh), (R0 - wall, Hh - 0.001), (R0 - wall, bot + 0.002),
                        (R0 - wall - 0.004, bot), (0, bot)], n=40, m=M['glass'], parent=root)
    lathe('Jar_RimRing', [(R0 + 0.0002, Hh - 0.013), (R0 + 0.0014, Hh - 0.0125), (R0 + 0.0014, Hh - 0.0035),
                          (R0 + 0.0002, Hh - 0.003)], n=40, m=M['brass'], parent=root)
    wg = group('Jar_Water', (0, 0, bot + 0.0004), parent=root)
    wl = 0.082
    lathe('Jar_Water_Mesh', [(0, 0), (R0 - wall - 0.0045, 0), (R0 - wall - 0.0006, 0.002), (R0 - wall - 0.0006, wl),
                             (0, wl)], n=36, m=M['water'], parent=wg)
    # etched brass ring at the budget cap (shown on the rinse jar only)
    lathe('Jar_CapRing', [(R0 + 0.0001, 0.071), (R0 + 0.0007, 0.0712), (R0 + 0.0007, 0.0728), (R0 + 0.0001, 0.073)],
          n=40, m=M['brass_dark'], parent=root)
    empty('Jar_Rim', (0, 0, Hh), parent=root)
    empty('Jar_WaterSurface', (0, 0, bot + 0.0004 + wl), parent=root)
    finish('jar', extras=dict(water_full_m=wl, water_base_m=bot + 0.0004))


# =============================================================== drying rack with brass rail

def build_rack():
    """Panel drying rack: a dowel rack 0.56 wide (x) by 0.30 deep (y) by 0.45 tall, closed at the west end and open
    at the east end toward the board. Three slots run along x, so a panel standing in a slot (its width along x)
    slides in and out along the brass rail, the way it rides off the board."""
    lib.reset()
    M = palette_mats()
    root = group('Rack')
    W, D, H = 0.56, 0.30, 0.45
    box('Rack_Base', (W, D, 0.022), (0, 0, 0.011), M['wood'], 0.003, parent=root)
    box('Rack_Foot_W', (0.04, D, 0.012), (-W / 2 + 0.03, 0, -0.0), M['wood_dark'], 0.002, parent=root)
    # closed west end board and one back post; a back top rail joins them
    box('Rack_EndBoard', (0.02, D, H), (-W / 2 + 0.01, 0, H / 2), M['wood'], 0.003, parent=root)
    box('Rack_BackPost', (0.024, 0.024, H - 0.022), (W / 2 - 0.012, D / 2 - 0.012, 0.022 + (H - 0.022) / 2), M['wood'],
        0.003, parent=root)
    box('Rack_BackRail', (W - 0.02, 0.024, 0.024), (0.01, D / 2 - 0.012, H - 0.012), M['wood'], 0.003, parent=root)
    # slots at y = -0.08, 0, 0.08: a groove in the base for each, dowel dividers between them
    slot_y = [-0.08, 0.0, 0.08]
    for i, y in enumerate(slot_y):
        box(f'Rack_Groove_{i}', (W - 0.05, 0.010, 0.004), (0.005, y, 0.0225), M['wood_dark'], 0.001, parent=root)
        empty(f'Rack_Slot_{i}', (0.0, y, 0.022), parent=root)
    for k, y in enumerate((-0.12, -0.04, 0.04, 0.12)):
        for m, x in enumerate((-0.17, 0.02, 0.21)):
            cyl(f'Rack_Dowel_{k}_{m}', 0.0055, 0.24, (x, y, 0.02), M['wood_dark'], n=10, parent=root)
    # brass rail: a two-rod channel along slot 1 from inside the rack out past the open east end toward the board
    zr, x0, x1 = 0.06, -0.05, W / 2 + 0.09
    for sy in (-1, 1):
        cyl(f'Rack_Rail_{sy}', 0.003, x1 - x0, (x0, sy * 0.008, zr), M['brass'], n=10, rot=(0, R(90), 0), parent=root)
    for j, x in enumerate((W / 2 - 0.03, x1 - 0.01)):
        cyl(f'Rack_RailPost_{j}', 0.0045, zr - (0.022 if j == 0 else 0.0), (x, 0, 0.022 if j == 0 else 0.0),
            M['brass_dark'], n=8, parent=root)
        cyl(f'Rack_RailBridge_{j}', 0.0035, 0.024, (x, -0.012, zr), M['brass_dark'], n=8, rot=(R(-90), 0, 0),
            parent=root)
    cyl('Rack_RailFoot', 0.01, 0.003, (x1 - 0.01, 0, 0.0), M['brass_dark'], n=12, parent=root)
    empty('Rack_Rail_Start', (x0, 0, zr + 0.003), parent=root)
    empty('Rack_Rail_End', (x1, 0, zr + 0.003), parent=root)
    finish('rack', extras=dict(size_m=[W, D, H], slots_y=slot_y))


# =============================================================== stool, side table, shelf

def build_stool():
    lib.reset()
    M = palette_mats()
    root = group('Stool')
    lathe('Stool_Seat', [(0, 0.615), (0.16, 0.615), (0.175, 0.625), (0.175, 0.642), (0.165, 0.65), (0.08, 0.647),
                         (0, 0.646)], n=28, m=M['wood'], parent=root)
    for i in range(3):
        a = R(90 + 120 * i)
        tube(f'Stool_Leg_{i}', [(0.10 * math.cos(a), 0.10 * math.sin(a), 0.616),
                                (0.175 * math.cos(a), 0.175 * math.sin(a), 0.006)], 0.0145, M['wood'], res=1,
             parent=root)
    ring = [(0.148 * math.cos(R(t * 30)), 0.148 * math.sin(R(t * 30)), 0.22) for t in range(12)]
    tube('Stool_Ring', ring, 0.008, M['wood_dark'], res=1, parent=root, closed=True)
    empty('Stool_SeatTop', (0, 0, 0.646), parent=root)
    finish('stool')


def build_sidetable():
    lib.reset()
    M = palette_mats()
    root = group('SideTable')
    W, D, H = 0.60, 0.45, 0.70
    box('SideTable_Top', (W, D, 0.024), (0, 0, H - 0.012), M['wood'], 0.003, parent=root)
    for sx in (-1, 1):
        for sy in (-1, 1):
            cyl(f'SideTable_Leg_{sx}_{sy}', 0.016, H - 0.024, (sx * (W / 2 - 0.05), sy * (D / 2 - 0.05), 0),
                M['wood'], n=12, r2=0.012, parent=root)
    for sy in (-1, 1):
        box(f'SideTable_Apron_{sy}', (W - 0.1, 0.018, 0.07), (0, sy * (D / 2 - 0.05), H - 0.06), M['wood'], 0.002,
            parent=root)
    for sx in (-1, 1):
        box(f'SideTable_ApronS_{sx}', (0.018, D - 0.1, 0.07), (sx * (W / 2 - 0.05), 0, H - 0.06), M['wood'], 0.002,
            parent=root)
    # open notebook (hand-built stand-in for the SimReady book) and a pencil
    nb = group('SideTable_Notebook', (0.05, 0.0, H), parent=root)
    nb.rotation_euler = (0, 0, R(-8))
    for sx in (-1, 1):
        p = box(f'SideTable_Notebook_Page_{sx}', (0.145, 0.21, 0.008), (sx * 0.074, 0, 0.0045), M['paper'], 0.002,
                parent=nb)
        p.rotation_euler = (0, R(-3 * sx), 0)
    box('SideTable_Notebook_Cover', (0.30, 0.215, 0.002), (0, 0, 0.001), M['ink'], 0.0008, parent=nb)
    cyl('SideTable_Pencil', 0.0035, 0.17, (-0.12, -0.13, H + 0.0035), mat('Pencil', '#C9A23A', 0.5), n=6,
        rot=(0, R(90), R(20)), parent=root)
    empty('SideTable_TopCentre', (0, 0, H), parent=root)
    finish('sidetable')


def build_shelf():
    lib.reset()
    M = palette_mats()
    root = group('Shelf')
    W, D, H = 0.90, 0.30, 1.80
    for sx in (-1, 1):
        box(f'Shelf_Side_{sx}', (0.022, D, H), (sx * (W / 2 - 0.011), 0, H / 2), M['wood'], 0.003, parent=root)
    levels = [0.06, 0.44, 0.82, 1.20, 1.58, 1.78]
    for i, z in enumerate(levels):
        box(f'Shelf_Board_{i}', (W - 0.044, D - 0.01, 0.02), (0, -0.005, z), M['wood'], 0.002, parent=root)
        empty(f'Shelf_Level_{i}', (0, 0, z + 0.01), parent=root)
    for z in (0.30, 1.40):
        box(f'Shelf_BackRail_{z}', (W - 0.044, 0.016, 0.06), (0, D / 2 - 0.008, z), M['wood_dark'], 0.002,
            parent=root)
    box('Shelf_Kick', (W - 0.044, 0.018, 0.05), (0, -D / 2 + 0.02, 0.025), M['wood_dark'], 0.002, parent=root)

    # sketchbook for painter version search_v0, spine out, Sepia on the spine (decal placed by the app)
    zb = levels[2] + 0.01
    sk = group('Sketchbook_search_v0', (-0.30, -0.02, zb), parent=root)
    sk['painter'] = 'search_v0'
    box('Sketchbook_search_v0_Block', (0.017, 0.21, 0.297), (0, 0, 0.1485), M['paper'], 0.001, parent=sk)
    box('Sketchbook_search_v0_Cover', (0.021, 0.214, 0.301), (0, 0.0015, 0.1505), mat('Sketchbook_Cover', '#3E4A55', 0.6),
        0.002, parent=sk)
    empty('Sketchbook_search_v0_SpineBadge', (0, -0.1085, 0.22), parent=sk)
    # plain books and paper beside it
    xs = -0.27
    for i, (t, h, d, m) in enumerate([(0.03, 0.24, 0.17, M['ink']), (0.022, 0.21, 0.15, M['book_green']),
                                      (0.04, 0.26, 0.19, mat('Book_Cover_Ochre', '#9C7A45', 0.7))]):
        xs += 0.03 + t / 2 if i else 0.03
        b = box(f'Shelf_Book_{i}', (t, d, h), (xs, -0.02, zb + h / 2), m, 0.002, parent=root)
        xs += t / 2
    lean = box('Shelf_Book_Lean', (0.03, 0.18, 0.25), (0.02, -0.02, zb + 0.12), M['ink'], 0.002, parent=root)
    lean.rotation_euler = (0, R(-18), 0)
    # small jars with lids (hand-built stand-ins), bottles, mug of spare brushes, bowl with a rag
    z3 = levels[3] + 0.01
    for i, x in enumerate((-0.32, -0.23)):
        g = group(f'Shelf_Jar_{i}', (x, -0.03, z3), parent=root)
        lathe(f'Shelf_Jar_{i}_Glass', [(0, 0), (0.033, 0), (0.035, 0.005), (0.035, 0.085), (0.03, 0.09), (0, 0.09)],
              n=20, m=M['glass'], parent=g)
        lathe(f'Shelf_Jar_{i}_Lid', [(0, 0.09), (0.031, 0.09), (0.032, 0.104), (0, 0.105)], n=20,
              m=M['brass_dark'] if i else M['cap'], parent=g)
        lathe(f'Shelf_Jar_{i}_Fill', [(0, 0.004), (0.031, 0.004), (0.031, 0.05 + 0.02 * i), (0, 0.05 + 0.02 * i)],
              n=16, m=mat(f'Jar_Fill_{i}', ('#C9A86A', '#7E8F9A')[i], 0.6), parent=g)
    for i, (x, label) in enumerate(((-0.12, 'gum_arabic'), (-0.06, 'masking_fluid'))):
        g = group(f'Shelf_Bottle_{label}', (x, -0.04, z3), parent=root)
        lathe(f'Shelf_Bottle_{label}_Glass', [(0, 0), (0.024, 0), (0.025, 0.004), (0.025, 0.075), (0.011, 0.092),
                                               (0.009, 0.105), (0, 0.105)], n=16, m=M['amber'], parent=g)
        lathe(f'Shelf_Bottle_{label}_Cap', [(0, 0.103), (0.0105, 0.103), (0.0105, 0.122), (0, 0.123)], n=12,
              m=M['cap'], parent=g)
    mug = group('Shelf_Mug', (0.12, -0.03, z3), parent=root)
    lathe('Shelf_Mug_Body', [(0, 0), (0.038, 0), (0.042, 0.006), (0.043, 0.095), (0.04, 0.097), (0.039, 0.012),
                             (0, 0.012)], n=24, m=mat('Mug_Glaze', '#A6683F', 0.3), parent=mug)
    tube('Shelf_Mug_Handle', [(0.042, 0, 0.075), (0.062, 0, 0.07), (0.064, 0, 0.035), (0.042, 0, 0.025)], 0.005,
         mat('Mug_Glaze', '#A6683F', 0.3), res=1, parent=mug)
    for i, (dx, dy, lean_a, col) in enumerate(((-0.01, 0.0, 8, '#6A2C22'), (0.012, 0.008, -6, '#25221F'),
                                                 (0.0, -0.012, 3, '#C49A6C'))):
        h = cyl(f'Shelf_Mug_Brush_{i}', 0.0035, 0.19, (dx, dy, 0.012), mat(f'Spare_Handle_{i}', col, 0.4), n=6,
                r2=0.002, parent=mug)
        h.rotation_euler = (R(lean_a), R(-lean_a * 0.6), 0)
    bowl = group('Shelf_Bowl', (0.28, -0.03, z3), parent=root)
    lathe('Shelf_Bowl_Body', [(0, 0), (0.035, 0), (0.06, 0.03), (0.075, 0.06), (0.071, 0.061), (0.055, 0.032),
                              (0.03, 0.008), (0, 0.008)], n=28, m=M['porcelain'], parent=bowl)
    sphere('Shelf_Bowl_Rag', 0.05, (0, 0, 0.04), M['cloth'], seg=12, rings=6, scale=(1.1, 0.9, 0.45), parent=bowl)
    # paper roll and a stack of fresh sheets on the bottom shelf
    z0 = levels[0] + 0.01
    box('Shelf_PaperStack', (0.34, 0.26, 0.04), (-0.2, -0.01, z0 + 0.02), M['paper'], 0.002, parent=root)
    cyl('Shelf_PaperRoll', 0.035, 0.6, (0.12, -0.03, z0 + 0.035), M['paper'], n=16, rot=(0, R(90), 0), parent=root)
    finish('shelf')


# =============================================================== lamp, cork boards, small bench props

def build_lamp():
    lib.reset()
    M = palette_mats()
    root = group('Lamp')
    cyl('Lamp_WallPlate', 0.05, 0.012, (0, 0, 0), M['brass'], n=32, bev=0.002, rot=(R(90), 0, 0), parent=root)
    cyl('Lamp_Knuckle', 0.012, 0.03, (0, -0.012, 0), M['brass_dark'], n=14, rot=(R(90), 0, 0), parent=root)
    arm = [(0, -0.035, 0), (0, -0.14, 0.01), (0, -0.30, -0.04), (0, -0.42, -0.14), (0, -0.46, -0.22)]
    tube('Lamp_Arm', arm, 0.0065, M['brass'], res=2, parent=root)
    head = group('Lamp_Head', (0, -0.46, -0.235), parent=root)
    head.rotation_euler = (R(-12), 0, 0)
    sphere('Lamp_Swivel', 0.012, (0, 0, 0.012), M['brass_dark'], seg=12, rings=6, parent=head)
    lathe('Lamp_Shade', [(0.018, 0.0), (0.036, -0.01), (0.06, -0.06), (0.085, -0.12), (0.088, -0.13)], n=36,
          m=M['brass'], parent=head)
    lathe('Lamp_ShadeInner', [(0.0865, -0.1295), (0.0835, -0.12), (0.0585, -0.06), (0.0345, -0.011), (0.017, -0.002)],
          n=36, m=mat('Lamp_ShadeInner', '#F1E6CF', 0.5), parent=head)
    cyl('Lamp_Socket', 0.014, 0.03, (0, 0, -0.04), M['brass_dark'], n=14, parent=head)
    sphere('Lamp_Bulb', 0.028, (0, 0, -0.072), mat('Lamp_Bulb', '#FFE3B0', 0.2, emit='#FFD58A', emit_strength=8.0),
           seg=16, rings=8, parent=head)
    empty('Lamp_Light', (0, 0, -0.075), parent=head)
    finish('lamp')


def build_corkboards():
    lib.reset()
    M = palette_mats()

    def framed(name, w, h, parent):
        g = group(name, parent=parent)
        box(f'{name}_Cork', (w, 0.012, h), (0, -0.006, 0), M['cork'], 0.002, parent=g)
        for nm, sz, loc in [('T', (w + 0.04, 0.02, 0.02), (0, -0.01, h / 2 + 0.01)),
                            ('B', (w + 0.04, 0.02, 0.02), (0, -0.01, -h / 2 - 0.01)),
                            ('L', (0.02, 0.02, h), (-w / 2 - 0.01, -0.01, 0)),
                            ('R', (0.02, 0.02, h), (w / 2 + 0.01, -0.01, 0))]:
            box(f'{name}_Frame{nm}', sz, loc, M['wood_dark'], 0.003, parent=g)
        return g

    def paper(name, w, h, loc, m, parent, tilt=0.0):
        bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, 0, 0))
        p = bpy.context.active_object
        p.name = name
        p.scale = (w, h, 1)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        p.rotation_euler = (R(90), tilt, 0)
        p.location = loc
        p.data.materials.append(m)
        p['keep_uv'] = 1
        p.parent = parent
        return p

    def pin(name, loc, parent):
        g = group(name, loc, parent=parent)
        cyl(f'{name}_Head', 0.005, 0.004, (0, -0.004, 0), M['brass'], n=12, rot=(R(90), 0, 0), bev=0.0008, parent=g)
        return g

    root = group('Corkboards')
    # cork strip above the board: target print, prompt card, credit slip
    cs = framed('CorkStrip', 0.84, 0.30, root)
    paper('Target_Print', 0.305, 0.229, (-0.17, -0.0125, 0.0), mat('Target_Print', '#F4F0E6', 0.85), cs)
    pin('Target_Pin_L', (-0.30, -0.013, 0.10), cs)
    pin('Target_Pin_R', (-0.04, -0.013, 0.10), cs)
    paper('Prompt_Card', 0.16, 0.11, (0.17, -0.0125, 0.03), mat('Prompt_Card', '#F7F2E4', 0.85), cs, tilt=R(-2))
    pin('Prompt_Pin', (0.17, -0.013, 0.075), cs)
    paper('Credit_Slip', 0.16, 0.026, (0.17, -0.0125, -0.07), mat('Credit_Slip', '#EFE8D6', 0.85), cs)
    pin('Credit_Pin', (0.245, -0.013, -0.07), cs)
    empty('CorkStrip_TargetCentre', (-0.17, -0.013, 0.0), parent=cs)
    # queue board on the east wall with paper slips (a gallery list in tier a)
    qb = framed('QueueBoard', 1.20, 0.80, root)
    qb.location = (3.0, 0, 0)
    for i in range(6):
        x = -0.42 + (i % 3) * 0.42
        z = 0.2 - (i // 3) * 0.3
        paper(f'Queue_Slip_{i}', 0.16, 0.09, (x, -0.0125, z), mat(f'Queue_Slip_{i}', '#F4EEDC', 0.85), qb,
              tilt=R((-3, 2, -1, 3, -2, 1)[i]))
        pin(f'Queue_Pin_{i}', (x, -0.013, z + 0.035), qb)
    finish('corkboards', extras=dict(cork_strip=[0.84, 0.30], queue_board=[1.20, 0.80]))


def build_props():
    """Small bench props: score card, calibration strip, tea cup, paper stack, plant, rope rail."""
    lib.reset()
    M = palette_mats()
    root = group('Props')
    # score card, face down; front carries the result (texture set by the app)
    sc = group('ScoreCard', (0, 0, 0), parent=root)
    box('ScoreCard_Back', (0.15, 0.10, 0.0008), (0, 0, 0.0004), mat('ScoreCard_Back', '#5E4B3C', 0.7), 0, parent=sc)
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, 0, 0.00085))
    fr = bpy.context.active_object
    fr.name = 'ScoreCard_Front'
    fr.scale = (0.148, 0.098, 1)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    fr.data.materials.append(mat('ScoreCard_Front', '#F4F0E6', 0.85))
    fr['keep_uv'] = 1
    fr.parent = sc
    # calibration strip: paper card with eight stripes (revealed in turn by the app)
    cal = group('CalStrip', (0.3, 0, 0), parent=root)
    box('CalStrip_Card', (0.20, 0.034, 0.0006), (0, 0, 0.0003), M['paper'], 0, parent=cal)
    for i, (pname, col) in enumerate(PIGMENTS):
        box(f'CalStrip_Stripe_{i}', (0.019, 0.026, 0.0002), (-0.0875 + i * 0.025, 0, 0.0007),
            mat(f'Wash_{pname}', col, 0.8), 0, parent=cal)
    # tea cup on a saucer (hand-built stand-in for the SimReady cup)
    cup = group('TeaCup', (0.6, 0, 0), parent=root)
    porc = M['porcelain']
    lathe('TeaCup_Saucer', [(0, 0), (0.035, 0), (0.07, 0.008), (0.072, 0.012), (0.068, 0.0115), (0.034, 0.005),
                            (0, 0.005)], n=28, m=porc, parent=cup)
    lathe('TeaCup_Body', [(0, 0.006), (0.025, 0.006), (0.038, 0.02), (0.045, 0.058), (0.043, 0.0595), (0.036, 0.022),
                          (0.024, 0.011), (0, 0.011)], n=28, m=porc, parent=cup)
    lathe('TeaCup_Tea', [(0, 0.045), (0.041, 0.045)], n=24, m=mat('Tea', '#6B3E1F', 0.08), parent=cup)
    tube('TeaCup_Handle', [(0.043, 0, 0.05), (0.06, 0, 0.048), (0.062, 0, 0.03), (0.04, 0, 0.022)], 0.0035, porc,
         res=1, parent=cup)
    # fresh sheets in the drawer
    box('PaperStack', (0.32, 0.24, 0.012), (0.9, 0, 0.006), M['paper'], 0.001, parent=root)
    # plant on the sill: terracotta pot, soil and strap leaves
    pl = group('Plant', (1.3, 0, 0), parent=root)
    lathe('Plant_Pot', [(0, 0.0), (0.05, 0.0), (0.065, 0.11), (0.072, 0.112), (0.072, 0.13), (0.064, 0.13),
                        (0.06, 0.115), (0, 0.115)], n=24, m=M['terracotta'], parent=pl)
    lathe('Plant_Soil', [(0, 0.118), (0.06, 0.118)], n=20, m=M['soil'], parent=pl)
    import random
    rnd = random.Random(7)
    for i in range(11):
        a = 2 * math.pi * i / 11 + rnd.uniform(-0.2, 0.2)
        h = rnd.uniform(0.22, 0.32)
        out = rnd.uniform(0.07, 0.13)
        pts = [(0.01 * math.cos(a), 0.01 * math.sin(a), 0.12),
               (out * 0.4 * math.cos(a), out * 0.4 * math.sin(a), 0.12 + h * 0.6),
               (out * math.cos(a), out * math.sin(a), 0.12 + h)]
        cu = bpy.data.curves.new(f'Plant_Leaf_{i}', 'CURVE')
        cu.dimensions = '3D'
        cu.extrude = 0.009
        sp = cu.splines.new('POLY')
        sp.points.add(len(pts) - 1)
        for p, c in zip(sp.points, pts):
            p.co = (*c, 1)
        sp.points[-1].radius = 0.2
        sp.points[0].radius = 0.5
        cu.twist_mode = 'MINIMUM'
        o = lib.link(bpy.data.objects.new(f'Plant_Leaf_{i}', cu))
        cu.materials.append(M['leaf'])
        o.parent = pl
    # rope rail with three brass posts, 2.40 m long, rope sagging between posts
    rr = group('RopeRail', (0, 1.0, 0), parent=root)
    for i, x in enumerate((-1.2, 0.0, 1.2)):
        lathe(f'RopeRail_Post_{i}', [(0, 0), (0.12, 0), (0.12, 0.02), (0.03, 0.035), (0.018, 0.06), (0.018, 0.86),
                                     (0.03, 0.88), (0.032, 0.9), (0, 0.92)], n=20, m=M['brass'], loc=(x, 0, 0),
              parent=rr)
    for seg, (xa, xb) in enumerate(((-1.2, 0.0), (0.0, 1.2))):
        pts = []
        for k in range(13):
            t = k / 12
            pts.append((xa + (xb - xa) * t, -0.02, 0.87 - 0.14 * math.sin(math.pi * t)))
        tube(f'RopeRail_Rope_{seg}', pts, 0.014, M['rope'], res=1, parent=rr)
    finish('props')


# =============================================================== wooden mannequin (seated visitor, standing scale figure)

def mannequin(name, seated, M, loc):
    g = group(name, loc)
    wood = M['beech']
    joint = M['wood_dark']
    # proportions for a 1.80 m figure: crotch 0.86, knee 0.50, shoulder 1.47, head 0.23 tall
    hip_z = 0.92
    pel = group(f'{name}_Pelvis', (0, 0, hip_z), parent=g)
    sphere(f'{name}_Hips', 0.13, (0, 0, 0.0), wood, seg=14, rings=7, scale=(1.0, 0.7, 0.75), parent=pel)
    sphere(f'{name}_Waist', 0.045, (0, 0, 0.13), joint, seg=10, rings=5, parent=pel)
    sphere(f'{name}_Chest', 0.16, (0, 0, 0.36), wood, seg=14, rings=7, scale=(1.0, 0.65, 1.15), parent=pel)
    cyl(f'{name}_Neck', 0.035, 0.10, (0, 0, 0.53), wood, n=10, parent=pel)
    sphere(f'{name}_Head', 0.095, (0, 0, 0.72), wood, seg=14, rings=8, scale=(0.85, 0.95, 1.2), parent=pel)
    for sx in (-1, 1):
        sh = group(f'{name}_Shoulder_{sx}', (sx * 0.19, 0, 0.47), parent=pel)
        sphere(f'{name}_ShoulderBall_{sx}', 0.04, (0, 0, 0), joint, seg=10, rings=5, parent=sh)
        sh.rotation_euler = (R(-22 if seated else 0), 0, 0)
        tube(f'{name}_UpperArm_{sx}', [(0, 0, -0.03), (0, 0, -0.28)], 0.038, wood, res=2, parent=sh)
        el = group(f'{name}_Elbow_{sx}', (0, 0, -0.30), parent=sh)
        el.rotation_euler = (R(-48 if seated else -5), 0, 0)
        sphere(f'{name}_ElbowBall_{sx}', 0.032, (0, 0, 0), joint, seg=10, rings=5, parent=el)
        tube(f'{name}_Forearm_{sx}', [(0, 0, -0.03), (0, 0, -0.25)], 0.032, wood, res=2, parent=el)
        sphere(f'{name}_Hand_{sx}', 0.045, (0, 0, -0.31), wood, seg=10, rings=5, scale=(0.6, 0.35, 1.0), parent=el)
        hp = group(f'{name}_Hip_{sx}', (sx * 0.085, 0, -0.04), parent=pel)
        hp.rotation_euler = (R(-90 if seated else 0), 0, R(sx * 4 if seated else 0))
        sphere(f'{name}_HipBall_{sx}', 0.05, (0, 0, 0), joint, seg=10, rings=5, parent=hp)
        tube(f'{name}_Thigh_{sx}', [(0, 0, -0.04), (0, 0, -0.38)], 0.058, wood, res=2, parent=hp)
        kn = group(f'{name}_Knee_{sx}', (0, 0, -0.42), parent=hp)
        kn.rotation_euler = (R(118 if seated else 0), 0, 0)
        sphere(f'{name}_KneeBall_{sx}', 0.045, (0, 0, 0), joint, seg=10, rings=5, parent=kn)
        tube(f'{name}_Shin_{sx}', [(0, 0, -0.035), (0, 0, -0.40)], 0.043, wood, res=2, parent=kn)
        box(f'{name}_Foot_{sx}', (0.08, 0.22, 0.05), (0, -0.06, -0.455), wood, 0.012, parent=kn)
    if seated:
        # sit the pelvis on a 0.646 m stool seat, thighs forward, shins tucked back so the feet rest on the foot ring
        pel.location = (0, 0.06, 0.646 + 0.075)
    return g


def build_mannequin():
    lib.reset()
    M = palette_mats()
    root = group('Mannequins')
    mannequin('Mannequin_Standing', False, M, (0, 0, 0))
    mannequin('Mannequin_Seated', True, M, (1.0, 0, 0))
    finish('mannequin', extras=dict(standing_height_m=1.80))


BUILDERS = dict(room=build_room, bench=build_bench, board=build_board, arm=build_arm, brushes=build_brushes,
                palette=build_palette, jar=build_jar, rack=build_rack, stool=build_stool, sidetable=build_sidetable,
                shelf=build_shelf, lamp=build_lamp, corkboards=build_corkboards, props=build_props,
                mannequin=build_mannequin)

REPORT = {}
if __name__ == '__main__':
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    names = argv or list(BUILDERS)
    rp = os.path.join(HERE, 'build_report.json')
    if os.path.exists(rp):
        REPORT.update(json.load(open(rp)))
    for n in names:
        BUILDERS[n]()
    json.dump(REPORT, open(rp, 'w'), indent=1, sort_keys=True)
    print('REPORT', json.dumps({k: (v['tris'], v['raw_bytes']) for k, v in REPORT.items()}))
