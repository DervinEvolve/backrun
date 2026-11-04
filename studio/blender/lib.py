"""Modelling helpers and a small GLB writer for the Backrun studio asset pack.

The Blender build on this machine (4.0.2, system Python 3.12) has no numpy, so the
bundled glTF exporter cannot load. This writer covers what the studio needs:
node hierarchy with pivots, meshes with corner normals, UVs, one primitive per
material, PBR factors, and empties as named nodes for attach points.

Axes: Blender is Z-up with -Y toward the viewer. glTF is Y-up with +Z toward the
viewer. A Blender point (x, y, z) is written as (x, z, -y).
"""

import bpy
import bmesh
import json
import math
import struct
from mathutils import Matrix, Vector

C = Matrix(((1, 0, 0, 0), (0, 0, 1, 0), (0, -1, 0, 0), (0, 0, 0, 1)))
CI = C.inverted()

MATS = {}


# ----------------------------------------------------------------- scene / materials

def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    MATS.clear()


def srgb_to_lin(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def hex_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def mat(name, color='#808080', rough=0.6, metal=0.0, emit=None, emit_strength=1.0, alpha=1.0, double=False):
    """Material by name; colour is sRGB hex, written to glTF as linear factors."""
    lin = tuple(srgb_to_lin(c) for c in hex_rgb(color))
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.diffuse_color = (*lin, alpha)
    MATS[name] = dict(color=lin, rough=rough, metal=metal, alpha=alpha, double=double,
                      emit=tuple(srgb_to_lin(c) for c in hex_rgb(emit)) if emit else None,
                      emit_strength=emit_strength)
    return m


def link(obj):
    bpy.context.scene.collection.objects.link(obj)
    return obj


def set_mat(obj, m):
    obj.data.materials.clear()
    obj.data.materials.append(m)
    return obj


def smooth(obj, angle=35):
    me = obj.data
    for p in me.polygons:
        p.use_smooth = True
    me.use_auto_smooth = True
    me.auto_smooth_angle = math.radians(angle)
    return obj


def bevel(obj, width=0.002, segments=2, angle=40):
    if width <= 0:
        return obj
    mod = obj.modifiers.new('Bevel', 'BEVEL')
    mod.width = width
    mod.segments = segments
    mod.limit_method = 'ANGLE'
    mod.angle_limit = math.radians(angle)
    mod.harden_normals = True
    smooth(obj, 30)
    return obj


def empty(name, loc=(0, 0, 0), parent=None, size=0.01):
    e = bpy.data.objects.new(name, None)
    e.empty_display_size = size
    e.location = loc
    link(e)
    if parent:
        e.parent = parent
    e['attach'] = 1
    return e


def group(name, loc=(0, 0, 0), parent=None):
    e = bpy.data.objects.new(name, None)
    e.location = loc
    link(e)
    if parent:
        e.parent = parent
    return e


def parent_keep(child, parent):
    mw = child.matrix_world.copy()
    child.parent = parent
    child.matrix_world = mw
    return child


def mesh_obj(name, bm, m=None, smooth_angle=None):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    obj = link(bpy.data.objects.new(name, me))
    if m is not None:
        set_mat(obj, m)
    if smooth_angle:
        smooth(obj, smooth_angle)
    return obj


def box(name, size, loc=(0, 0, 0), m=None, bev=0.0015, rot=(0, 0, 0), parent=None, segs=2):
    """Box of real size (x, y, z) centred at loc."""
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=Vector(size), verts=bm.verts)
    obj = mesh_obj(name, bm, m)
    obj.location = loc
    obj.rotation_euler = rot
    if bev:
        bevel(obj, min(bev, min(size) * 0.45), segs)
    if parent:
        obj.parent = parent
    return obj


def cyl(name, r, h, loc=(0, 0, 0), m=None, n=24, rot=(0, 0, 0), bev=0.0, parent=None, r2=None, cap=True):
    """Cylinder (or cone frustum with r2) along local Z, base at z=0 of local, centred XY."""
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=cap, cap_tris=False, segments=n, radius1=r,
                          radius2=r if r2 is None else r2, depth=h)
    bmesh.ops.translate(bm, vec=Vector((0, 0, h / 2)), verts=bm.verts)
    obj = mesh_obj(name, bm, m, smooth_angle=35)
    obj.location = loc
    obj.rotation_euler = rot
    if bev:
        bevel(obj, bev, 2)
    if parent:
        obj.parent = parent
    return obj


def sphere(name, r, loc=(0, 0, 0), m=None, seg=16, rings=8, scale=(1, 1, 1), parent=None):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=seg, v_segments=rings, radius=r)
    bmesh.ops.scale(bm, vec=Vector(scale), verts=bm.verts)
    obj = mesh_obj(name, bm, m, smooth_angle=80)
    obj.location = loc
    if parent:
        obj.parent = parent
    return obj


def lathe(name, profile, n=32, m=None, loc=(0, 0, 0), parent=None, angle=35, close_top=False, close_bottom=False):
    """Revolve a profile [(r, z), ...] about local Z. r == 0 points collapse to the axis."""
    bm = bmesh.new()
    rings = []
    for (r, z) in profile:
        if r <= 1e-6:
            rings.append([bm.verts.new((0, 0, z))])
        else:
            rings.append([bm.verts.new((r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n), z))
                          for i in range(n)])
    for a, b in zip(rings, rings[1:]):
        if len(a) == 1 and len(b) == 1:
            continue
        if len(a) == 1:
            for i in range(n):
                bm.faces.new((a[0], b[i], b[(i + 1) % n]))
        elif len(b) == 1:
            for i in range(n):
                bm.faces.new((a[i], b[0], a[(i + 1) % n]))
        else:
            for i in range(n):
                bm.faces.new((a[i], a[(i + 1) % n], b[(i + 1) % n], b[i]))
    if close_bottom and len(rings[0]) > 1:
        bm.faces.new(list(reversed(rings[0])))
    if close_top and len(rings[-1]) > 1:
        bm.faces.new(rings[-1])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    obj = mesh_obj(name, bm, m, smooth_angle=angle)
    obj.location = loc
    if parent:
        obj.parent = parent
    return obj


def tube(name, pts, radius, m=None, res=2, parent=None, closed=False, segs=None):
    """Swept round tube through points (a poly spline), converted to mesh at export."""
    cu = bpy.data.curves.new(name, 'CURVE')
    cu.dimensions = '3D'
    cu.bevel_depth = radius
    cu.bevel_resolution = res
    cu.use_fill_caps = True
    sp = cu.splines.new('POLY')
    sp.points.add(len(pts) - 1)
    for p, c in zip(sp.points, pts):
        p.co = (c[0], c[1], c[2], 1)
    sp.use_cyclic_u = closed
    obj = link(bpy.data.objects.new(name, cu))
    if m is not None:
        cu.materials.append(m)
    if parent:
        obj.parent = parent
    return obj


def prism(name, poly, depth, m=None, z0=0.0, parent=None, bev=0.0):
    """Extrude a closed XY polygon [(x, y), ...] from z0 up by depth."""
    bm = bmesh.new()
    bot = [bm.verts.new((x, y, z0)) for x, y in poly]
    top = [bm.verts.new((x, y, z0 + depth)) for x, y in poly]
    bm.faces.new(list(reversed(bot)))
    bm.faces.new(top)
    k = len(poly)
    for i in range(k):
        bm.faces.new((bot[i], bot[(i + 1) % k], top[(i + 1) % k], top[i]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    obj = mesh_obj(name, bm, m)
    if bev:
        bevel(obj, bev, 1)
    if parent:
        obj.parent = parent
    return obj


def boolean(target, cutter, op='DIFFERENCE', apply=True):
    mod = target.modifiers.new('Bool', 'BOOLEAN')
    mod.operation = op
    mod.solver = 'EXACT'
    mod.object = cutter
    cutter.hide_render = True
    cutter['skip'] = 1
    if apply:
        bpy.context.view_layer.objects.active = target
        for o in bpy.context.view_layer.objects:
            o.select_set(o == target)
        # move bevel after boolean so edges of the cut get bevelled too
        bpy.ops.object.modifier_apply(modifier=mod.name)
        bpy.data.objects.remove(cutter, do_unlink=True)
    return target


def apply_mods(obj):
    bpy.context.view_layer.objects.active = obj
    for o in bpy.context.view_layer.objects:
        o.select_set(o == obj)
    for mod in list(obj.modifiers):
        bpy.ops.object.modifier_apply(modifier=mod.name)
    return obj


def join(objs, name):
    """Join meshes (modifiers applied first) into one object keeping materials."""
    for o in objs:
        if o.type != 'MESH':
            bpy.context.view_layer.objects.active = o
            for x in bpy.context.view_layer.objects:
                x.select_set(x == o)
            bpy.ops.object.convert(target='MESH')
        apply_mods(o)
    bpy.ops.object.select_all(action='DESELECT')
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join()
    o = bpy.context.view_layer.objects.active
    o.name = name
    o.data.name = name
    return o


# ----------------------------------------------------------------- GLB writer

class _Buf:
    def __init__(self):
        self.data = bytearray()
        self.views = []
        self.accessors = []

    def add(self, raw, target=None):
        while len(self.data) % 4:
            self.data.append(0)
        off = len(self.data)
        self.data += raw
        v = {"buffer": 0, "byteOffset": off, "byteLength": len(raw)}
        if target:
            v["target"] = target
        self.views.append(v)
        return len(self.views) - 1

    def accessor(self, raw, ctype, count, typ, target=None, mn=None, mx=None):
        view = self.add(raw, target)
        a = {"bufferView": view, "componentType": ctype, "count": count, "type": typ}
        if mn is not None:
            a["min"] = mn
            a["max"] = mx
        self.accessors.append(a)
        return len(self.accessors) - 1


def _mesh_data(obj, dg):
    """Return {material_name: (positions, normals, uvs, indices)} in glTF axes."""
    ev = obj.evaluated_get(dg)
    me = ev.to_mesh()
    try:
        me.calc_normals_split()
    except Exception:
        pass
    me.calc_loop_triangles()
    keep_uv = bool(obj.get('keep_uv')) and me.uv_layers.active is not None
    uvl = me.uv_layers.active.data if keep_uv else None
    uv_scale = float(obj.get('uv_scale', 1.0))
    sc = obj.matrix_world.to_scale()
    slots = [s.material.name if s.material else 'Default' for s in obj.material_slots] or ['Default']
    prims = {}
    for tri in me.loop_triangles:
        mname = slots[min(tri.material_index, len(slots) - 1)]
        P = prims.setdefault(mname, dict(pos=[], nrm=[], uv=[], idx=[], lut={}))
        fn = tri.normal
        ax = max(range(3), key=lambda i: abs(fn[i]))
        for li, vi in zip(tri.loops, tri.vertices):
            co = me.vertices[vi].co
            n = me.loops[li].normal if me.has_custom_normals or me.use_auto_smooth else (
                me.vertices[vi].normal if tri.use_smooth else fn)
            if uvl is not None:
                u, v = uvl[li].uv
            else:
                w = (co.x * sc.x, co.y * sc.y, co.z * sc.z)
                if ax == 0:
                    u, v = (w[1] if fn[0] > 0 else -w[1]), w[2]
                elif ax == 1:
                    u, v = (-w[0] if fn[1] > 0 else w[0]), w[2]
                else:
                    u, v = w[0], w[1]
                u, v = u * uv_scale, v * uv_scale
            p = (co.x, co.z, -co.y)
            nn = (n[0], n[2], -n[1])
            t = (u, 1.0 - v)
            key = (round(p[0], 6), round(p[1], 6), round(p[2], 6), round(nn[0], 3), round(nn[1], 3),
                   round(nn[2], 3), round(t[0], 5), round(t[1], 5))
            k = P['lut'].get(key)
            if k is None:
                k = len(P['pos'])
                P['lut'][key] = k
                P['pos'].append(p)
                P['nrm'].append(nn)
                P['uv'].append(t)
            P['idx'].append(k)
    ev.to_mesh_clear()
    return prims


def export_glb(path, extras=None):
    dg = bpy.context.evaluated_depsgraph_get()
    buf = _Buf()
    gl = {"asset": {"version": "2.0", "generator": "backrun-studio lib.py"}, "scene": 0,
          "scenes": [{"nodes": []}], "nodes": [], "meshes": [], "materials": []}
    mat_index = {}
    stats = {"tris": 0}

    def material(name):
        if name in mat_index:
            return mat_index[name]
        d = MATS.get(name, dict(color=(0.5, 0.5, 0.5), rough=0.6, metal=0.0, alpha=1.0, double=False,
                                emit=None, emit_strength=1.0))
        m = {"name": name, "pbrMetallicRoughness": {
            "baseColorFactor": [d['color'][0], d['color'][1], d['color'][2], d['alpha']],
            "metallicFactor": d['metal'], "roughnessFactor": d['rough']}}
        if d['alpha'] < 1:
            m["alphaMode"] = "BLEND"
        if d['double']:
            m["doubleSided"] = True
        if d['emit']:
            m["emissiveFactor"] = list(d['emit'])
            if d['emit_strength'] != 1.0:
                m.setdefault("extensions", {})["KHR_materials_emissive_strength"] = {
                    "emissiveStrength": d['emit_strength']}
                gl.setdefault("extensionsUsed", [])
                if "KHR_materials_emissive_strength" not in gl["extensionsUsed"]:
                    gl["extensionsUsed"].append("KHR_materials_emissive_strength")
        gl["materials"].append(m)
        mat_index[name] = len(gl["materials"]) - 1
        return mat_index[name]

    def mesh(obj):
        prims = _mesh_data(obj, dg)
        out = []
        for mname, P in prims.items():
            if not P['idx']:
                continue
            n = len(P['pos'])
            pos = P['pos']
            mn = [min(p[i] for p in pos) for i in range(3)]
            mx = [max(p[i] for p in pos) for i in range(3)]
            a_pos = buf.accessor(struct.pack('<%df' % (3 * n), *[c for p in pos for c in p]), 5126, n, "VEC3",
                                 34962, mn, mx)
            a_nrm = buf.accessor(struct.pack('<%df' % (3 * n), *[c for p in P['nrm'] for c in p]), 5126, n,
                                 "VEC3", 34962)
            a_uv = buf.accessor(struct.pack('<%df' % (2 * n), *[c for p in P['uv'] for c in p]), 5126, n, "VEC2",
                                34962)
            ni = len(P['idx'])
            if n < 65536:
                raw = struct.pack('<%dH' % ni, *P['idx'])
                ct = 5123
            else:
                raw = struct.pack('<%dI' % ni, *P['idx'])
                ct = 5125
            a_idx = buf.accessor(raw, ct, ni, "SCALAR", 34963)
            stats["tris"] += ni // 3
            out.append({"attributes": {"POSITION": a_pos, "NORMAL": a_nrm, "TEXCOORD_0": a_uv},
                        "indices": a_idx, "material": material(mname)})
        if not out:
            return None
        gl["meshes"].append({"name": obj.name, "primitives": out})
        return len(gl["meshes"]) - 1

    def node(obj):
        local = obj.matrix_world.copy() if obj.parent is None else obj.parent.matrix_world.inverted() @ obj.matrix_world
        t, q, s = (C @ local @ CI).decompose()
        nd = {"name": obj.name}
        if t.length > 1e-9:
            nd["translation"] = [t.x, t.y, t.z]
        if abs(q.w - 1) > 1e-9:
            nd["rotation"] = [q.x, q.y, q.z, q.w]
        if (s - Vector((1, 1, 1))).length > 1e-9:
            nd["scale"] = [s.x, s.y, s.z]
        ex = {k: obj[k] for k in obj.keys() if k not in ('skip', 'keep_uv', 'uv_scale', '_RNA_UI')
              and isinstance(obj[k], (int, float, str))}
        if ex:
            nd["extras"] = ex
        gl["nodes"].append(nd)
        idx = len(gl["nodes"]) - 1
        kids = []
        if obj.type in ('MESH', 'CURVE', 'FONT'):
            mi = mesh(obj)
            if mi is not None:
                if obj.children:
                    # keep meshes on leaf nodes so later quantization never touches a pivot
                    gl["nodes"].append({"name": obj.name + "_geo", "mesh": mi})
                    kids.append(len(gl["nodes"]) - 1)
                else:
                    nd["mesh"] = mi
        for c in obj.children:
            if c.get('skip') or c.hide_render:
                continue
            kids.append(node(c))
        if kids:
            nd["children"] = kids
        return idx

    for obj in bpy.context.scene.objects:
        if obj.parent is None and not obj.get('skip') and not obj.hide_render:
            gl["scenes"][0]["nodes"].append(node(obj))
    if extras:
        gl["scenes"][0]["extras"] = extras
    gl["buffers"] = [{"byteLength": len(buf.data)}]
    gl["bufferViews"] = buf.views
    gl["accessors"] = buf.accessors
    if not gl["materials"]:
        del gl["materials"]
    js = json.dumps(gl, separators=(',', ':')).encode()
    while len(js) % 4:
        js += b' '
    bin_ = bytes(buf.data)
    while len(bin_) % 4:
        bin_ += b'\0'
    total = 12 + 8 + len(js) + 8 + len(bin_)
    with open(path, 'wb') as f:
        f.write(struct.pack('<III', 0x46546C67, 2, total))
        f.write(struct.pack('<II', len(js), 0x4E4F534A))
        f.write(js)
        f.write(struct.pack('<II', len(bin_), 0x004E4942))
        f.write(bin_)
    print("EXPORTED", path, "tris", stats["tris"], "bytes", total)
    return stats["tris"], total
