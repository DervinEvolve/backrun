// Backrun studio: the room, its lights, the stage cameras, and the replay of a recorded run driving the sheet, the
// arm, the palette, the jars, the panel and the score sheet. Every moving thing is a function of the replay clock.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { RectAreaLightUniformsLib } from 'three/addons/lights/RectAreaLightUniformsLib.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { N8AOPass } from 'n8ao';
import cameras from './data/cameras.json';
import layout from './data/layout.json';
import * as TX from './textures.js';
import { assetPreview } from './preview.js';
import { loadRun, PIGMENTS, PIGMENT_INDEX, T } from './replay.js';
import { SheetPainter } from './sheet.js';
import { makeArmIK } from './ik.js';
import { buildChoreo } from './choreo.js';
import { createDashboard, SPEEDS } from './dashboard.js';

const params = new URLSearchParams(location.search);
const STILL = params.has('still');
const AO = params.get('ao') !== '0';
const FIGURE = params.has('figure');
const D2R = Math.PI / 180;
const stageEl = document.getElementById('stage');
const viewW = () => stageEl.clientWidth || innerWidth;
const viewH = () => stageEl.clientHeight || innerHeight;

const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(STILL ? 1 : Math.min(devicePixelRatio, 2));
renderer.setSize(viewW(), viewH());
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.9;
renderer.outputColorSpace = THREE.SRGBColorSpace;
stageEl.appendChild(renderer.domElement);

const loader = new GLTFLoader().setMeshoptDecoder(MeshoptDecoder);
const load = (url) => new Promise((res, rej) => loader.load(url, res, undefined, rej));
const loadImage = (url) => new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = url; });

if (params.has('asset')) {
  assetPreview(renderer, load, params.get('asset'));
} else {
  main();
}

async function main() {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color('#2A231C');
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  scene.environmentIntensity = 0.32;

  const gallery = await (await fetch('runs/gallery.json')).json();
  const runName = params.get('run') ?? gallery.runs[0].name;
  const runBase = `runs/${runName}`;

  const names = ['room', 'bench', 'board', 'arm', 'brushes', 'palette', 'jar', 'rack', 'stool', 'sidetable', 'shelf',
    'lamp', 'corkboards', 'props', 'mannequin'];
  const [G, boxA, boxB, replay] = await Promise.all([
    Promise.all(names.map((n) => load(`models/${n}.glb`))).then((a) => Object.fromEntries(a.map((g, i) => [names[i], g.scene]))),
    load('models/simready/box_cardboard_a01.glb').then((g) => g.scene),
    load('models/simready/box_cardboard_b01.glb').then((g) => g.scene),
    loadRun(runBase),
  ]);
  const frameImages = await Promise.all(replay.frames.map((f) => loadImage(`${runBase}/${f.file}`)));
  const targetTex = await new THREE.TextureLoader().loadAsync(`${runBase}/target.png`);
  targetTex.colorSpace = THREE.SRGBColorSpace; targetTex.flipY = false; targetTex.anisotropy = 8;
  const sheet = new SheetPainter(replay, frameImages);
  const badgeTex = await svgTexture('brand/sepia.svg', 256);

  const place = (obj, key, extra = {}) => {
    const L = { ...layout[key], ...extra };
    obj.position.fromArray(L.pos);
    if (L.rot) obj.rotation.y = L.rot * D2R;
    if (L.scale) obj.scale.setScalar(L.scale);
    scene.add(obj);
    return obj;
  };
  const take = (root, name) => {
    const o = root.getObjectByName(name);
    o.removeFromParent();
    return o;
  };

  // ------------------------------------------------------------------ assembly
  place(G.room, 'room');
  place(G.bench, 'bench');
  place(G.board, 'board');
  place(G.arm, 'arm');
  place(G.palette, 'palette');
  const rack = place(G.rack, 'rack');
  place(G.brushes, 'brushes');
  place(G.shelf, 'shelf');
  place(G.sidetable, 'sidetable');
  place(G.lamp, 'lamp');
  const jarClean = place(G.jar, 'jar_clean');
  const jarRinse = place(G.jar.clone(true), 'jar_rinse');
  place(take(G.corkboards, 'CorkStrip'), 'corkstrip');
  place(take(G.corkboards, 'QueueBoard'), 'queueboard');
  place(take(G.props, 'TeaCup'), 'teacup');
  const plant = place(take(G.props, 'Plant'), 'plant');
  place(take(G.props, 'RopeRail'), 'roperail');
  const score = place(take(G.props, 'ScoreCard'), 'scorecard');
  score.rotation.x = Math.PI; // face down: the result side lies on the bench
  const scoreY = score.position.y + 0.0009;
  score.position.y = scoreY;
  const cal = place(take(G.props, 'CalStrip'), 'calstrip');
  cal.rotation.order = 'YXZ';
  cal.rotation.x = 30 * D2R;
  cal.position.y += 0.017 * Math.sin(30 * D2R);
  place(boxA, 'box_a');
  place(boxB, 'box_b');
  for (const [i, x] of layout.stools.x.entries()) {
    const s = G.stool.clone(true);
    s.position.set(x, 0, layout.stools.z);
    s.rotation.y = (i * 37 % 20 - 10) * D2R;
    scene.add(s);
  }
  const seated = take(G.mannequin, 'Mannequin_Seated');
  seated.position.set(layout.stools.x[layout.visitor.stool], 0, layout.stools.z);
  seated.rotation.y = Math.PI;
  scene.add(seated);
  const standing = take(G.mannequin, 'Mannequin_Standing');
  standing.position.set(1.3, 0, -0.55);
  standing.rotation.y = Math.PI * 0.82;
  standing.visible = FIGURE;
  scene.add(standing);

  // brushes: each keeps its pot slot; the one for the current phase rides in the collet
  const mount = G.arm.getObjectByName('Arm_BrushMount');
  const brushes = {};
  for (const [id, nm] of [['wash', 'Brush_Wash'], ['round12', 'Brush_Round12'], ['round4', 'Brush_Round4']]) {
    const o = G.brushes.getObjectByName(nm);
    brushes[id] = { o, slot: o.parent, tip: o.getObjectByName(`${nm}_Tip`), label: { wash: '50 mm flat wash', round12: 'size 12 round', round4: 'size 4 round' }[id] };
  }
  let brushInHand = null;
  function holdBrush(id) {
    if (brushInHand === id) return;
    if (brushInHand) { const b = brushes[brushInHand]; b.slot.add(b.o); b.o.position.set(0, 0, 0); b.o.rotation.set(0, 0, 0); }
    const b = brushes[id];
    mount.add(b.o); b.o.position.set(0, 0, 0); b.o.rotation.set(Math.PI, 0, 0);
    brushInHand = id;
    ik.setBrush(b.tip, b.o);
  }

  // Sepia decal on the sketchbook spine
  const spine = G.shelf.getObjectByName('Sketchbook_search_v0_SpineBadge');
  const decal = new THREE.Mesh(new THREE.PlaneGeometry(0.017, 0.017),
    new THREE.MeshStandardMaterial({ map: badgeTex, transparent: true, roughness: 0.5, alphaTest: 0.05 }));
  decal.position.z = 0.0006;
  spine.add(decal);

  // ------------------------------------------------------------------ materials
  const wood = TX.woodTexture(1), woodStain = TX.woodTexture(2, 1024, true), floor = TX.floorTexture();
  const brick = TX.brickTexture(), plaster = TX.plasterTexture(), cork = TX.corkTexture(), paperN = TX.paperNormal();
  const brushed = TX.brushedMetal();
  const rep = (t, s) => { const c = t.clone(); c.repeat.set(1 / s, 1 / s); c.needsUpdate = true; return c; };
  const withSet = (m, set, tile, ns = 1) => {
    m.map = rep(set.map, tile);
    if (set.normalMap) { m.normalMap = rep(set.normalMap, tile); m.normalScale.set(ns, ns); }
    if (set.roughnessMap) m.roughnessMap = rep(set.roughnessMap, tile);
  };
  const run = replay.index.run;
  const tp = run.target_path ?? '';
  const promptLines = tp.startsWith('synthetic:') ? [['Built-in test scene:', 30], [tp.split(':')[1], 44, 'Georgia, serif']]
    : run.prompt ? [['Prompt:', 28], [String(run.prompt).slice(0, 28), 34, 'Georgia, serif']] : [['Owner\'s photo', 40, 'Georgia, serif']];
  const scoreFront = TX.cardTexture([['Backrun score sheet', 20]], { w: 256, h: 176, top: 40 });
  const cards = {
    Target_Print: targetTex,
    Prompt_Card: TX.cardTexture(promptLines, { w: 512, h: 352 }),
    Credit_Slip: TX.cardTexture([[`run ${replay.index.name}, painter ${run.painter}, seed ${run.seed}`, 26, 'system-ui, sans-serif']], { w: 512, h: 84, top: 54 }),
    ScoreCard_Back: TX.cardTexture([], {
      w: 256, h: 176, bg: '#5E4B3C', draw: (g, w, h) => {
        g.strokeStyle = '#B5A642'; g.lineWidth = 3; g.strokeRect(10, 10, w - 20, h - 20);
      },
    }),
    ScoreCard_Front: scoreFront,
  };
  const shared = new Map();
  const glassMat = (src, color, opacity) => new THREE.MeshPhysicalMaterial({
    name: src.name, color, roughness: 0.03, metalness: 0, transparent: true, opacity, ior: 1.5,
    specularIntensity: 1, clearcoat: 1, clearcoatRoughness: 0.03, envMapIntensity: 2.4, depthWrite: false,
    side: THREE.DoubleSide, sheen: 0, iridescence: 0.08,
  });
  function upgrade(m) {
    if (shared.has(m.name)) return shared.get(m.name);
    let out = m;
    const n = m.name;
    if (n === 'Floor_Oak') withSet(m, floor, floor.tile, 0.6);
    else if (n === 'Brick_Limewash') withSet(m, brick, brick.tile, 1.2);
    else if (n === 'Plaster') withSet(m, plaster, plaster.tile, 0.5);
    else if (n === 'Wood_BenchTop') withSet(m, woodStain, 1.6, 0.7);
    else if (['Wood', 'Wood_Walnut', 'Wood_Beech', 'Window_Paint', 'Handle_Natural'].includes(n)) withSet(m, wood, 0.6, 0.35);
    else if (n === 'Cork') withSet(m, cork, cork.tile, 0.8);
    else if (n === 'Glass') { out = glassMat(m, '#EEF2EE', 0.13); }
    else if (n === 'Glass_Amber') { out = glassMat(m, '#8A5524', 0.7); }
    else if (n === 'Water_Clean') {
      out = new THREE.MeshPhysicalMaterial({ name: n, color: '#A9C2BE', roughness: 0.02, transparent: true, opacity: 0.55,
        clearcoat: 1, ior: 1.33, depthWrite: false, envMapIntensity: 1.6, side: THREE.DoubleSide });
    } else if (n === 'Paper_Sheet') {
      m.map = sheet.texture; m.color.set('#ffffff'); m.normalMap = paperN; m.normalScale.set(0.55, 0.55); m.roughness = 0.9;
      m.normalMap = rep(paperN, 0.5);
    } else if (['Paper', 'Carrier_Board', 'Kraft_Tape', 'Sketchbook_Cover', 'Rag_Cloth'].includes(n)) {
      m.normalMap = rep(paperN, 0.15); m.normalScale.set(0.3, 0.3);
    } else if (cards[n]) { m.map = cards[n]; m.color.set('#ffffff'); }
    else if (n.startsWith('Queue_Slip')) {
      m.map = TX.cardTexture([['gallery', 40, 'system-ui, sans-serif'], [replay.index.name, 34]], { w: 384, h: 216, top: 80 });
      m.color.set('#ffffff');
    } else if (n.startsWith('Brass') || n === 'Nickel') {
      // brushed metal: streaks along the tube, anisotropic highlights
      out = new THREE.MeshPhysicalMaterial({ name: n, color: m.color, metalness: 1, roughness: n === 'Brass_Aged' ? 0.42 : 0.3,
        roughnessMap: brushed.roughnessMap, normalMap: brushed.normalMap, normalScale: new THREE.Vector2(0.25, 0.25),
        anisotropy: 0.75, anisotropyRotation: Math.PI / 2, envMapIntensity: 1.5 });
    } else if (n === 'Lamp_Bulb') { m.toneMapped = false; m.emissiveIntensity = 6; }
    else if (n.startsWith('opaque__cardboard')) m.color.multiplyScalar(0.9);
    shared.set(n, out);
    return out;
  }
  const noShadow = /Glass|Water|Paper_Sheet|Kraft|Lamp_Bulb|Paint_Spot|Water_Ring|Target_Print|Prompt_Card|Credit_Slip|Queue_Slip|Enamel/;
  scene.traverse((o) => {
    if (!o.isMesh) return;
    o.material = upgrade(o.material);
    o.castShadow = !noShadow.test(o.material.name);
    o.receiveShadow = true;
    if (o.material.transparent) o.renderOrder = o.material.name.startsWith('Water') ? 1 : 2;
  });
  for (const b of Object.values(brushes)) b.o.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
  // the rinse jar's own water material: its murk follows the rinses
  const rinseMat = new THREE.MeshPhysicalMaterial({ name: 'Water_Rinse', color: '#A9C2BE', roughness: 0.1, transparent: true,
    opacity: 0.55, depthWrite: false, side: THREE.DoubleSide, clearcoat: 1, ior: 1.33 });
  jarRinse.getObjectByName('Jar_Water').traverse((o) => { if (o.isMesh) { o.material = rinseMat; o.renderOrder = 1; } });
  jarClean.getObjectByName('Jar_CapRing').visible = false;
  // cakes need their own materials so each can darken as it drains
  const cakes = PIGMENTS.map((_, i) => G.palette.getObjectByName(`Palette_Cake_${i}`));

  // outside the window: a late-afternoon sky with a low roofline
  const sky = new THREE.Mesh(new THREE.PlaneGeometry(6, 4), new THREE.MeshBasicMaterial({ map: skyTexture(), toneMapped: true }));
  sky.material.color.setScalar(2.2);
  sky.position.set(-3.6, 1.6, -0.8);
  sky.rotation.y = Math.PI / 2;
  scene.add(sky);

  // ------------------------------------------------------------------ lights
  RectAreaLightUniformsLib.init();
  const benchTarget = new THREE.Vector3(-0.1, 0.76, -1.5);
  const sun = new THREE.DirectionalLight('#FFCB98', 2.8);
  const sunDir0 = new THREE.Vector3(-2.6, 1.05, 0.62).normalize();
  sun.position.copy(benchTarget).addScaledVector(sunDir0, 9);
  sun.target.position.copy(benchTarget);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  Object.assign(sun.shadow.camera, { left: -3.2, right: 3.2, top: 2.6, bottom: -2.6, near: 3, far: 16 });
  sun.shadow.bias = -0.0004;
  sun.shadow.normalBias = 0.015;
  sun.shadow.radius = 6;
  sun.shadow.blurSamples = 16;
  scene.add(sun, sun.target);

  const lampAnchor = G.lamp.getObjectByName('Lamp_Light');
  const LAMP = 3.2;
  const lamp = new THREE.SpotLight('#FFB872', LAMP, 0, 0.78, 0.8, 2);
  scene.updateMatrixWorld(true);
  lampAnchor.getWorldPosition(lamp.position);
  const sheetCentre = new THREE.Vector3();
  G.board.getObjectByName('Sheet_Centre').getWorldPosition(sheetCentre);
  lamp.target.position.copy(sheetCentre).add(new THREE.Vector3(0.0, 0, 0.05));
  lamp.castShadow = true;
  lamp.shadow.mapSize.set(1024, 1024);
  lamp.shadow.bias = -0.0003;
  lamp.shadow.camera.near = 0.04;
  lamp.shadow.radius = 4;
  scene.add(lamp, lamp.target);
  const bulbGlow = new THREE.PointLight('#FFB06A', 0.12, 1.2, 2);
  bulbGlow.position.copy(lamp.position);
  scene.add(bulbGlow);
  const hemi = new THREE.HemisphereLight('#CBD8E6', '#7A6248', 0.55);
  scene.add(hemi);
  const skyFill = new THREE.RectAreaLight('#D2E1F2', 2.2, 1.08, 1.68);
  skyFill.position.set(-2.47, 1.8, -0.8);
  skyFill.lookAt(0, 1.4, -0.8);
  scene.add(skyFill);

  // ------------------------------------------------------------------ cut-room walls
  const shadowOnly = new THREE.MeshBasicMaterial({ colorWrite: false, depthWrite: false });
  const walls = {};
  const winParts = [...['Window_Frame', 'Window_Blind', 'Window_Glass', 'Window_Sill'].map((n) => G.room.getObjectByName(n)), plant];
  for (const side of ['N', 'S', 'E', 'W']) {
    const full = G.room.getObjectByName(`Wall_${side}`);
    const stub = G.room.getObjectByName(`Wall_${side}_Stub`);
    const meshes = [];
    const extra = side === 'W' ? winParts : [];
    for (const root of [full, ...extra]) root.traverse((o) => { if (o.isMesh) { o.userData.mat = o.material; meshes.push(o); } });
    stub.traverse((o) => { if (o.isMesh) o.castShadow = false; });
    walls[side] = { meshes, stub, cut: null };
  }
  const plane = { N: (p) => p.z < -2.0, S: (p) => p.z > 2.0, E: (p) => p.x > 2.5, W: (p) => p.x < -2.5 };
  function updateCut(cam) {
    for (const [side, w] of Object.entries(walls)) {
      const cut = plane[side](cam.position);
      if (cut === w.cut) continue;
      w.cut = cut;
      w.stub.visible = cut;
      for (const m of w.meshes) {
        if (m.name === 'Window_Glass') { m.visible = !cut; continue; }
        m.material = cut ? shadowOnly : m.userData.mat;
        m.layers.set(cut ? 1 : 0);
      }
    }
  }

  // ------------------------------------------------------------------ the panel: lifted off the board for the rack dries
  const panel = G.board.getObjectByName('Panel');
  scene.updateMatrixWorld(true);
  scene.attach(panel);
  const panelHome = { p: panel.position.clone(), q: panel.quaternion.clone() };
  const panelBox = new THREE.Box3().setFromObject(panel);
  const pSize = new THREE.Vector3();
  panel.quaternion.identity(); panel.updateMatrixWorld(true);
  new THREE.Box3().setFromObject(panel.getObjectByName('Panel_Carrier'), true).getSize(pSize);
  panel.quaternion.copy(panelHome.q); panel.updateMatrixWorld(true);
  const halfX = pSize.x / 2, halfZ = pSize.z / 2;
  const boardN = new THREE.Vector3(0, 1, 0).applyQuaternion(panelHome.q);
  const qVert = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI / 2); // face south, painting upright
  rack.updateMatrixWorld(true);
  const railStart = rack.getObjectByName('Rack_Rail_Start').getWorldPosition(new THREE.Vector3());
  const railEnd = rack.getObjectByName('Rack_Rail_End').getWorldPosition(new THREE.Vector3());
  const railTop = railStart.y + 0.002; // the panel's lower edge rides between the two rods
  const rackZ = railStart.z;
  const inRackX = railStart.x + 0.02 + halfX;
  const carryKeys = [
    { u: 0, p: panelHome.p.clone(), q: panelHome.q.clone() },
    { u: 0.2, p: panelHome.p.clone().addScaledVector(boardN, 0.06), q: panelHome.q.clone() },
    { u: 0.55, p: new THREE.Vector3(railEnd.x + halfX - 0.02, Math.max(0.905, panelBox.max.y + 0.02) + halfZ, rackZ), q: qVert },
    { u: 0.8, p: new THREE.Vector3(railEnd.x - halfX - 0.03, railTop + halfZ, rackZ), q: qVert },
    { u: 1, p: new THREE.Vector3(inRackX, railTop + halfZ, rackZ), q: qVert },
  ];
  function placePanel(out) {
    let k = 0;
    while (k < carryKeys.length - 2 && out > carryKeys[k + 1].u) k++;
    const A = carryKeys[k], B = carryKeys[k + 1];
    const u = smooth((out - A.u) / (B.u - A.u));
    panel.position.lerpVectors(A.p, B.p, u);
    panel.quaternion.slerpQuaternions(A.q, B.q, u);
  }

  // ------------------------------------------------------------------ anchors for the brush
  const J = ['Arm_Yaw', 'Arm_Shoulder', 'Arm_Elbow', 'Arm_Wrist'].map((n) => G.arm.getObjectByName(n));
  const ik = makeArmIK(J, G.arm);
  const armBase = J[0].getWorldPosition(new THREE.Vector3());
  scene.updateMatrixWorld(true);
  const cornerNodes = ['TL', 'TR', 'BL'].map((c) => panel.getObjectByName(`Sheet_Corner_${c}`));
  const homeCorners = cornerNodes.map((n) => n.getWorldPosition(new THREE.Vector3()));
  const ex = new THREE.Vector3().subVectors(homeCorners[1], homeCorners[0]);
  const ey = new THREE.Vector3().subVectors(homeCorners[2], homeCorners[0]);
  const sheetN = ex.clone().cross(ey).normalize();
  if (sheetN.y < 0) sheetN.negate();
  const W = replay.W, H = replay.H;
  const wellTops = cakes.map((c) => c.getWorldPosition(new THREE.Vector3()));
  const slants = [0, 1].map((j) => G.palette.getObjectByName(`Palette_Slant_${j}`).getWorldPosition(new THREE.Vector3()));
  const rinseCentre = jarRinse.getWorldPosition(new THREE.Vector3());
  const jarWaterBase = jarRinse.getObjectByName('Jar_Water').position.y;
  const potTop = G.brushes.getObjectByName('Brush_Pot').getWorldPosition(new THREE.Vector3()).add(new THREE.Vector3(0, 0.255, 0));
  const parkPos = new THREE.Vector3(0.5, 0.93, -1.42);
  const levels = { wells: new Array(8).fill(1), clean: 0.88, rinse: 0.78 };
  const maxPig = Math.max(...replay.totals.pig, 1);
  const A = {
    sheet: (x, y) => ({ p: homeCorners[0].clone().addScaledVector(ex, x / W).addScaledVector(ey, y / H), n: sheetN }),
    cakeTop: (i, t) => {
      const lv = 1 - 0.7 * replay.amountsAt(t).well[i] / maxPig;
      return new THREE.Vector3(wellTops[i].x, wellTops[i].y + 0.006 * lv, wellTops[i].z);
    },
    slant: (j) => slants[j],
    rinse: () => ({ centre: rinseCentre, surface: rinseCentre.y + jarWaterBase + 0.082 * levels.rinse }),
    rim: () => {
      const d = new THREE.Vector3(armBase.x - rinseCentre.x, 0, armBase.z - rinseCentre.z).normalize();
      return new THREE.Vector3(rinseCentre.x + d.x * 0.041, rinseCentre.y + 0.11, rinseCentre.z + d.z * 0.041);
    },
    pot: () => potTop,
    park: () => ({ p: parkPos, want: A.lean(parkPos, 0.3), contact: null }),
    lean: (p, amount, n = new THREE.Vector3(0, 1, 0)) => {
      const toward = new THREE.Vector3(armBase.x - p.x, 0, armBase.z - p.z).normalize();
      return n.clone().multiplyScalar(-1).addScaledVector(toward, -amount).normalize();
    },
  };
  const choreo = buildChoreo(replay, A);

  // ------------------------------------------------------------------ cameras and controls
  const persp = new THREE.PerspectiveCamera(35, viewW() / viewH(), 0.02, 40);
  const ortho = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 60);
  const ov = cameras.overview;
  function setOrthoFrustum() {
    const a = viewW() / viewH(), h = ov.half_height * Math.max(1, 1.35 / a);
    Object.assign(ortho, { left: -h * a, right: h * a, top: h, bottom: -h });
    ortho.updateProjectionMatrix();
  }
  setOrthoFrustum();
  const ovTarget = new THREE.Vector3().fromArray(ov.target);
  const ovDir = new THREE.Vector3(-Math.sin(ov.azimuth_deg * D2R) * Math.cos(ov.elevation_deg * D2R),
    Math.sin(ov.elevation_deg * D2R), Math.cos(ov.azimuth_deg * D2R) * Math.cos(ov.elevation_deg * D2R));
  ortho.position.copy(ovTarget).addScaledVector(ovDir, 20);
  ortho.lookAt(ovTarget);
  persp.layers.enable(1); ortho.layers.enable(1);
  let active = persp;
  const controls = new OrbitControls(persp, renderer.domElement);
  const ob = cameras.orbit;
  Object.assign(controls, { enableDamping: !STILL, dampingFactor: 0.08, minDistance: 0.25,
    maxDistance: ob.max_distance, maxPolarAngle: ob.max_polar_deg * D2R, screenSpacePanning: true,
    minAzimuthAngle: ob.min_azimuth_deg * D2R, maxAzimuthAngle: ob.max_azimuth_deg * D2R });
  const orthoControls = new OrbitControls(ortho, renderer.domElement);
  Object.assign(orthoControls, { enableRotate: false, minZoom: 0.6, maxZoom: 4, enabled: false, screenSpacePanning: true });
  orthoControls.target.copy(ovTarget);
  const tb = ob.target_box.map((a) => new THREE.Vector3().fromArray(a));
  const cb = ob.camera_box.map((a) => new THREE.Vector3().fromArray(a));
  let presetHolds = false;
  controls.addEventListener('start', () => { presetHolds = false; fly = null; stopTour(); });
  controls.addEventListener('change', () => {
    controls.target.clamp(tb[0], tb[1]);
    if (!presetHolds && persp.position.y < ob.min_height) persp.position.y = ob.min_height;
    persp.position.clamp(cb[0], cb[1]);
  });

  // a narrow (phone) view widens the preset's field of view so the same subject stays in frame
  const fovFor = (f) => { const a = viewW() / viewH(); return a >= 1.2 ? f : Math.min(78, (2 * Math.atan(Math.tan((f * D2R) / 2) * (1.5 / a) * 0.8)) / D2R); };
  let current = 'stroke';
  let fly = null; // { p0, t0, p1, t1, a, dur, fov0, fov1 } in UI seconds
  let uiTime = 0;
  function flyTo(pos, target, fov = persp.fov, dur = 1.0) {
    if (active !== persp) { active = persp; controls.enabled = true; orthoControls.enabled = false; }
    fly = { p0: persp.position.clone(), t0: controls.target.clone(), p1: pos.clone(), t1: target.clone(), a: uiTime, dur, fov0: persp.fov, fov1: fovFor(fov) };
    presetHolds = true;
  }
  function setView(id, instant = false) {
    follow = false;
    if (id === 'overview') {
      fly = null;
      active = ortho; controls.enabled = false; orthoControls.enabled = true;
      ortho.zoom = 1; ortho.updateProjectionMatrix();
      orthoControls.target.copy(ovTarget);
      ortho.position.copy(ovTarget).addScaledVector(ovDir, 20);
    } else {
      const s = cameras.stages.find((c) => c.id === id);
      persp.up.fromArray(s.up ?? [0, 1, 0]);
      const wasOrtho = active === ortho;
      if (instant || wasOrtho || STILL) {
        fly = null;
        active = persp; controls.enabled = true; orthoControls.enabled = false;
        persp.fov = fovFor(s.fov); persp.position.fromArray(s.position); controls.target.fromArray(s.target);
        persp.lookAt(controls.target); persp.updateProjectionMatrix(); controls.update();
        presetHolds = true;
      } else flyTo(new THREE.Vector3().fromArray(s.position), new THREE.Vector3().fromArray(s.target), s.fov);
    }
    updateCut(active);
    current = id;
  }

  // ------------------------------------------------------------------ post: soft AO (N8AO), bloom on lamp and window only
  function makeComposer(cam) {
    const c = new EffectComposer(renderer);
    if (AO) {
      const ao = new N8AOPass(scene, cam, viewW(), viewH());
      Object.assign(ao.configuration, { aoRadius: cam.isOrthographicCamera ? 0.6 : 0.22, distanceFalloff: cam.isOrthographicCamera ? 0.6 : 0.25,
        intensity: cam.isOrthographicCamera ? 2.2 : 2.6, aoSamples: 12, denoiseSamples: 6, denoiseRadius: 10, halfRes: true,
        color: new THREE.Color('#2A1C12'), screenSpaceRadius: false, gammaCorrection: false });
      c.addPass(ao);
    } else c.addPass(new RenderPass(scene, cam));
    c.addPass(new UnrealBloomPass(new THREE.Vector2(viewW(), viewH()), 0.22, 0.4, 3.2));
    c.addPass(new OutputPass());
    return c;
  }
  const composers = { persp: makeComposer(persp), ortho: makeComposer(ortho) };
  function resize() {
    renderer.setSize(viewW(), viewH());
    persp.aspect = viewW() / viewH(); persp.updateProjectionMatrix();
    setOrthoFrustum();
    for (const c of Object.values(composers)) c.setSize(viewW(), viewH());
  }
  addEventListener('resize', resize);

  // ------------------------------------------------------------------ the replay clock
  const STEP = 1 / 60;
  const clock = { t: Number(params.get('t') ?? 0), speed: 1, playing: !STILL && !params.has('paused'), acc: 0 };
  let scores = null, scoresLoading = false;
  function advance(dt) {
    if (!clock.playing) return;
    clock.acc += Math.min(0.1, dt) * clock.speed;
    const n = Math.floor(clock.acc / STEP + 1e-9);
    clock.acc -= n * STEP;
    clock.t = Math.min(replay.duration, clock.t + n * STEP);
    if (clock.t >= replay.duration) clock.playing = false;
  }
  function setTime(t) { clock.t = Math.min(replay.duration, Math.max(0, t)); clock.acc = 0; }

  // ------------------------------------------------------------------ apply the replay at time t
  const rinseTint = (() => {
    // the murk takes the colour of the pigments rinsed out so far
    const out = [];
    let r = 0, g = 0, b = 0, n = 0;
    for (const s of replay.strokeSegs) {
      if (!s.rinse) continue;
      const prev = s.prevStroke;
      const hex = prev.colour;
      r += parseInt(hex.slice(1, 3), 16); g += parseInt(hex.slice(3, 5), 16); b += parseInt(hex.slice(5, 7), 16); n++;
      out.push(new THREE.Color(`rgb(${Math.round(r / n)},${Math.round(g / n)},${Math.round(b / n)})`));
    }
    return out;
  })();
  const clearWater = new THREE.Color('#A9C2BE');
  const tipW = new THREE.Vector3();
  let lastPose = null;
  const scoreCardDrawn = { done: false };
  function apply(t) {
    sheet.showGhosts = ui.ghosts;
    sheet.draw(t);
    const am = replay.amountsAt(t);
    // wells drain with the pigment picked up; the most used well ends at 30 percent
    for (let i = 0; i < 8; i++) {
      const lv = 1 - 0.7 * am.well[i] / maxPig;
      levels.wells[i] = lv;
      cakes[i].scale.y = lv;
    }
    levels.clean = 0.88 - 0.42 * am.water / Math.max(1e-6, replay.totals.water);
    jarClean.getObjectByName('Jar_Water').scale.y = levels.clean;
    levels.rinse = 0.85;
    jarRinse.getObjectByName('Jar_Water').scale.y = levels.rinse;
    const k = Math.floor(am.rinses), f = am.rinses - k;
    const murk = 1 - Math.exp(-am.rinses / 7);
    const tintA = k > 0 ? rinseTint[Math.min(rinseTint.length - 1, k - 1)] : clearWater;
    const tintB = rinseTint[Math.min(rinseTint.length - 1, k)] ?? tintA;
    const tint = tintA.clone().lerp(tintB, f);
    rinseMat.color.copy(clearWater).lerp(tint, Math.min(0.85, murk * 1.2));
    rinseMat.opacity = 0.55 + 0.38 * murk;
    rinseMat.roughness = 0.1 + 0.4 * murk;

    // the panel
    const pn = replay.panelAt(t);
    placePanel(pn.out);
    panel.updateMatrixWorld(true);

    // the brush in the collet: the phase's brush, swapped at the pot midway through a phase dry
    const seg = replay.segAt(t);
    let want = seg ? seg.brush : replay.segs[0].brush;
    if (seg?.brushChange) want = t < (seg.brushChange.a + seg.brushChange.b) / 2 ? seg.brushChange.from : seg.brushChange.to;
    else if (seg && seg.kind !== 'stroke' && seg.nextStroke && seg.kind !== 'phase') want = seg.brush;
    holdBrush(want);
    const pose = choreo.pose(t);
    lastPose = pose;
    lastPose.ik = ik.solve(pose.p, pose.want);
    brushes[brushInHand].tip.getWorldPosition(tipW);

    // light: the lamp brightens while the brush is on the paper; the sun creeps over the run
    const onPaper = pose.contact === 'sheet' ? 1 : 0;
    lamp.intensity = LAMP * (1 + 0.22 * onPaper);
    const runP = Math.min(1, t / replay.duration);
    const sd = sunDir0.clone().applyAxisAngle(new THREE.Vector3(0, 1, 0), (runP - 0.5) * 10 * D2R);
    sd.y = sunDir0.y - 0.08 * runP;
    sd.normalize();
    sun.position.copy(benchTarget).addScaledVector(sd, 9);
    sun.color.set('#FFCB98').lerp(new THREE.Color('#FFB27A'), runP);

    // the score sheet turns over at the end
    if (t >= replay.endOps - 3 && !scores && !scoresLoading) {
      scoresLoading = true;
      fetch(`${runBase}/scores.json`).then((r) => r.json()).then((s) => { scores = s; });
    }
    if (scores && !scoreCardDrawn.done) { drawScoreFront(scoreFront, scores.final); scoreCardDrawn.done = true; }
    const flip = smooth((t - replay.revealAt) / T.revealTurn);
    score.rotation.x = Math.PI * (1 - flip);
    score.position.y = scoreY + 0.035 * Math.sin(Math.PI * flip) + (flip > 0.5 ? -0.0009 + 0.0004 : 0);
  }

  // ------------------------------------------------------------------ dashboard, picking, cards, tour
  const ui = { playing: clock.playing, speed: 1, view: current, touring: false, follow: false, ghosts: true, scores: null };
  let follow = false;
  let tour = null;
  function stopTour() { if (tour) { const sp = tour.savedSpeed ?? 1; tour = null; clock.speed = sp; } }
  const dash = createDashboard(replay, cameras.stages, {
    onPlay: () => { if (clock.t >= replay.duration) setTime(0); clock.playing = !clock.playing; },
    onStep: () => { clock.playing = false; setTime(clock.t + STEP); },
    onSpeed: (s) => { clock.speed = s; },
    onStage: (id) => { stopTour(); setView(id); },
    onScrub: (t) => { stopTour(); setTime(t); },
    onLegend: (i) => { stopTour(); lookAtWell(i); showCard({ kind: 'well', i }); },
    onTour: () => (tour ? stopTour() : startTour()),
    onFollow: () => { follow = !follow; if (follow && active !== persp) setView('stroke'); if (follow) { follow = true; fly = null; } },
    onGhosts: () => { ui.ghosts = !ui.ghosts; },
  });

  function lookAtWell(i) {
    const w = wellTops[i];
    flyTo(new THREE.Vector3(w.x + 0.26, w.y + 0.2, w.z + 0.22), new THREE.Vector3(w.x - 0.01, w.y, w.z), 32);
    current = `well${i}`;
  }

  const detail = document.getElementById('detail'), detailBody = document.getElementById('detail-body');
  let card = null;
  function showCard(c) { card = c; detail.hidden = false; renderCard(); }
  function hideCard() { card = null; detail.hidden = true; }
  const kv = (k, v) => `<div class="rowkv"><span class="muted">${k}</span><span>${v}</span></div>`;
  function renderCard() {
    if (!card) return;
    const t = clock.t, am = replay.amountsAt(t), seg = replay.segAt(t);
    let html = '';
    if (card.kind === 'brush') {
      const st = seg?.kind === 'stroke' ? seg : seg?.nextStroke;
      html = `<h2>Brush: ${brushes[brushInHand].label}</h2>`;
      if (st) {
        const pig = Object.entries(st.o.op.pigments).map(([k, v]) => `${PIGMENTS[PIGMENT_INDEX[k]][1]} ${v.toFixed(2)}`).join(', ');
        html += kv('Stroke', `${st.strokeNo} of ${replay.totals.strokes}${seg === st ? '' : ' (next)'}`)
          + kv('Width', `${(st.o.op.width * replay.pxMm).toFixed(1)} mm (${st.o.op.width.toFixed(1)} px)`)
          + kv('Water', st.o.op.water.toFixed(2)) + `<div class="small">${pig}</div>`
          + kv('Doing', lastPose?.contact === 'sheet' ? 'laying the stroke' : lastPose?.contact?.startsWith('well') ? 'loading at the well'
            : lastPose?.contact === 'rinse' ? 'rinsing' : seg?.kind === 'phase' ? 'changing brush' : 'moving');
      }
      html += `<button class="btn" id="card-follow">${follow ? 'Stop following' : 'Follow the tip'}</button>`;
    } else if (card.kind === 'well') {
      const [k, label] = PIGMENTS[card.i];
      const n = replay.strokeSegs.filter((s) => k in s.o.op.pigments && s.lay[1] <= t).length;
      html = `<h2>${label}</h2>` + kv('Strokes using it', n) + kv('Pigment added', `${am.pig[card.i].toFixed(0)} of ${replay.totals.pig[card.i].toFixed(0)}`)
        + kv('Cake level', `${(levels.wells[card.i] * 100).toFixed(0)} % (staging scale)`);
    } else if (card.kind === 'clean') {
      html = '<h2>Clean water jar</h2>' + kv('Water loaded so far', `${am.water.toFixed(2)} of ${replay.totals.water.toFixed(2)}`) + kv('Level', `${(levels.clean * 100).toFixed(0)} %`)
        + '<div class="small muted">Sum of each stroke\'s water load from the ledger.</div>';
    } else if (card.kind === 'rinse') {
      html = '<h2>Rinse jar</h2>' + kv('Rinses so far', `${Math.floor(am.rinses)} of ${replay.totals.rinses}`)
        + '<div class="small muted">A rinse is staged whenever the next stroke\'s main pigment differs.</div>';
    } else if (card.kind === 'target') {
      html = `<h2>Target</h2><div class="small">${tp.startsWith('synthetic:') ? `Built-in test scene "${tp.split(':')[1]}"` : tp}</div>`;
    } else if (card.kind === 'rack') {
      const p = replay.panelAt(t);
      html = '<h2>Drying rack</h2>' + kv('Panel', p.stage === 'board' ? 'on the board' : p.stage === 'rack' ? 'drying in the rack' : 'on the rail')
        + '<div class="small muted">The ride to the rack is staging; the dry and its length are recorded.</div>';
    }
    detailBody.innerHTML = html;
    const fb = document.getElementById('card-follow');
    if (fb) fb.onclick = () => { follow = !follow; fly = null; if (active !== persp) setView('stroke'); };
  }

  const ray = new THREE.Raycaster();
  ray.layers.set(0);
  let downAt = null;
  renderer.domElement.addEventListener('pointerdown', (e) => { downAt = [e.clientX, e.clientY]; stopTour(); });
  renderer.domElement.addEventListener('pointerup', (e) => {
    if (!downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 5) return;
    const b = renderer.domElement.getBoundingClientRect();
    ray.setFromCamera(new THREE.Vector2(((e.clientX - b.left) / b.width) * 2 - 1, -((e.clientY - b.top) / b.height) * 2 + 1), active);
    const hits = ray.intersectObjects(scene.children, true).filter((h) => h.object.visible && h.object.material?.colorWrite !== false);
    const what = hits.length ? classify(hits[0].object) : null;
    if (what) showCard(what); else hideCard();
  });
  function classify(o) {
    for (let p = o; p; p = p.parent) {
      const n = p.name ?? '';
      if (n.startsWith('Brush_Wash') || n.startsWith('Brush_Round') || n.startsWith('Arm')) return { kind: 'brush' };
      const m = n.match(/^Palette_Cake_(\d)$/); if (m) return { kind: 'well', i: Number(m[1]) };
      if (p === jarClean) return { kind: 'clean' };
      if (p === jarRinse) return { kind: 'rinse' };
      if (n === 'Target_Print') return { kind: 'target' };
      if (p === rack) return { kind: 'rack' };
      if (p === score) { document.getElementById('guess')?.focus(); return null; }
      if (n.startsWith('Floor') || n.startsWith('Wall')) return null;
    }
    return null;
  }

  const events = replay.events();
  const tourOrder = ['prompt', 'calibration', 'load', 'search', 'stroke', 'rinse', 'dry', 'reveal'];
  function startTour() {
    tour = { k: -1, savedSpeed: clock.speed, holdUntil: 0 };
    nextTourStage();
  }
  function nextTourStage() {
    tour.k++;
    if (tour.k >= tourOrder.length) { stopTour(); return; }
    const id = tourOrder[tour.k];
    setView(id);
    tour.event = events[id];
    setTime(Math.max(0, tour.event - (id === 'dry' ? 0.8 : 1.6)));
    clock.playing = true;
  }
  function tourTick() {
    if (!tour) return;
    const d = clock.t - tour.event;
    clock.speed = Math.abs(d) < 0.5 ? 0.25 : 1; // slow down at the event, eased in from the known schedule
    if (d > (tourOrder[tour.k] === 'reveal' ? 2.5 : tourOrder[tour.k] === 'dry' ? 2.0 : 1.4)) nextTourStage();
  }

  addEventListener('keydown', (e) => {
    if (e.target instanceof HTMLInputElement) return;
    const s = cameras.stages.find((c) => c.key === e.key);
    if (s) { stopTour(); setView(s.id); }
    else if (e.key === '0' || e.key === 'o') { stopTour(); setView('overview'); }
    else if (e.key === ' ') { e.preventDefault(); clock.playing = !clock.playing; }
    else if (e.key === '+' || e.key === '=') clock.speed = SPEEDS[Math.min(SPEEDS.length - 1, SPEEDS.indexOf(clock.speed) + 1)] ?? 1;
    else if (e.key === '-') clock.speed = SPEEDS[Math.max(0, SPEEDS.indexOf(clock.speed) - 1)] ?? 1;
    else if (e.key === 'Escape') hideCard();
  });

  // ------------------------------------------------------------------ frame
  let last = performance.now();
  let fpsFrames = 0, fpsT0 = performance.now(), fps = 0;
  function frame() {
    const now = performance.now();
    const dt = (now - last) / 1000;
    last = now;
    uiTime += Math.min(0.1, dt);
    advance(dt);
    tourTick();
    apply(clock.t);
    if (fly) {
      const u = smooth((uiTime - fly.a) / fly.dur);
      persp.position.lerpVectors(fly.p0, fly.p1, u);
      persp.position.y += Math.sin(Math.PI * u) * 0.04 * fly.p0.distanceTo(fly.p1);
      controls.target.lerpVectors(fly.t0, fly.t1, u);
      persp.fov = fly.fov0 + (fly.fov1 - fly.fov0) * u; persp.updateProjectionMatrix();
      persp.lookAt(controls.target);
      if (u >= 1) fly = null;
    }
    if (follow && active === persp && !fly) {
      const off = persp.position.clone().sub(controls.target);
      const len = Math.min(0.75, Math.max(0.32, off.length()));
      off.setLength(len);
      controls.target.lerp(tipW, 0.18);
      persp.position.copy(controls.target).add(off);
    }
    (active === persp ? controls : orthoControls).update();
    updateCut(active);
    ui.playing = clock.playing; ui.speed = clock.speed; ui.view = current; ui.touring = !!tour; ui.follow = follow; ui.scores = scores;
    dash.update(clock.t, ui);
    if (card) renderCard();
    composers[active === persp ? 'persp' : 'ortho'].render();
    fpsFrames++;
    if (now - fpsT0 > 1000) { fps = (fpsFrames * 1000) / (now - fpsT0); fpsFrames = 0; fpsT0 = now; }
  }
  setView(params.get('view') ?? 'stroke', true);

  function stats() {
    let tris = 0, meshes = 0;
    const seen = new Set();
    scene.traverseVisible((o) => {
      if (!o.isMesh) return;
      meshes++;
      const g = o.geometry;
      tris += (g.index ? g.index.count : g.attributes.position.count) / 3;
      seen.add(g);
    });
    renderer.info.reset();
    renderer.info.autoReset = false;
    renderer.render(scene, active);
    const calls = renderer.info.render.calls, rendered = renderer.info.render.triangles;
    renderer.info.autoReset = true;
    return { triangles: Math.round(tris), renderedTrianglesWithShadows: rendered, meshes, uniqueGeometries: seen.size,
      drawCallsIncludingShadows: calls, textures: renderer.info.memory.textures, fps: +fps.toFixed(1) };
  }
  // where the brush is against what it should touch, at time t
  function contactAt(t) {
    apply(t);
    const p = lastPose;
    brushes[brushInHand].tip.getWorldPosition(tipW);
    const res = { t: +t.toFixed(3), contact: p.contact, ikError_mm: +(p.ik.error * 1000).toFixed(2), brush: brushInHand };
    if (p.contact === 'sheet') {
      res.tipToSheet_mm = +(tipW.clone().sub(homeCorners[0]).dot(sheetN) * 1000).toFixed(2);
      const sp = A.sheet(p.onSheet.x, p.onSheet.y).p;
      res.tipToPath_mm = +(tipW.distanceTo(sp) * 1000).toFixed(2);
    }
    if (p.contact?.startsWith('well:')) {
      const i = Number(p.contact.slice(5));
      res.well = i; res.main = replay.segAt(t)?.main; res.tipToCake_mm = +((tipW.y - A.cakeTop(i, t).y) * 1000).toFixed(2);
    }
    return res;
  }
  const project = (v) => {
    const q = v.clone().project(active);
    const b = renderer.domElement.getBoundingClientRect();
    return { x: b.left + (q.x * 0.5 + 0.5) * b.width, y: b.top + (1 - (q.y * 0.5 + 0.5)) * b.height };
  };

  window.__app = {
    setView, views: ['overview', ...cameras.stages.map((s) => s.id)], replay,
    renderOnce: () => { frame(); return true; },
    setTime, getTime: () => clock.t, play: () => { clock.playing = true; }, pause: () => { clock.playing = false; },
    setSpeed: (s) => { clock.speed = s; }, clock: () => ({ ...clock }), events,
    stats, contactAt, scene,
    screenOf: (what) => {
      if (what === 'brush') return project(tipW.clone().add(new THREE.Vector3(0, 0.05, 0)));
      if (what === 'tip') return project(tipW);
      const m = /^well(\d)$/.exec(what); if (m) return project(wellTops[Number(m[1])]);
      if (what === 'score') return project(score.getWorldPosition(new THREE.Vector3()));
      return null;
    },
    camera: () => ({ position: active.position.toArray().map((v) => +v.toFixed(3)), target: controls.target.toArray().map((v) => +v.toFixed(3)) }),
    subjects: () => ({ brush: brushes[brushInHand].o, panel, palette: G.palette, jarRinse, score, arm: G.arm }),
    camObj: () => active,
  };
  apply(clock.t);
  document.body.dataset.ready = '1';
  if (!STILL) renderer.setAnimationLoop(frame);
  else frame();
}

function smooth(x) { x = Math.min(1, Math.max(0, x)); return x * x * (3 - 2 * x); }

function drawScoreFront(tex, f) {
  const c = tex.image, g = c.getContext('2d');
  g.fillStyle = '#F4F0E6'; g.fillRect(0, 0, c.width, c.height);
  g.strokeStyle = '#B5A642'; g.lineWidth = 2; g.strokeRect(8, 8, c.width - 16, c.height - 16);
  g.fillStyle = '#3A3128';
  g.font = '20px Georgia, serif'; g.fillText('total', 22, 44);
  g.font = 'bold 54px Georgia, serif'; g.fillText(f ? f.total.toFixed(2) : '?', 22, 100);
  g.font = '15px system-ui, sans-serif';
  if (f) g.fillText(`lab_l2 ${f.lab_l2.toFixed(0)}  ssim ${f.ssim.toFixed(2)}`, 22, 130);
  g.fillText('lower is better', 22, 152);
  tex.needsUpdate = true;
}

function skyTexture() {
  const c = document.createElement('canvas');
  c.width = 512; c.height = 512;
  const g = c.getContext('2d');
  const grad = g.createLinearGradient(0, 0, 0, 512);
  grad.addColorStop(0, '#9FB7CF'); grad.addColorStop(0.45, '#D9D3C2'); grad.addColorStop(0.62, '#F1D7A8'); grad.addColorStop(1, '#E8C48E');
  g.fillStyle = grad; g.fillRect(0, 0, 512, 512);
  // distant roofs and a tree line, soft and low in contrast
  g.fillStyle = 'rgba(120, 104, 90, 0.55)';
  g.beginPath(); g.moveTo(0, 360);
  const roofs = [[0, 340], [60, 330], [90, 300], [130, 330], [200, 320], [230, 290], [260, 318], [330, 312], [380, 280], [420, 310], [512, 300]];
  for (const [x, y] of roofs) g.lineTo(x, y);
  g.lineTo(512, 512); g.lineTo(0, 512); g.fill();
  g.fillStyle = 'rgba(96, 104, 80, 0.5)';
  for (let i = 0; i < 40; i++) { g.beginPath(); g.arc(i * 14 + 5, 335 + Math.sin(i * 1.7) * 8, 18 + (i % 3) * 5, 0, Math.PI * 2); g.fill(); }
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

async function svgTexture(url, size) {
  const img = new Image();
  img.src = url;
  await img.decode();
  const c = document.createElement('canvas');
  c.width = c.height = size;
  c.getContext('2d').drawImage(img, 0, 0, size, size);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}
