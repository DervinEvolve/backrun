// Backrun studio, Pass 1: the assembled room, its lights, the stage cameras and the overview.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { RectAreaLightUniformsLib } from 'three/addons/lights/RectAreaLightUniformsLib.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { GTAOPass } from 'three/addons/postprocessing/GTAOPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import cameras from './data/cameras.json';
import layout from './data/layout.json';
import * as TX from './textures.js';
import { assetPreview } from './preview.js';

const params = new URLSearchParams(location.search);
const STILL = params.has('still');
const AO = params.get('ao') !== '0';
const FIGURE = params.has('figure');
const D2R = Math.PI / 180;

const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(STILL ? 1 : Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
renderer.outputColorSpace = THREE.SRGBColorSpace;
document.getElementById('stage').appendChild(renderer.domElement);

const loader = new GLTFLoader().setMeshoptDecoder(MeshoptDecoder);
const load = (url) => new Promise((res, rej) => loader.load(url, res, undefined, rej));

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

  const names = ['room', 'bench', 'board', 'arm', 'brushes', 'palette', 'jar', 'rack', 'stool', 'sidetable', 'shelf',
    'lamp', 'corkboards', 'props', 'mannequin'];
  const [G, boxA, boxB] = await Promise.all([
    Promise.all(names.map((n) => load(`models/${n}.glb`))).then((a) => Object.fromEntries(a.map((g, i) => [names[i], g.scene]))),
    load('models/simready/box_cardboard_a01.glb').then((g) => g.scene),
    load('models/simready/box_cardboard_b01.glb').then((g) => g.scene),
  ]);
  const sheetTex = await new THREE.TextureLoader().loadAsync('placeholder/wet_in_wet.png');
  sheetTex.colorSpace = THREE.SRGBColorSpace;
  sheetTex.flipY = false;
  sheetTex.anisotropy = 8;
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
  place(G.rack, 'rack');
  place(G.brushes, 'brushes');
  place(G.shelf, 'shelf');
  place(G.sidetable, 'sidetable');
  place(G.lamp, 'lamp');
  const jarClean = place(G.jar, 'jar_clean');
  const jarRinse = place(G.jar.clone(true), 'jar_rinse');
  place(take(G.corkboards, 'CorkStrip'), 'corkstrip');
  place(take(G.corkboards, 'QueueBoard'), 'queueboard');
  const cup = place(take(G.props, 'TeaCup'), 'teacup');
  place(take(G.props, 'Plant'), 'plant');
  place(take(G.props, 'RopeRail'), 'roperail');
  const score = place(take(G.props, 'ScoreCard'), 'scorecard');
  score.rotation.x = Math.PI; // face down: the result side lies on the bench
  score.position.y += 0.0009;
  const cal = place(take(G.props, 'CalStrip'), 'calstrip');
  cal.rotation.order = 'YXZ';
  cal.rotation.x = 30 * D2R;
  cal.position.y += 0.017 * Math.sin(30 * D2R);
  place(boxA, 'box_a');
  place(boxB, 'box_b');
  place(boxA.clone(true), 'box_floor');
  for (const [i, x] of layout.stools.x.entries()) {
    const s = G.stool.clone(true);
    s.position.set(x, 0, layout.stools.z);
    s.rotation.y = (i * 37 % 20 - 10) * D2R;
    scene.add(s);
  }
  const seated = take(G.mannequin, 'Mannequin_Seated');
  seated.position.set(layout.stools.x[layout.visitor.stool], 0, layout.stools.z);
  seated.rotation.y = Math.PI; // faces the bench (north)
  scene.add(seated);
  const standing = take(G.mannequin, 'Mannequin_Standing');
  standing.position.set(1.3, 0, -0.55);
  standing.rotation.y = Math.PI * 0.82;
  standing.visible = FIGURE;
  scene.add(standing);

  // the painting brush (size 12 round) in the collet; the other two stay in the pot
  const brush = G.brushes.getObjectByName('Brush_Round12');
  const mount = G.arm.getObjectByName('Arm_BrushMount');
  mount.add(brush);
  brush.position.set(0, 0, 0);
  brush.rotation.set(Math.PI, 0, 0);

  // well levels and jar states (placeholder values until the ledger drives them)
  [0.95, 0.7, 0.55, 0.85, 0.6, 0.9, 0.8, 0.65].forEach((v, i) => {
    G.palette.getObjectByName(`Palette_Cake_${i}`).scale.y = v;
  });
  jarClean.getObjectByName('Jar_Water').scale.y = 0.86;
  jarClean.getObjectByName('Jar_CapRing').visible = false;
  jarRinse.getObjectByName('Jar_Water').scale.y = 0.74;

  // Sepia decal on the sketchbook spine
  const spine = G.shelf.getObjectByName('Sketchbook_search_v0_SpineBadge');
  const decal = new THREE.Mesh(new THREE.PlaneGeometry(0.017, 0.017),
    new THREE.MeshStandardMaterial({ map: badgeTex, transparent: true, roughness: 0.5, alphaTest: 0.05 }));
  decal.position.z = 0.0006;
  spine.add(decal);

  // ------------------------------------------------------------------ materials
  const wood = TX.woodTexture(1), woodStain = TX.woodTexture(2, 512, true), floor = TX.floorTexture();
  const brick = TX.brickTexture(), plaster = TX.plasterTexture(), cork = TX.corkTexture(), paperN = TX.paperNormal();
  const rep = (t, s) => { const c = t.clone(); c.repeat.set(1 / s, 1 / s); c.needsUpdate = true; return c; };
  const withSet = (m, set, tile, ns = 1) => {
    m.map = rep(set.map, tile);
    if (set.normalMap) { m.normalMap = rep(set.normalMap, tile); m.normalScale.set(ns, ns); }
    if (set.roughnessMap) m.roughnessMap = rep(set.roughnessMap, tile);
  };
  const cards = {
    Target_Print: TX.cardTexture([['Target print', 38], ['placeholder until a run', 26], ['folder is loaded', 26]],
      { w: 512, h: 384, bg: '#E9E4D8', top: 150 }),
    Prompt_Card: TX.cardTexture([['Built-in test scene:', 30], ['horizon', 44, 'Georgia, serif']], { w: 512, h: 352 }),
    Credit_Slip: TX.cardTexture([['target: staging placeholder', 30, 'system-ui, sans-serif']], { w: 512, h: 84, top: 54 }),
    ScoreCard_Back: TX.cardTexture([], {
      w: 256, h: 176, bg: '#5E4B3C', draw: (g, w, h) => {
        g.strokeStyle = '#B5A642'; g.lineWidth = 3; g.strokeRect(10, 10, w - 20, h - 20);
      },
    }),
  };
  const shared = new Map();
  const glassMat = (src, color, opacity) => new THREE.MeshPhysicalMaterial({
    name: src.name, color, roughness: 0.04, metalness: 0, transparent: true, opacity, ior: 1.5,
    clearcoat: 1, clearcoatRoughness: 0.05, envMapIntensity: 1.6, depthWrite: false,
  });
  function upgrade(m) {
    if (shared.has(m.name)) return shared.get(m.name);
    let out = m;
    const n = m.name;
    if (n === 'Floor_Oak') withSet(m, floor, floor.tile, 0.6);
    else if (n === 'Brick_Limewash') withSet(m, brick, brick.tile, 1.2);
    else if (n === 'Plaster') withSet(m, plaster, plaster.tile, 0.5);
    else if (n === 'Wood_BenchTop') withSet(m, woodStain, 0.9, 0.5);
    else if (['Wood', 'Wood_Walnut', 'Wood_Beech', 'Window_Paint', 'Handle_Natural'].includes(n)) withSet(m, wood, 0.6, 0.35);
    else if (n === 'Cork') withSet(m, cork, cork.tile, 0.8);
    else if (n === 'Glass') { out = glassMat(m, '#F2EFE6', 0.16); }
    else if (n === 'Glass_Amber') { out = glassMat(m, '#8A5524', 0.7); }
    else if (n === 'Water_Clean') {
      out = new THREE.MeshPhysicalMaterial({ name: n, color: '#D4E2E2', roughness: 0.02, transparent: true, opacity: 0.32,
        clearcoat: 1, ior: 1.33, depthWrite: false, envMapIntensity: 1.2 });
    } else if (n === 'Paper_Sheet') {
      m.map = sheetTex; m.color.set('#ffffff'); m.normalMap = paperN; m.normalScale.set(0.25, 0.25); m.roughness = 0.88;
    } else if (['Paper', 'Carrier_Board', 'Kraft_Tape', 'Sketchbook_Cover', 'Rag_Cloth'].includes(n)) {
      m.normalMap = rep(paperN, 0.15); m.normalScale.set(0.3, 0.3);
    } else if (cards[n]) { m.map = cards[n]; m.color.set('#ffffff'); }
    else if (n.startsWith('Queue_Slip')) {
      m.map = TX.cardTexture([['gallery', 40, 'system-ui, sans-serif'], ['recorded run', 34]], { w: 384, h: 216, top: 80 });
      m.color.set('#ffffff');
    } else if (n.startsWith('Brass') || n === 'Nickel') m.envMapIntensity = 1.4;
    else if (n === 'Lamp_Bulb') { m.toneMapped = false; m.emissiveIntensity = 6; }
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
  const rinseMat = shared.get('Water_Clean').clone();
  rinseMat.color.set('#8E8A7C'); rinseMat.opacity = 0.62;
  jarRinse.getObjectByName('Jar_Water').traverse((o) => { if (o.isMesh) o.material = rinseMat; });

  // outside the window: a late-afternoon sky with a low roofline
  const sky = new THREE.Mesh(new THREE.PlaneGeometry(6, 4), new THREE.MeshBasicMaterial({ map: skyTexture(), toneMapped: true }));
  sky.material.color.setScalar(2.2);
  sky.position.set(-3.6, 1.6, -0.8);
  sky.rotation.y = Math.PI / 2;
  scene.add(sky);

  // ------------------------------------------------------------------ lights
  RectAreaLightUniformsLib.init();
  const benchTarget = new THREE.Vector3(-0.1, 0.76, -1.5);
  const sun = new THREE.DirectionalLight('#FFC58F', 3.4); // about 3500 K
  const sunDir = new THREE.Vector3(-2.6, 1.05, 0.62).normalize();
  sun.position.copy(benchTarget).addScaledVector(sunDir, 9);
  sun.target.position.copy(benchTarget);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  Object.assign(sun.shadow.camera, { left: -3.2, right: 3.2, top: 2.6, bottom: -2.6, near: 3, far: 16 });
  sun.shadow.bias = -0.0004;
  sun.shadow.normalBias = 0.015;
  sun.shadow.radius = 4;
  scene.add(sun, sun.target);

  const lampAnchor = G.lamp.getObjectByName('Lamp_Light');
  const lamp = new THREE.SpotLight('#FFB06A', 9, 0, 0.78, 0.75, 2); // about 2700 K
  scene.updateMatrixWorld(true);
  lampAnchor.getWorldPosition(lamp.position);
  const sheetCentre = new THREE.Vector3();
  G.board.getObjectByName('Sheet_Centre').getWorldPosition(sheetCentre);
  lamp.target.position.copy(sheetCentre).add(new THREE.Vector3(0.0, 0, 0.05));
  lamp.castShadow = true;
  lamp.shadow.mapSize.set(1024, 1024);
  lamp.shadow.bias = -0.0003;
  lamp.shadow.camera.near = 0.04;
  lamp.shadow.radius = 3;
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
  const winParts = ['Window_Frame', 'Window_Blind', 'Window_Glass'].map((n) => G.room.getObjectByName(n));
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
      }
    }
  }

  // ------------------------------------------------------------------ arm pose (Pass 1: a static IK pose on the sheet)
  const J = ['Arm_Yaw', 'Arm_Shoulder', 'Arm_Elbow', 'Arm_Wrist'].map((n) => G.arm.getObjectByName(n));
  const tipNode = brush.getObjectByName('Brush_Round12_Tip');
  const corners = ['TL', 'TR', 'BL'].map((c) => G.board.getObjectByName(`Sheet_Corner_${c}`).getWorldPosition(new THREE.Vector3()));
  const sheetN = new THREE.Vector3().subVectors(corners[1], corners[0]).cross(new THREE.Vector3().subVectors(corners[2], corners[0])).normalize();
  if (sheetN.y < 0) sheetN.negate();
  const poseOnSheet = (u, v) => {
    // u, v in 0..1 across the sheet from its top-left corner
    const p = corners[0].clone().addScaledVector(new THREE.Vector3().subVectors(corners[1], corners[0]), u)
      .addScaledVector(new THREE.Vector3().subVectors(corners[2], corners[0]), v);
    return solveIK(J, tipNode, brush, p, sheetN, G.arm);
  };
  const ik = poseOnSheet(0.36, 0.42);

  // ------------------------------------------------------------------ cameras and controls
  const persp = new THREE.PerspectiveCamera(35, innerWidth / innerHeight, 0.02, 40);
  const ortho = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 60);
  const ov = cameras.overview;
  function setOrthoFrustum() {
    const a = innerWidth / innerHeight, h = ov.half_height;
    Object.assign(ortho, { left: -h * a, right: h * a, top: h, bottom: -h });
    ortho.updateProjectionMatrix();
  }
  setOrthoFrustum();
  const ovTarget = new THREE.Vector3().fromArray(ov.target);
  const ovDir = new THREE.Vector3(-Math.sin(ov.azimuth_deg * D2R) * Math.cos(ov.elevation_deg * D2R),
    Math.sin(ov.elevation_deg * D2R), Math.cos(ov.azimuth_deg * D2R) * Math.cos(ov.elevation_deg * D2R));
  ortho.position.copy(ovTarget).addScaledVector(ovDir, 20);
  ortho.lookAt(ovTarget);

  let active = persp;
  const controls = new OrbitControls(persp, renderer.domElement);
  const ob = cameras.orbit;
  Object.assign(controls, { enableDamping: !STILL, dampingFactor: 0.08, minDistance: ob.min_distance,
    maxDistance: ob.max_distance, maxPolarAngle: ob.max_polar_deg * D2R, screenSpacePanning: true });
  const orthoControls = new OrbitControls(ortho, renderer.domElement);
  Object.assign(orthoControls, { enableRotate: false, minZoom: 0.6, maxZoom: 4, enabled: false, screenSpacePanning: true });
  orthoControls.target.copy(ovTarget);
  const tb = ob.target_box.map((a) => new THREE.Vector3().fromArray(a));
  let presetHolds = false; // a preset may sit below the free camera's minimum height until the user drags
  controls.addEventListener('start', () => { presetHolds = false; });
  controls.addEventListener('change', () => {
    controls.target.clamp(tb[0], tb[1]);
    if (!presetHolds && persp.position.y < ob.min_height) persp.position.y = ob.min_height;
  });

  const label = document.getElementById('view-name');
  function setView(id) {
    if (id === 'overview') {
      active = ortho; controls.enabled = false; orthoControls.enabled = true;
      ortho.zoom = 1; ortho.updateProjectionMatrix();
      label.textContent = `0  ${ov.name}`;
    } else {
      const s = cameras.stages.find((c) => c.id === id);
      active = persp; controls.enabled = true; orthoControls.enabled = false;
      persp.fov = s.fov; persp.up.fromArray(s.up ?? [0, 1, 0]);
      persp.position.fromArray(s.position); controls.target.fromArray(s.target);
      persp.lookAt(controls.target); persp.updateProjectionMatrix();
      controls.update();
      presetHolds = true;
      label.textContent = `${s.key}  ${s.name}`;
    }
    updateCut(active);
    current = id;
  }
  let current = 'stroke';

  // ------------------------------------------------------------------ post: AO, bloom on lamp and window only, output
  function makeComposer(cam) {
    const c = new EffectComposer(renderer);
    c.addPass(new RenderPass(scene, cam));
    if (AO) {
      const ao = new GTAOPass(scene, cam, innerWidth, innerHeight);
      ao.updateGtaoMaterial({ radius: cam.isOrthographicCamera ? 0.5 : 0.25, distanceExponent: 1.5, thickness: 1.5, scale: 1.0 });
      ao.blendIntensity = 0.85;
      c.addPass(ao);
    }
    c.addPass(new UnrealBloomPass(new THREE.Vector2(innerWidth, innerHeight), 0.32, 0.5, 2.6));
    c.addPass(new OutputPass());
    return c;
  }
  const composers = { persp: makeComposer(persp), ortho: makeComposer(ortho) };

  addEventListener('resize', () => {
    renderer.setSize(innerWidth, innerHeight);
    persp.aspect = innerWidth / innerHeight; persp.updateProjectionMatrix();
    setOrthoFrustum();
    for (const c of Object.values(composers)) c.setSize(innerWidth, innerHeight);
  });
  addEventListener('keydown', (e) => {
    const s = cameras.stages.find((c) => c.key === e.key);
    if (s) setView(s.id);
    else if (e.key === '0' || e.key === 'o') setView('overview');
  });
  renderer.domElement.addEventListener('dblclick', () => setView(current === 'overview' ? 'stroke' : 'overview'));

  function frame() {
    (active === persp ? controls : orthoControls).update();
    updateCut(active);
    composers[active === persp ? 'persp' : 'ortho'].render();
  }
  setView(params.get('view') ?? 'stroke');

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
    const calls = renderer.info.render.calls;
    renderer.info.autoReset = true;
    const tipToSheet = new THREE.Vector3().subVectors(tipNode.getWorldPosition(new THREE.Vector3()), corners[0]).dot(sheetN);
    return { triangles: Math.round(tris), meshes, uniqueGeometries: seen.size, drawCallsIncludingShadows: calls,
      textures: renderer.info.memory.textures, ik: { ...ik, tipToSheetPlane_mm: +(tipToSheet * 1000).toFixed(2) } };
  }

  window.__app = {
    setView, views: ['overview', ...cameras.stages.map((s) => s.id)],
    renderOnce: () => { frame(); return true; },
    stats, poseOnSheet, scene,
  };
  document.body.dataset.ready = '1';
  if (!STILL) renderer.setAnimationLoop(frame);
  else frame();
}

// coordinate descent IK on yaw, shoulder, elbow and wrist; the brush tip goes to p,
// with the brush leaning about 25 degrees off the sheet normal back toward the arm
function solveIK(J, tipNode, brush, p, n, root) {
  const tip = new THREE.Vector3(), grip = new THREE.Vector3(), dir = new THREE.Vector3();
  const base = new THREE.Vector3();
  J[0].getWorldPosition(base);
  const toward = new THREE.Vector3(base.x - p.x, 0, base.z - p.z).normalize();
  // want = direction from grip to tip: mostly into the sheet, leaning away from the base
  const want = n.clone().multiplyScalar(-1).addScaledVector(toward, -0.42).normalize();
  const lim = [[-Math.PI, Math.PI], [-0.4, 1.5], [0, 2.7], [-1.6, 2.2]];
  const q = [Math.atan2(p.x - base.x, p.z - base.z), 0.5, 1.4, 0.6];
  const apply = () => {
    J[0].rotation.y = q[0]; J[1].rotation.x = q[1]; J[2].rotation.x = q[2]; J[3].rotation.x = q[3];
    root.updateMatrixWorld(true);
  };
  const cost = () => {
    apply();
    tipNode.getWorldPosition(tip); brush.getWorldPosition(grip);
    dir.subVectors(tip, grip).normalize();
    return tip.distanceToSquared(p) + 0.0004 * (1 - dir.dot(want));
  };
  let best = cost(), step = 0.3;
  for (let it = 0; it < 4000 && step > 1e-6; it++) {
    let improved = false;
    for (let j = 0; j < 4; j++) {
      for (const s of [step, -step]) {
        const old = q[j];
        q[j] = Math.min(lim[j][1], Math.max(lim[j][0], old + s));
        const c = cost();
        if (c < best - 1e-14) { best = c; improved = true; } else q[j] = old;
      }
    }
    if (!improved) step *= 0.5;
  }
  apply();
  tipNode.getWorldPosition(tip);
  return { tipError_mm: +(tip.distanceTo(p) * 1000).toFixed(2), angles_deg: q.map((a) => +(a / D2R).toFixed(1)) };
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
