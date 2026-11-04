// Asset preview: ?asset=<name>[&yaw=deg] shows one GLB on a neutral floor with a 1.8 m figure block and a 1 m ruler,
// for scale checks. Not part of the studio itself.
import * as THREE from 'three';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

export async function assetPreview(renderer, load, name) {
  const params = new URLSearchParams(location.search);
  document.querySelectorAll('.panel').forEach((p) => { p.style.display = 'none'; });
  const scene = new THREE.Scene();
  scene.background = new THREE.Color('#D9D3C6');
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  scene.environmentIntensity = 0.6;
  const sun = new THREE.DirectionalLight('#FFE2C0', 2.5);
  sun.position.set(-3, 5, 4);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  Object.assign(sun.shadow.camera, { left: -2, right: 2, top: 2, bottom: -2 });
  scene.add(sun, new THREE.HemisphereLight('#ffffff', '#8a7a66', 0.5));

  const url = name.includes('/') ? `models/${name}.glb` : `models/${name}.glb`;
  const g = (await load(url)).scene;
  g.traverse((o) => { if (o.isMesh) { o.castShadow = o.receiveShadow = true; } });
  scene.add(g);
  const box = new THREE.Box3().setFromObject(g);
  const size = box.getSize(new THREE.Vector3());
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(20, 20), new THREE.MeshStandardMaterial({ color: '#BDB4A4', roughness: 0.9 }));
  ground.rotation.x = -Math.PI / 2; ground.position.y = Math.min(0, box.min.y) - 0.0005; ground.receiveShadow = true;
  scene.add(ground);
  // 1.8 m figure block and a 1 m ruler with 10 cm ticks
  const fig = new THREE.Mesh(new THREE.BoxGeometry(0.45, 1.8, 0.25), new THREE.MeshStandardMaterial({ color: '#6C7A89' }));
  fig.position.set(box.min.x - 0.45, ground.position.y + 0.9, box.min.z - 0.3);
  scene.add(fig);
  for (let i = 0; i < 10; i++) {
    const t = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.004, 0.02),
      new THREE.MeshBasicMaterial({ color: i % 2 ? '#222' : '#eee' }));
    t.position.set(box.min.x + 0.05 + i * 0.1, ground.position.y + 0.002, box.max.z + 0.06);
    scene.add(t);
  }
  const all = new THREE.Box3().setFromObject(scene.getObjectByProperty('isMesh', true) ? g : g).union(new THREE.Box3().setFromObject(fig));
  const big = Math.max(size.x, size.y, size.z) < 0.6 && !params.has('figure') ? box : all;
  if (big === box) fig.visible = false;
  const c = big.getCenter(new THREE.Vector3()), r = big.getSize(new THREE.Vector3()).length() * 0.62;
  const yaw = (+params.get('yaw') || 35) * Math.PI / 180;
  const cam = new THREE.PerspectiveCamera(30, innerWidth / innerHeight, 0.01, 100);
  cam.position.set(c.x + Math.sin(yaw) * r * 2.6, c.y + r * 1.1, c.z + Math.cos(yaw) * r * 2.6);
  cam.lookAt(c);
  renderer.render(scene, cam);
  window.__app = { renderOnce: () => { renderer.render(scene, cam); return true; },
    stats: () => ({ size_m: size.toArray().map((v) => +v.toFixed(3)) }) };
  document.body.dataset.ready = '1';
}
