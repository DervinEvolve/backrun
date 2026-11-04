// Damped least squares IK for the desk arm: base yaw, shoulder, elbow and wrist pitch. The brush tip goes to a
// point with the brush along a wanted direction. It always starts from the same seed for a given target, so the
// pose is a pure function of the target (no warm start), which keeps every frame independent of the one before.
import * as THREE from 'three';

const LIM = [[-Math.PI, Math.PI], [-0.6, 1.6], [0, 2.8], [-1.8, 2.3]];
const tip = new THREE.Vector3(), grip = new THREE.Vector3(), dir = new THREE.Vector3(), base = new THREE.Vector3();

export function makeArmIK(J, root) {
  let tipNode = null, gripNode = null;
  const setBrush = (brushTip, brushGrip) => { tipNode = brushTip; gripNode = brushGrip; };
  const apply = (q) => {
    J[0].rotation.y = q[0]; J[1].rotation.x = q[1]; J[2].rotation.x = q[2]; J[3].rotation.x = q[3];
  };
  const evalq = (q, out) => {
    apply(q);
    tipNode.updateWorldMatrix(true, false);
    tipNode.getWorldPosition(tip);
    gripNode.getWorldPosition(grip);
    dir.subVectors(tip, grip).normalize();
    out[0] = tip.x; out[1] = tip.y; out[2] = tip.z; out[3] = dir.x; out[4] = dir.y; out[5] = dir.z;
  };
  const f0 = new Float64Array(6), f1 = new Float64Array(6);
  const Jm = Array.from({ length: 6 }, () => new Float64Array(4));

  // W: metres of tip error worth one unit of direction error
  function core(p, want, q, W, iters) {
    const goal = [p.x, p.y, p.z, want.x * W, want.y * W, want.z * W];
    const res = (f) => [goal[0] - f[0], goal[1] - f[1], goal[2] - f[2], goal[3] - f[3] * W, goal[4] - f[4] * W, goal[5] - f[5] * W];
    let lambda = 0.02;
    for (let it = 0; it < iters; it++) {
      evalq(q, f0);
      const r = res(f0);
      const err = r.reduce((s, v) => s + v * v, 0);
      if (err < 1e-12) break;
      for (let j = 0; j < 4; j++) {
        const h = 1e-4, o = q[j];
        q[j] = o + h; evalq(q, f1); q[j] = o;
        for (let k = 0; k < 6; k++) Jm[k][j] = ((f1[k] - f0[k]) / h) * (k >= 3 ? W : 1);
      }
      const A = Array.from({ length: 4 }, () => new Float64Array(5));
      for (let a = 0; a < 4; a++) {
        for (let b = 0; b < 4; b++) { let s = 0; for (let k = 0; k < 6; k++) s += Jm[k][a] * Jm[k][b]; A[a][b] = s + (a === b ? lambda * lambda : 0); }
        let s = 0; for (let k = 0; k < 6; k++) s += Jm[k][a] * r[k]; A[a][4] = s;
      }
      const dq = gauss(A);
      const trial = q.map((v, j) => Math.min(LIM[j][1], Math.max(LIM[j][0], v + dq[j])));
      evalq(trial, f1);
      const err2 = res(f1).reduce((s, v) => s + v * v, 0);
      if (err2 < err) { for (let j = 0; j < 4; j++) q[j] = trial[j]; lambda = Math.max(0.002, lambda * 0.5); } else lambda *= 3;
    }
    return q;
  }

  // p: world target for the tip; want: unit direction grip -> tip. The direction gives way if the tip cannot reach.
  function solve(p, want) {
    J[0].getWorldPosition(base);
    let q = core(p, want, [Math.atan2(p.x - base.x, p.z - base.z), 0.45, 1.5, 0.7], 0.06, 30);
    evalq(q, f0);
    if (Math.hypot(f0[0] - p.x, f0[1] - p.y, f0[2] - p.z) > 2e-4) q = core(p, want, q, 0.008, 25);
    apply(q);
    root.updateMatrixWorld(true);
    tipNode.getWorldPosition(tip);
    return { q, error: tip.distanceTo(p) };
  }
  return { solve, setBrush };
}

function gauss(A) {
  const n = 4;
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    [A[c], A[p]] = [A[p], A[c]];
    const d = A[c][c] || 1e-12;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = A[r][c] / d;
      for (let k = c; k <= n; k++) A[r][k] -= f * A[c][k];
    }
  }
  return [0, 1, 2, 3].map((i) => A[i][4] / (A[i][i] || 1e-12));
}
