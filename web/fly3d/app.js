// Fly 3D front end: draws what the Python brain simulation reports, and sends your clicks back.
// Nothing here decides behaviour; the fly's movement comes from the server (neurons -> motor readout).
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const $ = (id) => document.getElementById(id);
const api = (op, args = {}) => fetch("/api/cmd", { method: "POST", body: JSON.stringify({ op, ...args }) });

const ROLE_COLORS = {
  sense_looming: [255, 120, 40], sense_pursuit: [90, 230, 120], sense_hearing: [110, 160, 255],
  out_escape: [255, 245, 140], out_steer: [120, 255, 200], middle_steer: [120, 255, 200],
  middle_gf: [255, 190, 90], middle_feedback: [70, 85, 130],
  sense_touch: [255, 120, 200], relay_vab3: [255, 120, 200], court_p1: [255, 90, 170], middle_court: [255, 150, 210],
  out_song: [255, 60, 150], sense_smell: [190, 150, 255], smell_pn: [190, 150, 255],
  sense_taste: [255, 200, 80], middle_taste: [255, 210, 110], out_feed: [255, 230, 60],
  out_flight: [120, 220, 255], out_wing_power: [80, 200, 255], out_wing_steer: [80, 200, 255],
};
const LEGEND = [["looming", "sense_looming"], ["female-chase", "sense_pursuit"], ["ears", "sense_hearing"],
  ["escape", "out_escape"], ["courtship", "court_p1"], ["taste/eating", "sense_taste"], ["smell", "sense_smell"],
  ["flight/wings", "out_flight"], ["steering", "out_steer"]];
const BEHAV = { escape: "#ffd84f", courtship: "#ff6ebe", steering: "#78ffc8", feeding: "#ffaa46", vision: "#8cc8ff" };
const LEVELS = [["escape", "Escape", "#ffd84f"], ["chase", "Chase female", "#ff6ebe"], ["song", "Song", "#ff3c96"],
  ["feeding", "Eating", "#ffaa46"], ["smell", "Smell", "#be96ff"], ["flight", "Wing power", "#50c8ff"]];

let S = null, W = [40, 40], tool = "food", chase = true;
// ?cam=wide: follow the fly from further back and higher (shows the whole scene)
const WIDE = new URLSearchParams(location.search).get("cam") === "wide";
const CAM_BACK = WIDE ? 13 : 7, CAM_UP = WIDE ? 6.5 : 3.2;

// ------------------------------------------------------------------ 3D world
const canvas = $("scene");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1c26);
const camera = new THREE.PerspectiveCamera(55, 1, 0.05, 300);
camera.position.set(0, 9, 18);
const orbit = new OrbitControls(camera, canvas);
orbit.enabled = false;
scene.add(new THREE.HemisphereLight(0xdde8ff, 0x3a5a2a, 1.1));
const sun = new THREE.DirectionalLight(0xfff2dd, 1.0);      // shadow-casting key light
sun.position.set(-8, 30, 6); sun.castShadow = true;
sun.shadow.camera.left = -30; sun.shadow.camera.right = 30; sun.shadow.camera.top = 30; sun.shadow.camera.bottom = -30;
sun.shadow.mapSize.set(1024, 1024);
scene.add(sun);

function groundTexture() {
  const c = document.createElement("canvas"); c.width = c.height = 512;
  const g = c.getContext("2d");
  g.fillStyle = "#4f7d3a"; g.fillRect(0, 0, 512, 512);
  for (let i = 0; i < 9000; i++) {
    g.fillStyle = `hsl(${95 + Math.random() * 30}, ${35 + Math.random() * 25}%, ${22 + Math.random() * 18}%)`;
    g.fillRect(Math.random() * 512, Math.random() * 512, 2, 2 + Math.random() * 4);
  }
  const t = new THREE.CanvasTexture(c); t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(6, 6);
  t.colorSpace = THREE.SRGBColorSpace; return t;
}
const ground = new THREE.Mesh(new THREE.PlaneGeometry(40, 40), new THREE.MeshStandardMaterial({ map: groundTexture() }));
ground.rotation.x = -Math.PI / 2; ground.receiveShadow = true; scene.add(ground);
// the world is a closed box: 4 walls + ceiling. One-sided (they face inward), so a camera outside the box
// looks straight through them instead of being blocked.
const BOX_H = 12;
function panelTexture(base, line, n) {
  const c = document.createElement("canvas"); c.width = c.height = 256;
  const g = c.getContext("2d"); g.fillStyle = base; g.fillRect(0, 0, 256, 256);
  for (let i = 0; i < 1800; i++) { g.fillStyle = `rgba(0,0,0,${Math.random() * 0.05})`; g.fillRect(Math.random() * 256, Math.random() * 256, 3, 3); }
  g.strokeStyle = line; g.lineWidth = 3; g.strokeRect(1.5, 1.5, 253, 253);
  const t = new THREE.CanvasTexture(c); t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(n, n * BOX_H / 40);
  t.colorSpace = THREE.SRGBColorSpace; return t;
}
const wallMat = new THREE.MeshStandardMaterial({ map: panelTexture("#d8cdb8", "#b3a488", 8), side: THREE.FrontSide, roughness: 0.9 });
for (const [x, z, ry] of [[0, -20, 0], [0, 20, Math.PI], [-20, 0, Math.PI / 2], [20, 0, -Math.PI / 2]]) {
  const w = new THREE.Mesh(new THREE.PlaneGeometry(40, BOX_H), wallMat);
  w.position.set(x, BOX_H / 2, z); w.rotation.y = ry; w.receiveShadow = true; scene.add(w);
}
const ceilTex = panelTexture("#ecebe6", "#c9c6bb", 8); ceilTex.repeat.set(8, 8);
const ceiling = new THREE.Mesh(new THREE.PlaneGeometry(40, 40), new THREE.MeshStandardMaterial({ map: ceilTex, side: THREE.FrontSide }));
ceiling.rotation.x = Math.PI / 2; ceiling.position.y = BOX_H; scene.add(ceiling);
for (const [x, z] of [[-10, -10], [10, -10], [-10, 10], [10, 10]]) {   // ceiling lamps
  const lamp = new THREE.Mesh(new THREE.BoxGeometry(3, 0.1, 3), new THREE.MeshBasicMaterial({ color: 0xfffbe8 }));
  lamp.position.set(x, BOX_H - 0.06, z); scene.add(lamp);
  const pl = new THREE.PointLight(0xfff4dd, 60, 30, 1.6); pl.position.set(x, BOX_H - 0.8, z); scene.add(pl);
}
// decorative plants (not part of the simulation)
const rnd = (() => { let s = 7; return () => (s = (s * 16807) % 2147483647) / 2147483647; })();
for (let i = 0; i < 70; i++) {
  const h = 0.6 + rnd() * 1.8;
  const m = new THREE.Mesh(new THREE.ConeGeometry(0.12 + rnd() * 0.2, h, 5),
    new THREE.MeshStandardMaterial({ color: new THREE.Color().setHSL(0.25 + rnd() * 0.08, 0.5, 0.25 + rnd() * 0.12) }));
  const edge = rnd() < 0.5;
  m.position.set((rnd() - 0.5) * 40, h / 2, edge ? (rnd() < 0.5 ? -19 : 19) - (rnd() - 0.5) * 2 : (rnd() - 0.5) * 40);
  if (!edge && rnd() < 0.6) continue;
  m.castShadow = true; scene.add(m);
}

const toScene = (p) => new THREE.Vector3(p[0] - W[0] / 2, p[2] ?? 0, p[1] - W[1] / 2);
const yawOf = (h) => -Math.atan2(h[1], h[0]);

function makeFly(bodyColor, eyeColor, scale = 1) {
  const g = new THREE.Group();
  const mat = new THREE.MeshStandardMaterial({ color: bodyColor, roughness: 0.6 });
  const abdomen = new THREE.Mesh(new THREE.SphereGeometry(0.5, 16, 12), mat);
  abdomen.scale.set(0.9, 0.45, 0.45); abdomen.position.x = -0.25; abdomen.castShadow = true; g.add(abdomen);
  const thorax = new THREE.Mesh(new THREE.SphereGeometry(0.25, 14, 10), mat);
  thorax.position.x = 0.22; g.add(thorax);
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.17, 12, 10), mat);
  head.position.x = 0.5; g.add(head);
  const eyeM = new THREE.MeshStandardMaterial({ color: eyeColor, roughness: 0.3 });
  for (const s of [-1, 1]) {
    const e = new THREE.Mesh(new THREE.SphereGeometry(0.11, 10, 8), eyeM);
    e.position.set(0.55, 0.05, 0.12 * s); g.add(e);
  }
  const wingM = new THREE.MeshStandardMaterial({ color: 0xdfe8ff, transparent: true, opacity: 0.55, side: THREE.DoubleSide });
  g.wings = [];
  for (const s of [-1, 1]) {
    const pivot = new THREE.Group(); pivot.position.set(0.2, 0.18, 0.08 * s);
    const w = new THREE.Mesh(new THREE.CircleGeometry(0.5, 16), wingM);
    w.scale.set(1.0, 0.35, 1); w.rotation.x = -Math.PI / 2; w.position.set(-0.35, 0, 0.12 * s);
    pivot.add(w); pivot.side = s; g.add(pivot); g.wings.push(pivot);
  }
  const prob = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 0.35), new THREE.MeshStandardMaterial({ color: 0xffdc78 }));
  prob.rotation.z = Math.PI / 2 + 0.6; prob.position.set(0.62, -0.18, 0); prob.visible = false; g.add(prob); g.prob = prob;
  for (let k = 0; k < 3; k++) for (const s of [-1, 1]) {
    const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, 0.5), mat);
    leg.position.set(0.3 - k * 0.18, -0.2, 0.2 * s); leg.rotation.x = 0.7 * s; g.add(leg);
  }
  g.scale.setScalar(scale);
  return g;
}
function makePredator() {
  const g = new THREE.Group();
  const mat = new THREE.MeshStandardMaterial({ color: 0x7a1e28, roughness: 0.5 });
  const body = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.06, 2.4, 10), mat);
  body.rotation.z = Math.PI / 2; body.castShadow = true; g.add(body);
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.3, 12, 10), mat); head.position.x = 1.25; g.add(head);
  const eyeM = new THREE.MeshStandardMaterial({ color: 0xffd23c, emissive: 0x553300 });
  for (const s of [-1, 1]) { const e = new THREE.Mesh(new THREE.SphereGeometry(0.17, 10, 8), eyeM); e.position.set(1.35, 0.1, 0.18 * s); g.add(e); }
  const wingM = new THREE.MeshStandardMaterial({ color: 0xffe0e0, transparent: true, opacity: 0.45, side: THREE.DoubleSide });
  g.wings = [];
  for (const x of [0.55, 0.15]) for (const s of [-1, 1]) {
    const pivot = new THREE.Group(); pivot.position.set(x, 0.1, 0);
    const w = new THREE.Mesh(new THREE.PlaneGeometry(0.35, 1.5), wingM); w.rotation.x = -Math.PI / 2; w.position.z = 0.8 * s;
    pivot.add(w); pivot.side = s; g.add(pivot); g.wings.push(pivot);
  }
  return g;
}
function makeFood() {
  const g = new THREE.Group();
  const f = new THREE.Mesh(new THREE.SphereGeometry(0.45, 18, 14), new THREE.MeshStandardMaterial({ color: 0xf07a28, roughness: 0.4 }));
  f.position.y = 0.45; f.castShadow = true; g.add(f);
  const stem = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 0.25), new THREE.MeshStandardMaterial({ color: 0x4f8a2a }));
  stem.position.y = 0.95; g.add(stem);
  return g;
}

const fly = makeFly(0x7a5028, 0xd22820, 0.9); scene.add(fly);
const shadow = new THREE.Mesh(new THREE.CircleGeometry(0.45, 20), new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.35 }));
shadow.rotation.x = -Math.PI / 2; shadow.position.y = 0.02; scene.add(shadow);
const tether = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
  new THREE.LineDashedMaterial({ color: 0xffffff, dashSize: 0.2, gapSize: 0.2, transparent: true, opacity: 0.5 }));
scene.add(tether);
const ring = new THREE.Mesh(new THREE.TorusGeometry(0.9, 0.05, 8, 32), new THREE.MeshBasicMaterial({ color: 0xffe65a }));
ring.visible = false; scene.add(ring);
const entities = new Map();
let flyTarget = new THREE.Vector3(), flyYaw = 0;

function syncEntities(list) {
  const seen = new Set();
  for (const e of list) {
    seen.add(e.id);
    let o = entities.get(e.id);
    if (!o) {
      o = e.kind === "food" ? makeFood() : e.kind === "female" ? makeFly(0xb0825a, 0xc83c3c, 1.05) : makePredator();
      o.kind = e.kind; scene.add(o); entities.set(e.id, o);
      o.position.copy(toScene(e.pos));
    }
    o.target = toScene(e.pos); o.yaw = yawOf(e.heading); o.resting = e.resting;
    if (e.kind === "food") o.scale.setScalar(0.4 + 0.6 * Math.max(e.amount, 0));
  }
  for (const [id, o] of entities) if (!seen.has(id)) { scene.remove(o); entities.delete(id); }
}

// ------------------------------------------------------------------ 3D brain
const bcanvas = $("brain");
const brenderer = new THREE.WebGLRenderer({ canvas: bcanvas, antialias: true });
brenderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const bscene = new THREE.Scene();
const bcam = new THREE.PerspectiveCamera(40, 1, 1, 2000);
bcam.position.set(0, 0, 130);
const borbit = new OrbitControls(bcam, bcanvas);
borbit.enableDamping = true;
let B = null, bPoints = null, bColors = null, strokeGroup = new THREE.Group(), strokeR = 8;
bscene.add(strokeGroup);

async function loadBrain() {
  B = await (await fetch("/api/brain")).json();
  const n = B.point_neuron.length;
  const pos = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {             // dataset: x = fly's left, y = ventral, z = posterior
    pos[3 * i] = -B.points[3 * i]; pos[3 * i + 1] = -B.points[3 * i + 1]; pos[3 * i + 2] = B.points[3 * i + 2];
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  bColors = new Float32Array(n * 3);
  geo.setAttribute("color", new THREE.BufferAttribute(bColors, 3));
  bPoints = new THREE.Points(geo, new THREE.PointsMaterial({ size: 1.6, vertexColors: true }));
  bscene.add(bPoints);
  geo.computeBoundingBox();
  const bb = geo.boundingBox, c = bb.getCenter(new THREE.Vector3()), sz = bb.getSize(new THREE.Vector3());
  borbit.target.copy(c);
  bcam.position.set(c.x, c.y, c.z + Math.max(sz.x, sz.y) * 1.25);
  $("brainlegend").innerHTML = LEGEND.map(([t, r]) => `<span><b style="background:rgb(${ROLE_COLORS[r]})"></b>${t}</span>`).join("")
    + `<span><b style="background:#c21a1a"></b>removed</span>`;
  $("credits").textContent = `Real wiring: male-cns v1.0 (neuPrint). ${B.n.toLocaleString()} neurons, `
    + `${(B.synapses / 1e6).toFixed(2)}M synapses. Design choices: sensor gains, motor/flight map, world.`;
}

function updateBrainColors(rates, alive) {
  if (!bPoints) return;
  const pn = B.point_neuron;
  for (let i = 0; i < pn.length; i++) {
    const k = pn[i];
    if (!((alive[k >> 3] >> (7 - (k & 7))) & 1)) { bColors[3 * i] = 0.55; bColors[3 * i + 1] = 0.05; bColors[3 * i + 2] = 0.05; continue; }
    const c = ROLE_COLORS[B.roles[k]] || [180, 180, 180];
    const a = 0.16 + 0.84 * Math.min(rates[k] / 2 / 25, 1);
    bColors[3 * i] = c[0] / 255 * a; bColors[3 * i + 1] = c[1] / 255 * a; bColors[3 * i + 2] = c[2] / 255 * a;
  }
  bPoints.geometry.attributes.color.needsUpdate = true;
}

// click the brain (without dragging) = stroke at that point
let bdown = null;
bcanvas.addEventListener("pointerdown", (e) => (bdown = [e.clientX, e.clientY]));
bcanvas.addEventListener("pointerup", (e) => {
  if (!bdown || Math.hypot(e.clientX - bdown[0], e.clientY - bdown[1]) > 4 || !bPoints) return;
  const r = bcanvas.getBoundingClientRect();
  const ray = new THREE.Raycaster(); ray.params.Points.threshold = 2.5;
  ray.setFromCamera(new THREE.Vector2((e.clientX - r.left) / r.width * 2 - 1, -(e.clientY - r.top) / r.height * 2 + 1), bcam);
  const hit = ray.intersectObject(bPoints)[0];
  if (!hit) return;
  const p = hit.point;
  api("stroke", { p: [-p.x, -p.y, p.z], r: strokeR });
});
function drawStrokes(list) {
  if (strokeGroup.children.length === list.length) return;
  strokeGroup.clear();
  for (const s of list) {
    const m = new THREE.Mesh(new THREE.SphereGeometry(s.r, 14, 10), new THREE.MeshBasicMaterial({ color: 0xff4040, wireframe: true, transparent: true, opacity: 0.5 }));
    m.position.set(-s.p[0], -s.p[1], s.p[2]); strokeGroup.add(m);
  }
}

// ------------------------------------------------------------------ panel UI
function renderCircuits(st) {
  const el = $("circuits");
  if (!el.children.length) {
    el.innerHTML = st.circuits.map((c) => `<div class="circ" data-key="${c.key}"><span class="box" style="border-color:${BEHAV[c.behav] || "#aaa"}"></span>
      <span>${c.label} <span style="color:var(--dim)">[${c.n}]</span></span><span class="bar"><i style="background:${BEHAV[c.behav] || "#aaa"}"></i></span><span class="hz"></span></div>`).join("");
    el.onclick = (e) => { const d = e.target.closest(".circ"); if (d) api("circuit", { key: d.dataset.key }); };
  }
  st.circuits.forEach((c, i) => {
    const d = el.children[i];
    d.classList.toggle("dead", c.removed);
    d.querySelector(".bar > i").style.width = c.removed ? "0" : `${Math.min(c.hz / 30, 1) * 100}%`;
    d.querySelector(".hz").textContent = c.removed ? "REMOVED" : `${c.hz.toFixed(1)} Hz`;
  });
}
const pct = (g) => `${Math.round(g * 100)}%`;
const SYN = [1, 0.7, 0.4], INH = [1, 0.5, 0.2], SIZES = [5, 8, 12];
let injuryKey = "";
function renderInjury(st) {
  const j = st.injury, neck = j.neck.join("");
  const key = JSON.stringify([neck, j.syn, j.inh, j.decline, strokeR, (j.seizure * 100).toFixed(0), j.seizure_on, j.removed]);
  if (key === injuryKey) return;
  injuryKey = key;
  const btn = (id, label, on) => `<button data-id="${id}" class="${on ? "on" : ""}">${label}</button>`;
  $("injury").innerHTML = [
    btn("neckLR", "Neck cut: both", neck === "LR"), btn("neckL", "Neck: left half", neck === "L"), btn("neckR", "Neck: right half", neck === "R"),
    btn("lose10", "Lose 10% cells", false), btn("lose30", "Lose 30% cells", false), btn("decline", `Slow decline: ${j.decline ? "ON" : "off"}`, j.decline),
    btn("syn", `All synapses ${pct(j.syn)}`, j.syn < 1), btn("inh", `Brakes ${pct(j.inh)}`, j.inh < 1), btn("size", `Stroke size ${strokeR * 8 * 2} µm`, false),
    `<div class="note">Neck cut = brain-to-body cells severed (the fly can't fly or walk; eating still works, it's in the head). `
    + `Brakes = inhibitory synapses (weak brakes → seizures). Click the brain above = stroke there. `
    + `Seizure meter: ${(j.seizure * 100).toFixed(1)}% of cells > 80 Hz${j.seizure_on ? " — SEIZURE" : ""}.</div>`,
  ].join("");
  const rem = $("removed");
  rem.textContent = `${j.removed.toLocaleString()} of ${B ? B.n.toLocaleString() : "?"} neurons removed` + (neck ? ` · neck cut ${neck}` : "");
  rem.classList.toggle("hurt", j.removed > 0 || !!neck);
}
$("injury").onclick = (e) => {
  const id = e.target.dataset.id; if (!id || !S) return;
  const j = S.injury, neck = j.neck.join("");
  if (id.startsWith("neck")) { const want = id.slice(4); api("neck", { sides: neck === want ? "" : want }); }
  else if (id === "lose10") api("lose", { frac: 0.1 });
  else if (id === "lose30") api("lose", { frac: 0.3 });
  else if (id === "decline") api("decline");
  else if (id === "syn") api("gains", { syn: SYN[(SYN.indexOf(j.syn) + 1) % 3] ?? 1 });
  else if (id === "inh") api("gains", { inh: INH[(INH.indexOf(j.inh) + 1) % 3] ?? 1 });
  else if (id === "size") strokeR = SIZES[(SIZES.indexOf(strokeR) + 1) % 3];
};
$("heal").onclick = () => api("heal");
$("typebox").addEventListener("keydown", (e) => {
  e.stopPropagation();
  if (e.key === "Enter" && e.target.value.trim()) { api("type", { name: e.target.value.trim() }); e.target.value = ""; }
});
$("levels").innerHTML = LEVELS.map(([k, name, col]) => `<div class="lv"><span>${name}</span><span class="bar"><i id="lv-${k}" style="background:${col}"></i></span></div>`).join("");

function setTool(t) { tool = t; document.querySelectorAll(".tool").forEach((b) => b.classList.toggle("on", b.dataset.tool === t)); }
document.querySelectorAll(".tool").forEach((b) => (b.onclick = () => setTool(b.dataset.tool)));
function toggleCam() { chase = !chase; orbit.enabled = !chase; $("cam").textContent = chase ? "V: chase camera" : "V: free camera"; if (!chase) orbit.target.copy(fly.position); }
$("cam").onclick = toggleCam;
$("pause").onclick = () => api("pause");
$("fast").onclick = () => api("fast");
$("cleararena").onclick = () => api("clear_arena");
addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") return;
  const k = e.key.toLowerCase();
  if ("1234".includes(k) && k) setTool(["food", "female", "predator", "erase"][+k - 1]);
  else if (k === "v") toggleCam();
  else if (k === " ") { e.preventDefault(); api("pause"); }
  else if (k === "f") api("fast");
  else if (k === "r") api("clear_arena");
  else if (k === "c") api("heal");
  else if (k === "t") { e.preventDefault(); $("typebox").focus(); }
});

// place things by clicking the ground
let down = null;
canvas.addEventListener("pointerdown", (e) => (down = [e.clientX, e.clientY]));
canvas.addEventListener("contextmenu", (e) => e.preventDefault());
canvas.addEventListener("pointerup", (e) => {
  if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) return;
  const r = canvas.getBoundingClientRect();
  const ray = new THREE.Raycaster();
  ray.setFromCamera(new THREE.Vector2((e.clientX - r.left) / r.width * 2 - 1, -(e.clientY - r.top) / r.height * 2 + 1), camera);
  const hit = ray.intersectObject(ground)[0];
  if (!hit) return;
  const x = hit.point.x + W[0] / 2, y = hit.point.z + W[1] / 2;
  if (e.button === 2 || tool === "erase") api("remove", { x, y });
  else api("spawn", { kind: tool, x, y, z: 7 });
});

// ------------------------------------------------------------------ state polling
const b64 = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));
async function poll() {
  try {
    S = await (await fetch("/api/state")).json();
    W = S.world;
    const f = S.fly;
    flyTarget = toScene(f.pos); flyYaw = yawOf(f.heading);
    syncEntities(S.entities);
    const lab = $("label"); lab.textContent = S.label + (S.paused ? " (paused)" : "") + (S.fast ? " ×4" : "");
    lab.style.color = /SEIZ|PARAL|FALL|CRASH/.test(S.label) ? "#ff5a5a" : /ESCAP|ALARM/.test(S.label) ? "#ffd84f"
      : /FEED/.test(S.label) ? "#ffaa46" : /CHAS|SING/.test(S.label) ? "#ff6ebe" : /SMELL/.test(S.label) ? "#be96ff" : "#6fe68a";
    $("stats").textContent = `t ${S.t.toFixed(0)}s · escapes ${S.stats.jumps} · caught ${S.stats.caught} · eaten ${S.stats.eaten} · `
      + `height ${f.pos[2].toFixed(1)} · wing power ${(f.lift * 100).toFixed(0)}%`;
    const ev = $("event"); ev.style.display = S.event ? "block" : "none"; ev.textContent = S.event;
    ev.style.color = /CAUGHT|CRASH|STROKE|DESTROY/.test(S.event) ? "#ff6a6a" : "#ffd84f";
    for (const [k] of LEVELS) $(`lv-${k}`).style.width = `${Math.min(S.levels[k] ?? 0, 1) * 100}%`;
    renderCircuits(S); renderInjury(S);
    $("typemsg").textContent = S.msg || (S.injury.types.length ? "removed types: " + S.injury.types.join(", ") : "");
    updateBrainColors(b64(S.rates), b64(S.alive));
    drawStrokes(S.injury.strokes);
  } catch (err) { $("label").textContent = "lost connection to the simulation — is the Python server running?"; }
  setTimeout(poll, 45);
}

// ------------------------------------------------------------------ animation
const clock = new THREE.Clock();
function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
  const bw = bcanvas.clientWidth, bh = bcanvas.clientHeight;
  brenderer.setSize(bw, bh, false); bcam.aspect = bw / bh; bcam.updateProjectionMatrix();
}
addEventListener("resize", resize);
function animate() {
  const dt = Math.min(clock.getDelta(), 0.1), t = clock.elapsedTime;
  fly.position.lerp(flyTarget, 1 - Math.exp(-dt * 12));
  const dy = ((flyYaw - fly.rotation.y + Math.PI * 3) % (Math.PI * 2)) - Math.PI;
  fly.rotation.y += dy * (1 - Math.exp(-dt * 10));
  const f = S?.fly;
  const flying = f?.flying && f.pos[2] > 0.05;
  fly.wings.forEach((w) => {
    if (f?.singing && w.side === 1) { w.rotation.set(0, -1.3, 0); return; }
    w.rotation.set(flying ? Math.sin(t * 60) * 0.9 * w.side : 0.05 * w.side, flying ? 0 : 0.35 * w.side, 0);
  });
  fly.position.y = flyTarget.y + (flying ? 0.35 : 0.22);
  fly.prob.visible = !!f?.feeding;
  shadow.position.set(fly.position.x, 0.02, fly.position.z);
  shadow.scale.setScalar(1 / (1 + fly.position.y * 0.25));
  tether.geometry.setFromPoints([fly.position, new THREE.Vector3(fly.position.x, 0, fly.position.z)]); tether.computeLineDistances();
  tether.visible = flying;
  ring.visible = !!f?.escaping; ring.position.copy(fly.position); ring.rotation.x = Math.PI / 2;
  for (const o of entities.values()) {
    o.position.lerp(o.target, 1 - Math.exp(-dt * 10));
    o.rotation.y = o.yaw;
    if (o.kind === "female") o.position.y = o.target.y + 0.25;
    if (o.kind === "predator") o.wings.forEach((w, i) => (w.rotation.x = o.resting ? 0.1 * w.side : Math.sin(t * 40 + i) * 0.5 * w.side));
  }
  if (chase) {
    const back = new THREE.Vector3(-Math.cos(fly.rotation.y), 0, Math.sin(fly.rotation.y));
    const want = fly.position.clone().addScaledVector(back, CAM_BACK).add(new THREE.Vector3(0, CAM_UP, 0));
    want.y = Math.min(Math.max(want.y, 0.8), BOX_H - 0.6);
    want.x = Math.min(Math.max(want.x, -19.5), 19.5); want.z = Math.min(Math.max(want.z, -19.5), 19.5);
    camera.position.lerp(want, 1 - Math.exp(-dt * 3));
    camera.lookAt(fly.position.x, fly.position.y + 0.3, fly.position.z);
  } else orbit.update();
  renderer.render(scene, camera);
  borbit.update(); brenderer.render(bscene, bcam);
  requestAnimationFrame(animate);
}

resize();
loadBrain().then(() => { poll(); animate(); });
