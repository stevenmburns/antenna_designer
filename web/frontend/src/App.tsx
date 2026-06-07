import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type RefObject,
} from "react";

// ===========================================================================
// API types — mirror the FastAPI shapes in web/server.py.
// ===========================================================================

type ComplexVal = { re: number; im: number };

type ParamSpec =
  | {
      key: string;
      default: number;
      min: number;
      max: number;
      step: number;
      is_int: boolean;
      is_complex?: undefined;
    }
  | {
      key: string;
      default: ComplexVal;
      is_complex: true;
    };

type BuilderSchema = {
  name: string;
  variants: string[];
  params: ParamSpec[];
};

type BuildersResp = { builders: BuilderSchema[] };

type SolveRequest = {
  builder: string;
  variant: string;
  params: Record<string, number | ComplexVal>;
  engine: string;
  pysim_basis?: string;
  ground?: string | null;
  far_field?: boolean;
};

type WireGeom = {
  p0: [number, number, number];
  p1: [number, number, number];
  n_seg: number;
  feed_voltage: ComplexVal | null;
};

type WireCurrent = {
  knot_positions: [number, number, number][];
  knot_currents_re: number[];
  knot_currents_im: number[];
};

type FarField = {
  thetas: number[];
  phis: number[];
  rings: number[][];
  max_gain: number;
  min_gain: number;
};

type SolveResponse = {
  builder: string;
  variant: string;
  engine: string;
  freq_mhz: number;
  z_per_feed: ComplexVal[];
  wires: WireGeom[];
  currents: WireCurrent[];
  far_field: FarField | null;
};

type SweepRequest = {
  builder: string;
  variant: string;
  params: Record<string, number | ComplexVal>;
  engine: string;
  pysim_basis?: string;
  ground?: string | null;
  band_start_mhz: number;
  band_stop_mhz: number;
  n_points: number;
};

type SweepResponse = {
  freqs_mhz: number[];
  z_per_feed: ComplexVal[][]; // [freq_idx][feed_idx]
};

type ConvergeRequest = {
  builder: string;
  variant: string;
  params: Record<string, number | ComplexVal>;
  engine: string;
  pysim_basis?: string;
  ground?: string | null;
  scales?: number[];
};

type ConvergeResponse = {
  scales: number[];
  n_segs_total: number[];
  z_per_feed: ComplexVal[][]; // [scale_idx][feed_idx]
};

// ===========================================================================
// Engine + ground choices.
// ===========================================================================

type EngineChoice = {
  id: string;
  letter: string;
  sub: string;
  engine: string;
  pysim_basis?: string;
};

const ENGINE_CHOICES: EngineChoice[] = [
  { id: "pynec", letter: "PyNEC", sub: "nec2", engine: "pynec" },
  { id: "pysim-tri", letter: "Tri", sub: "pysim", engine: "pysim", pysim_basis: "triangular" },
  { id: "pysim-sin", letter: "Sin", sub: "pysim", engine: "pysim", pysim_basis: "sinusoidal" },
  { id: "pysim-bsp", letter: "Bsp", sub: "pysim", engine: "pysim", pysim_basis: "bspline" },
];

type SlotId = "A" | "B" | "C";
type Slot = { id: SlotId; engineId: string; enabled: boolean };

// Letter-shape style used to distinguish each slot's markers/lines across
// every overlay chart. Slot A is always solid + filled; B + C use line
// dashes and marker glyphs.
const SLOT_STYLE: Record<SlotId, { dash: number[]; marker: "filled" | "hollow" | "x" }> = {
  A: { dash: [], marker: "filled" },
  B: { dash: [5, 4], marker: "hollow" },
  C: { dash: [2, 3], marker: "x" },
};

const GROUND_CHOICES = [
  { id: "free", label: "Free space" },
  { id: "pec", label: "PEC plane" },
  { id: "finite:13,0.005", label: "Finite (ε=13, σ=0.005)" },
];

type ProjMode = "auto" | "xy" | "xz" | "yz";
type Projection = "xy" | "xz" | "yz";
type FFCut = "azimuth" | "elevation";
type View = "wire" | "wire3d" | "smith" | "ff-az" | "ff-el" | "sweep" | "converge";

function swrFromZ(z_re: number, z_im: number, z0: number): number {
  const num_re = z_re - z0;
  const num_im = z_im;
  const den_re = z_re + z0;
  const den_im = z_im;
  const denom = den_re * den_re + den_im * den_im;
  if (denom === 0) return Infinity;
  const gR = (num_re * den_re + num_im * den_im) / denom;
  const gI = (num_im * den_re - num_re * den_im) / denom;
  const g = Math.hypot(gR, gI);
  if (g >= 1) return Infinity;
  return (1 + g) / (1 - g);
}

// ===========================================================================
// Schema-driven sliders.
// ===========================================================================

function fmtVal(v: number, step: number, isInt: boolean): string {
  if (isInt) return v.toFixed(0);
  const decs = Math.max(0, Math.min(4, -Math.floor(Math.log10(step))));
  return v.toFixed(decs);
}

function ParamSliders({
  params,
  values,
  onChange,
}: {
  params: ParamSpec[];
  values: Record<string, number | ComplexVal>;
  onChange: (key: string, val: number | ComplexVal) => void;
}) {
  return (
    <>
      {params.map((p) => {
        if (p.is_complex) {
          const v = (values[p.key] as ComplexVal | undefined) ?? p.default;
          return (
            <div className="field" key={p.key}>
              <label>
                <span>{p.key}</span>
                <span className="val">
                  {v.re.toFixed(2)} {v.im >= 0 ? "+" : "-"} {Math.abs(v.im).toFixed(2)}j
                </span>
              </label>
              <div className="complex-input">
                <input
                  type="number"
                  value={v.re}
                  step={0.01}
                  onChange={(e) =>
                    onChange(p.key, { re: parseFloat(e.target.value), im: v.im })
                  }
                />
                <input
                  type="number"
                  value={v.im}
                  step={0.01}
                  onChange={(e) =>
                    onChange(p.key, { re: v.re, im: parseFloat(e.target.value) })
                  }
                />
              </div>
            </div>
          );
        }
        const cur = (values[p.key] as number | undefined) ?? p.default;
        return (
          <div className="field" key={p.key}>
            <label>
              <span>{p.key}</span>
              <span className="val">{fmtVal(cur, p.step, p.is_int)}</span>
            </label>
            <input
              type="range"
              min={p.min}
              max={p.max}
              step={p.step}
              value={cur}
              onChange={(e) => onChange(p.key, parseFloat(e.target.value))}
            />
          </div>
        );
      })}
    </>
  );
}

// ===========================================================================
// ResizeObserver hook — drawing components fill their parent.
// ===========================================================================

function useFillSize<T extends HTMLElement>(): [RefObject<T>, number] {
  const ref = useRef<T>(null);
  const [size, setSize] = useState(480);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const cr = entries[0].contentRect;
      const s = Math.max(160, Math.min(cr.width, cr.height));
      setSize(s);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, size];
}

// ===========================================================================
// 2D wire renderer + current overlay.
// ===========================================================================

function project(p: [number, number, number], proj: Projection): [number, number] {
  switch (proj) {
    case "xy": return [p[0], p[1]];
    case "xz": return [p[0], p[2]];
    case "yz": return [p[1], p[2]];
  }
}

function pickProjection(wires: WireGeom[]): Projection {
  const mins = [Infinity, Infinity, Infinity];
  const maxs = [-Infinity, -Infinity, -Infinity];
  for (const w of wires) for (const p of [w.p0, w.p1]) for (let i = 0; i < 3; i++) {
    if (p[i] < mins[i]) mins[i] = p[i];
    if (p[i] > maxs[i]) maxs[i] = p[i];
  }
  const sp = [maxs[0] - mins[0], maxs[1] - mins[1], maxs[2] - mins[2]];
  const xy = sp[0] + sp[1], xz = sp[0] + sp[2], yz = sp[1] + sp[2];
  if (xz >= xy && xz >= yz) return "xz";
  if (yz >= xy) return "yz";
  return "xy";
}

function magToColor(t: number): string {
  const v = Math.max(0, Math.min(1, t));
  const r = Math.round(255 * (0.267 + v * (0.993 - 0.267)));
  const g = Math.round(255 * (0.005 + v * (0.906 - 0.005)));
  const b = Math.round(255 * (0.329 + v * (0.144 - 0.329)));
  return `rgb(${r},${g},${b})`;
}

function WireCanvas({
  wires,
  currents,
  size = 480,
  projMode = "auto",
  showCurrents = true,
  thumb = false,
}: {
  wires: WireGeom[];
  currents?: WireCurrent[] | null;
  size?: number;
  projMode?: ProjMode;
  showCurrents?: boolean;
  thumb?: boolean;
}) {
  const cur = showCurrents ? currents : null;
  if (wires.length === 0)
    return <svg width={size} height={size} className={thumb ? "thumb-canvas" : undefined} />;
  const proj: Projection = projMode === "auto" ? pickProjection(wires) : projMode;
  const proj2 = wires.map((w) => [project(w.p0, proj), project(w.p1, proj)] as const);
  const xs = proj2.flatMap((pq) => [pq[0][0], pq[1][0]]);
  const ys = proj2.flatMap((pq) => [pq[0][1], pq[1][1]]);
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  const ymin = Math.min(...ys), ymax = Math.max(...ys);
  const pad = thumb ? 0.06 : 0.08;
  const dx = Math.max(xmax - xmin, 1e-6);
  const dy = Math.max(ymax - ymin, 1e-6);
  const scale = Math.min((size * (1 - 2 * pad)) / dx, (size * (1 - 2 * pad)) / dy);
  const ox = (size - scale * dx) / 2 - scale * xmin;
  const oy = (size - scale * dy) / 2 - scale * ymin;
  const tx = (x: number) => ox + scale * x;
  const ty = (y: number) => size - (oy + scale * y);

  let peakI = 0;
  if (cur) for (const w of cur) for (let i = 0; i < w.knot_currents_re.length; i++) {
    const m = Math.hypot(w.knot_currents_re[i], w.knot_currents_im[i]);
    if (m > peakI) peakI = m;
  }

  return (
    <svg width={size} height={size} className={thumb ? "thumb-canvas" : undefined}>
      <rect x={0} y={0} width={size} height={size} fill="#0d1015" />
      {!thumb && (
        <text
          x={10}
          y={size - 8}
          fill="var(--muted)"
          fontSize={11}
          fontFamily="ui-monospace, monospace"
        >
          {`${proj} · ${dx.toFixed(2)} × ${dy.toFixed(2)} m`}
        </text>
      )}
      {wires.map((w, i) => {
        const [[a, b], [c, d]] = proj2[i];
        const isFeed = w.feed_voltage !== null;
        return (
          <line
            key={`w-${i}`}
            x1={tx(a)} y1={ty(b)} x2={tx(c)} y2={ty(d)}
            stroke={isFeed ? "var(--feed)" : "var(--wire)"}
            strokeWidth={isFeed ? (thumb ? 1.5 : 2.5) : thumb ? 0.8 : 1.0}
            opacity={cur && peakI > 0 ? 0.3 : 1}
          />
        );
      })}
      {cur && peakI > 0 && cur.flatMap((w, wi) => {
        const knots = w.knot_positions;
        const re = w.knot_currents_re, im = w.knot_currents_im;
        const segs = [];
        for (let k = 0; k < knots.length - 1; k++) {
          const [a, b] = project(knots[k], proj);
          const [c, d] = project(knots[k + 1], proj);
          const m = 0.5 * (Math.hypot(re[k], im[k]) + Math.hypot(re[k + 1], im[k + 1]));
          segs.push(
            <line
              key={`c-${wi}-${k}`}
              x1={tx(a)} y1={ty(b)} x2={tx(c)} y2={ty(d)}
              stroke={magToColor(m / peakI)}
              strokeWidth={thumb ? 1.5 : 3.0}
              strokeLinecap="round"
            />,
          );
        }
        return segs;
      })}
    </svg>
  );
}

// ===========================================================================
// 3D wire renderer — orbit + zoom + per-knot current overlay.
// ===========================================================================

function rotate3(
  p: [number, number, number],
  yawRad: number,
  pitchRad: number,
): [number, number, number] {
  const cy = Math.cos(yawRad), sy = Math.sin(yawRad);
  const cp = Math.cos(pitchRad), sp = Math.sin(pitchRad);
  const x1 = cy * p[0] - sy * p[1];
  const y1 = sy * p[0] + cy * p[1];
  const z1 = p[2];
  const y2 = cp * y1 - sp * z1;
  const z2 = sp * y1 + cp * z1;
  return [x1, y2, z2];
}

function Wire3DCanvas({
  wires,
  currents,
  size = 480,
  yawDeg,
  pitchDeg,
  zoom,
  showCurrents = true,
  thumb = false,
}: {
  wires: WireGeom[];
  currents?: WireCurrent[] | null;
  size?: number;
  yawDeg: number;
  pitchDeg: number;
  zoom: number;
  showCurrents?: boolean;
  thumb?: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(size * dpr);
    canvas.height = Math.floor(size * dpr);
    canvas.style.width = `${size}px`;
    canvas.style.height = `${size}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    ctx.fillStyle = "#0d1015";
    ctx.fillRect(0, 0, size, size);

    if (wires.length === 0) return;
    const yawRad = (yawDeg * Math.PI) / 180;
    const pitchRad = (pitchDeg * Math.PI) / 180;

    // Project p → (sx, sy, depth). After rotation, x = right, z = up, y = depth.
    type SP = { sx: number; sy: number; depth: number };
    const proj = (p: [number, number, number]): SP => {
      const [x, y, z] = rotate3(p, yawRad, pitchRad);
      return { sx: x, sy: z, depth: y };
    };

    const rotPairs = wires.map((w) => [proj(w.p0), proj(w.p1)] as const);
    let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
    for (const [a, b] of rotPairs) {
      xmin = Math.min(xmin, a.sx, b.sx);
      xmax = Math.max(xmax, a.sx, b.sx);
      ymin = Math.min(ymin, a.sy, b.sy);
      ymax = Math.max(ymax, a.sy, b.sy);
    }
    const dx = Math.max(xmax - xmin, 1e-6);
    const dy = Math.max(ymax - ymin, 1e-6);
    const pad = thumb ? 0.08 : 0.1;
    const fit = Math.min((size * (1 - 2 * pad)) / dx, (size * (1 - 2 * pad)) / dy);
    const scale = fit * zoom;
    const ox = (size - scale * dx) / 2 - scale * xmin;
    const oy = (size - scale * dy) / 2 - scale * ymin;
    const tx = (x: number) => ox + scale * x;
    const ty = (y: number) => size - (oy + scale * y);

    // Painter's algorithm — back wires first (larger depth → further away
    // along view ray). For a tied chain of wires this gives correct stacking
    // without a full z-buffer.
    type SegItem = {
      a: SP; b: SP;
      isFeed: boolean;
      color: string;
      width: number;
      midDepth: number;
    };

    const items: SegItem[] = [];
    wires.forEach((w, i) => {
      const [a, b] = rotPairs[i];
      const isFeed = w.feed_voltage !== null;
      items.push({
        a, b, isFeed,
        color: isFeed ? "var(--feed)" : "var(--wire)",
        width: isFeed ? (thumb ? 1.5 : 2.5) : thumb ? 0.8 : 1.0,
        midDepth: (a.depth + b.depth) / 2,
      });
    });

    let peakI = 0;
    if (showCurrents && currents) {
      for (const w of currents) for (let i = 0; i < w.knot_currents_re.length; i++) {
        const m = Math.hypot(w.knot_currents_re[i], w.knot_currents_im[i]);
        if (m > peakI) peakI = m;
      }
    }

    const curSegs: SegItem[] = [];
    if (showCurrents && currents && peakI > 0) {
      currents.forEach((w) => {
        const knots = w.knot_positions;
        const re = w.knot_currents_re, im = w.knot_currents_im;
        for (let k = 0; k < knots.length - 1; k++) {
          const a = proj(knots[k]);
          const b = proj(knots[k + 1]);
          const m = 0.5 * (Math.hypot(re[k], im[k]) + Math.hypot(re[k + 1], im[k + 1]));
          curSegs.push({
            a, b,
            isFeed: false,
            color: magToColor(m / peakI),
            width: thumb ? 1.5 : 3.0,
            midDepth: (a.depth + b.depth) / 2,
          });
        }
      });
    }

    // Sort all segments back-to-front. Base wires drawn dim if currents on.
    const all = [...items, ...curSegs];
    all.sort((p, q) => q.midDepth - p.midDepth);
    for (const it of all) {
      ctx.strokeStyle = it.color;
      ctx.globalAlpha = it.isFeed ? 1 : (curSegs.length > 0 && items.includes(it) ? 0.3 : 1);
      ctx.lineWidth = it.width;
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(tx(it.a.sx), ty(it.a.sy));
      ctx.lineTo(tx(it.b.sx), ty(it.b.sy));
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    // Compass: small axis triple at corner.
    if (!thumb) {
      ctx.font = "10px ui-monospace, monospace";
      const ax = 30, ay = size - 30, len = 16;
      const axes: [[number, number, number], string][] = [
        [[1, 0, 0], "x"], [[0, 1, 0], "y"], [[0, 0, 1], "z"],
      ];
      for (const [v, lbl] of axes) {
        const pr = rotate3(v, yawRad, pitchRad);
        const dx = pr[0] * len, dy = pr[2] * len;
        ctx.strokeStyle = "#4a5160";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.lineTo(ax + dx, ay - dy);
        ctx.stroke();
        ctx.fillStyle = "#cfd6e3";
        ctx.fillText(lbl, ax + dx + 2, ay - dy + 4);
      }
      ctx.fillStyle = "var(--muted)";
      ctx.fillText(`yaw ${yawDeg.toFixed(0)}° · pitch ${pitchDeg.toFixed(0)}° · ×${zoom.toFixed(2)}`, 10, 14);
    }
  }, [wires, currents, size, yawDeg, pitchDeg, zoom, showCurrents, thumb]);

  return <canvas ref={canvasRef} className={thumb ? "thumb-canvas" : undefined} />;
}

// ===========================================================================
// Smith chart.
// ===========================================================================

function feedColor(i: number, n: number, alpha = 0.95): string {
  const hue = n <= 1 ? 200 : (i * 360) / n;
  return `hsla(${hue}, 80%, 60%, ${alpha})`;
}

function SmithChart({
  z_per_feed,
  z0,
  size = 480,
  thumb = false,
  sweep,
  compareSlots,
}: {
  z_per_feed: ComplexVal[];
  z0: number;
  size?: number;
  thumb?: boolean;
  sweep?: SweepResponse | null;
  compareSlots?: { slotId: SlotId; z_per_feed: ComplexVal[] }[];
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(size * dpr);
    canvas.height = Math.floor(size * dpr);
    canvas.style.width = `${size}px`;
    canvas.style.height = `${size}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const cx = size / 2, cy = size / 2;
    const R = size / 2 - (thumb ? 4 : 12);

    ctx.fillStyle = "#0d1015";
    ctx.fillRect(0, 0, size, size);

    ctx.strokeStyle = "#2a313d";
    ctx.lineWidth = thumb ? 0.4 : 0.6;
    for (const rn of [0.2, 0.5, 1, 2, 5]) {
      ctx.beginPath();
      ctx.arc(cx + (rn / (rn + 1)) * R, cy, (1 / (rn + 1)) * R, 0, 2 * Math.PI);
      ctx.stroke();
    }
    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, R, 0, 2 * Math.PI);
    ctx.clip();
    for (const xn of [0.2, 0.5, 1, 2, 5]) {
      const rad = (1 / xn) * R;
      ctx.beginPath();
      ctx.arc(cx + R, cy - (1 / xn) * R, rad, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(cx + R, cy + (1 / xn) * R, rad, 0, 2 * Math.PI);
      ctx.stroke();
    }
    ctx.restore();
    ctx.strokeStyle = "#3a4150";
    ctx.lineWidth = thumb ? 0.6 : 0.8;
    ctx.beginPath();
    ctx.moveTo(cx - R, cy); ctx.lineTo(cx + R, cy);
    ctx.stroke();
    ctx.lineWidth = thumb ? 1.0 : 1.5;
    ctx.beginPath();
    ctx.arc(cx, cy, R, 0, 2 * Math.PI);
    ctx.stroke();
    if (!thumb) {
      ctx.fillStyle = "#4a5160";
      ctx.font = "10px ui-monospace, monospace";
      ctx.fillText(`Z₀ = ${z0}Ω`, 8, 14);
      ctx.fillText("+jX", cx + R - 26, cy - R + 14);
      ctx.fillText("−jX", cx + R - 26, cy + R - 4);
    }
    // Optional sweep trail — one polyline per feed across the band.
    const nFeedsCurrent = z_per_feed.length;
    if (sweep && sweep.z_per_feed.length > 1) {
      const nFeedsSweep = sweep.z_per_feed[0]?.length ?? 0;
      for (let f = 0; f < nFeedsSweep; f++) {
        ctx.strokeStyle = feedColor(f, nFeedsSweep, 0.55);
        ctx.lineWidth = thumb ? 1.2 : 1.8;
        ctx.beginPath();
        let first = true;
        for (let k = 0; k < sweep.freqs_mhz.length; k++) {
          const z = sweep.z_per_feed[k][f];
          if (!z) continue;
          const zn_r = z.re / z0, zn_x = z.im / z0;
          const dR = (zn_r + 1) * (zn_r + 1) + zn_x * zn_x;
          const gR = ((zn_r - 1) * (zn_r + 1) + zn_x * zn_x) / dR;
          const gI = (2 * zn_x) / dR;
          const px = cx + gR * R, py = cy - gI * R;
          if (first) { ctx.moveTo(px, py); first = false; }
          else ctx.lineTo(px, py);
        }
        ctx.stroke();
      }
    }

    const n = nFeedsCurrent;
    const drawMarker = (
      z: ComplexVal,
      i: number,
      slotId: SlotId,
      labelSuffix: string,
    ) => {
      const zn_r = z.re / z0, zn_x = z.im / z0;
      const denomR = (zn_r + 1) * (zn_r + 1) + zn_x * zn_x;
      const gR = ((zn_r - 1) * (zn_r + 1) + zn_x * zn_x) / denomR;
      const gI = (2 * zn_x) / denomR;
      const px = cx + gR * R, py = cy - gI * R;
      const r = thumb ? 3 : 6;
      const fc = feedColor(i, n);
      const m = SLOT_STYLE[slotId].marker;
      ctx.strokeStyle = "#0d1015";
      ctx.lineWidth = 1.5;
      if (m === "filled") {
        ctx.fillStyle = fc;
        ctx.beginPath();
        ctx.arc(px, py, r, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();
      } else if (m === "hollow") {
        ctx.fillStyle = "#0d1015";
        ctx.beginPath();
        ctx.arc(px, py, r, 0, 2 * Math.PI);
        ctx.fill();
        ctx.strokeStyle = fc;
        ctx.lineWidth = 2;
        ctx.stroke();
      } else {
        ctx.strokeStyle = fc;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(px - r, py - r); ctx.lineTo(px + r, py + r);
        ctx.moveTo(px - r, py + r); ctx.lineTo(px + r, py - r);
        ctx.stroke();
      }
      if (!thumb) {
        ctx.fillStyle = "#cfd6e3";
        ctx.font = "10px ui-monospace, monospace";
        ctx.fillText(`f${i}${labelSuffix}`, px + 10, py + 4);
      }
    };
    z_per_feed.forEach((z, i) => drawMarker(z, i, "A", ""));
    if (compareSlots) {
      for (const cs of compareSlots) {
        cs.z_per_feed.forEach((z, i) => drawMarker(z, i, cs.slotId, ` ${cs.slotId}`));
      }
    }
  }, [z_per_feed, z0, size, thumb]);
  return <canvas ref={canvasRef} className={thumb ? "thumb-canvas" : undefined} />;
}

// ===========================================================================
// Far-field polar plot — one cut (azimuth or elevation).
// ===========================================================================

function nearestIdx(arr: number[], target: number): number {
  let best = 0, bestD = Infinity;
  for (let i = 0; i < arr.length; i++) {
    const d = Math.abs(arr[i] - target);
    if (d < bestD) { bestD = d; best = i; }
  }
  return best;
}

function FarFieldPlot({
  ff,
  size = 480,
  cut,
  cutAngleDeg,
  thumb = false,
}: {
  ff: FarField;
  size?: number;
  cut: FFCut;
  cutAngleDeg: number;
  thumb?: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(size * dpr);
    canvas.height = Math.floor(size * dpr);
    canvas.style.width = `${size}px`;
    canvas.style.height = `${size}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    ctx.fillStyle = "#0d1015";
    ctx.fillRect(0, 0, size, size);

    const cx = size / 2, cy = size / 2;
    const R = size / 2 - (thumb ? 6 : 18);
    const DBI_TOP = Math.max(0, Math.ceil(ff.max_gain / 3) * 3 + 3);
    const DB_SPAN = 30;
    const dbToFrac = (db: number) => Math.max(0, (db - (DBI_TOP - DB_SPAN)) / DB_SPAN);

    ctx.strokeStyle = "#2a313d";
    ctx.lineWidth = thumb ? 0.4 : 0.6;
    ctx.fillStyle = "#4a5160";
    ctx.font = "9px ui-monospace, monospace";
    for (let d = 0; d <= 5; d++) {
      const rad = (R * (5 - d)) / 5;
      ctx.beginPath();
      ctx.arc(cx, cy, rad, 0, 2 * Math.PI);
      ctx.stroke();
      if (!thumb && d > 0 && d < 5) {
        ctx.fillText(`${(DBI_TOP - d * 6).toFixed(0)} dBi`, cx + 3, cy - rad + 10);
      }
    }
    ctx.strokeStyle = "#252a35";
    for (let a = 0; a < 360; a += 30) {
      const ar = (a * Math.PI) / 180;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + Math.cos(ar) * R, cy - Math.sin(ar) * R);
      ctx.stroke();
    }

    if (cut === "azimuth") {
      const theta0 = 90 - cutAngleDeg;
      const tI = nearestIdx(ff.thetas, theta0);
      ctx.strokeStyle = "hsla(200, 80%, 60%, 0.95)";
      ctx.lineWidth = thumb ? 1.2 : 1.8;
      ctx.beginPath();
      for (let pi = 0; pi < ff.phis.length; pi++) {
        const phi = (ff.phis[pi] * Math.PI) / 180;
        const dbi = ff.rings[tI][pi];
        const rad = dbToFrac(dbi) * R;
        const x = cx + Math.cos(phi) * rad;
        const y = cy - Math.sin(phi) * rad;
        if (pi === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.stroke();
    } else {
      const phiF = nearestIdx(ff.phis, cutAngleDeg);
      const phiB = nearestIdx(ff.phis, (cutAngleDeg + 180) % 360);
      ctx.strokeStyle = "hsla(40, 90%, 60%, 0.95)";
      ctx.lineWidth = thumb ? 1.2 : 1.8;
      ctx.beginPath();
      for (let ti = 0; ti < ff.thetas.length; ti++) {
        const theta = ff.thetas[ti];
        const dbi = ff.rings[ti][phiF];
        const ang = ((90 - theta) * Math.PI) / 180;
        const rad = dbToFrac(dbi) * R;
        const x = cx + Math.cos(ang) * rad;
        const y = cy - Math.sin(ang) * rad;
        if (ti === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      for (let ti = ff.thetas.length - 1; ti >= 0; ti--) {
        const theta = ff.thetas[ti];
        const dbi = ff.rings[ti][phiB];
        const ang = ((90 + theta) * Math.PI) / 180;
        const rad = dbToFrac(dbi) * R;
        ctx.lineTo(cx + Math.cos(ang) * rad, cy - Math.sin(ang) * rad);
      }
      ctx.stroke();
    }

    if (!thumb) {
      ctx.fillStyle = "#cfd6e3";
      ctx.font = "11px ui-monospace, monospace";
      if (cut === "azimuth") {
        ctx.fillText("φ=0°",   cx + R - 30, cy + 14);
        ctx.fillText("φ=90°",  cx + 6,      cy - R + 14);
        ctx.fillText("φ=180°", cx - R + 4,  cy + 14);
        ctx.fillText("φ=270°", cx + 6,      cy + R - 4);
        ctx.fillStyle = "hsla(200, 80%, 60%, 0.95)";
        ctx.fillRect(8, size - 16, 12, 2);
        ctx.fillStyle = "#cfd6e3";
        ctx.fillText(`az cut @ el ${cutAngleDeg.toFixed(0)}°`, 26, size - 11);
      } else {
        ctx.fillText("zenith",  cx + 4, 14);
        ctx.fillText("horizon", cx + R - 50, cy + 14);
        ctx.fillText("nadir",   cx + 4, size - 2);
        ctx.fillStyle = "hsla(40, 90%, 60%, 0.95)";
        ctx.fillRect(8, size - 16, 12, 2);
        ctx.fillStyle = "#cfd6e3";
        ctx.fillText(`el cut @ φ ${cutAngleDeg.toFixed(0)}°`, 26, size - 11);
      }
    }
  }, [ff, size, cut, cutAngleDeg, thumb]);
  return <canvas ref={canvasRef} className={thumb ? "thumb-canvas" : undefined} />;
}

// ===========================================================================
// SWR-vs-freq plot — one line per feed.
// ===========================================================================

function SwrPlot({
  sweep,
  z0,
  measFreqMhz,
  size = 480,
  thumb = false,
}: {
  sweep: SweepResponse | null;
  z0: number;
  measFreqMhz: number;
  size?: number;
  thumb?: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(size * dpr);
    canvas.height = Math.floor(size * dpr);
    canvas.style.width = `${size}px`;
    canvas.style.height = `${size}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    ctx.fillStyle = "#0d1015";
    ctx.fillRect(0, 0, size, size);

    if (!sweep || sweep.freqs_mhz.length < 2) {
      if (!thumb) {
        ctx.fillStyle = "var(--muted)";
        ctx.font = "12px ui-monospace, monospace";
        ctx.fillText("Hit Sweep to compute", 12, size / 2);
      }
      return;
    }

    const padL = thumb ? 4 : 44;
    const padR = thumb ? 4 : 14;
    const padT = thumb ? 4 : 14;
    const padB = thumb ? 4 : 28;
    const W = size - padL - padR;
    const H = size - padT - padB;

    const fs = sweep.freqs_mhz;
    const f0 = fs[0], f1 = fs[fs.length - 1];
    const SWR_MAX = 5;

    // Compute SWR for every point/feed.
    const nFeeds = sweep.z_per_feed[0]?.length ?? 0;
    const swrSeries: number[][] = [];
    for (let f = 0; f < nFeeds; f++) {
      const ser = sweep.z_per_feed.map((zRow) =>
        Math.min(SWR_MAX, swrFromZ(zRow[f].re, zRow[f].im, z0)),
      );
      swrSeries.push(ser);
    }

    const xT = (mhz: number) => padL + ((mhz - f0) / (f1 - f0)) * W;
    const yT = (swr: number) => padT + (1 - (swr - 1) / (SWR_MAX - 1)) * H;

    // Grid + axes.
    ctx.strokeStyle = "#2a313d";
    ctx.lineWidth = 0.6;
    ctx.fillStyle = "#4a5160";
    ctx.font = "10px ui-monospace, monospace";
    for (const s of [1.0, 1.5, 2.0, 3.0, 5.0]) {
      const y = yT(s);
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(padL + W, y);
      ctx.stroke();
      if (!thumb) ctx.fillText(`${s.toFixed(1)}`, 6, y + 3);
    }
    // SWR=2 line emphasised
    ctx.strokeStyle = "#ffd16640";
    ctx.lineWidth = 0.8;
    const y2 = yT(2);
    ctx.beginPath();
    ctx.moveTo(padL, y2);
    ctx.lineTo(padL + W, y2);
    ctx.stroke();

    // Freq ticks.
    if (!thumb) {
      ctx.strokeStyle = "#2a313d";
      ctx.fillStyle = "#4a5160";
      const nticks = 5;
      for (let i = 0; i <= nticks; i++) {
        const mhz = f0 + (i / nticks) * (f1 - f0);
        const x = xT(mhz);
        ctx.beginPath();
        ctx.moveTo(x, padT + H);
        ctx.lineTo(x, padT + H + 4);
        ctx.stroke();
        ctx.fillText(`${mhz.toFixed(2)}`, x - 14, padT + H + 16);
      }
      ctx.fillText("MHz", padL + W - 24, padT + H + 24);
      ctx.save();
      ctx.translate(12, padT + H / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText("SWR", 0, 0);
      ctx.restore();
    }

    // Per-feed SWR lines.
    swrSeries.forEach((ser, f) => {
      ctx.strokeStyle = feedColor(f, swrSeries.length);
      ctx.lineWidth = thumb ? 1.2 : 1.8;
      ctx.beginPath();
      for (let i = 0; i < ser.length; i++) {
        const x = xT(fs[i]);
        const y = yT(ser[i]);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
    });

    // Measurement-freq marker.
    if (measFreqMhz >= f0 && measFreqMhz <= f1) {
      ctx.strokeStyle = "#cfd6e3";
      ctx.setLineDash(thumb ? [2, 2] : [4, 3]);
      ctx.lineWidth = 1;
      const x = xT(measFreqMhz);
      ctx.beginPath();
      ctx.moveTo(x, padT);
      ctx.lineTo(x, padT + H);
      ctx.stroke();
      ctx.setLineDash([]);
      if (!thumb) {
        ctx.fillStyle = "#cfd6e3";
        ctx.fillText(`${measFreqMhz.toFixed(3)}`, x + 4, padT + 12);
      }
    }
  }, [sweep, z0, measFreqMhz, size, thumb]);
  return <canvas ref={canvasRef} className={thumb ? "thumb-canvas" : undefined} />;
}

// ===========================================================================
// Convergence plot — R, X vs total segment count, one pair per feed.
// ===========================================================================

function ConvergePlot({
  data,
  size = 480,
  thumb = false,
}: {
  data: ConvergeResponse | null;
  size?: number;
  thumb?: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(size * dpr);
    canvas.height = Math.floor(size * dpr);
    canvas.style.width = `${size}px`;
    canvas.style.height = `${size}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#0d1015";
    ctx.fillRect(0, 0, size, size);

    if (!data || data.scales.length < 2) {
      if (!thumb) {
        ctx.fillStyle = "var(--muted)";
        ctx.font = "12px ui-monospace, monospace";
        ctx.fillText("Hit Converge to compute", 12, size / 2);
      }
      return;
    }

    const padL = thumb ? 4 : 50;
    const padR = thumb ? 4 : 14;
    const padT = thumb ? 4 : 14;
    const padB = thumb ? 4 : 30;
    const W = size - padL - padR, H = size - padT - padB;

    const nFeeds = data.z_per_feed[0]?.length ?? 0;
    const Rs: number[][] = [], Xs: number[][] = [];
    for (let f = 0; f < nFeeds; f++) {
      Rs.push(data.z_per_feed.map((zRow) => zRow[f].re));
      Xs.push(data.z_per_feed.map((zRow) => zRow[f].im));
    }
    const allVals = [...Rs.flat(), ...Xs.flat()];
    const lo = Math.min(...allVals), hi = Math.max(...allVals);
    const pad = (hi - lo) * 0.1 || 1;
    const yMin = lo - pad, yMax = hi + pad;
    const xs = data.n_segs_total;
    const xMin = xs[0], xMax = xs[xs.length - 1];

    const xT = (n: number) => padL + ((n - xMin) / Math.max(1, xMax - xMin)) * W;
    const yT = (v: number) => padT + (1 - (v - yMin) / Math.max(1e-6, yMax - yMin)) * H;

    // Grid + axes
    ctx.strokeStyle = "#2a313d";
    ctx.lineWidth = 0.6;
    ctx.fillStyle = "#4a5160";
    ctx.font = "10px ui-monospace, monospace";
    const yTicks = 5;
    for (let i = 0; i <= yTicks; i++) {
      const v = yMin + (i / yTicks) * (yMax - yMin);
      const y = yT(v);
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(padL + W, y);
      ctx.stroke();
      if (!thumb) ctx.fillText(v.toFixed(0), 4, y + 3);
    }
    if (!thumb) {
      for (let i = 0; i < xs.length; i++) {
        ctx.fillText(`${xs[i]}`, xT(xs[i]) - 10, padT + H + 14);
      }
      ctx.save();
      ctx.translate(14, padT + H / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText("Z (Ω)", 0, 0);
      ctx.restore();
      ctx.fillText("total n_segs", padL + W - 70, padT + H + 24);
    }

    // R = solid, X = dashed
    Rs.forEach((ser, f) => {
      ctx.strokeStyle = feedColor(f, nFeeds);
      ctx.lineWidth = thumb ? 1.2 : 1.8;
      ctx.setLineDash([]);
      ctx.beginPath();
      ser.forEach((v, i) => {
        const x = xT(xs[i]), y = yT(v);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
      // dots
      ser.forEach((v, i) => {
        ctx.fillStyle = feedColor(f, nFeeds);
        ctx.beginPath();
        ctx.arc(xT(xs[i]), yT(v), thumb ? 1.5 : 2.5, 0, 2 * Math.PI);
        ctx.fill();
      });
    });
    Xs.forEach((ser, f) => {
      ctx.strokeStyle = feedColor(f, nFeeds, 0.7);
      ctx.lineWidth = thumb ? 1 : 1.5;
      ctx.setLineDash(thumb ? [2, 2] : [4, 3]);
      ctx.beginPath();
      ser.forEach((v, i) => {
        const x = xT(xs[i]), y = yT(v);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
    });
    ctx.setLineDash([]);

    // Legend
    if (!thumb) {
      ctx.fillStyle = "#cfd6e3";
      ctx.fillText("solid = R, dashed = X", padL + 6, padT + 12);
    }
  }, [data, size, thumb]);
  return <canvas ref={canvasRef} className={thumb ? "thumb-canvas" : undefined} />;
}

// ===========================================================================
// Helpers
// ===========================================================================

function defaultValuesFor(schema: BuilderSchema): Record<string, number | ComplexVal> {
  const out: Record<string, number | ComplexVal> = {};
  for (const p of schema.params) out[p.key] = p.default;
  return out;
}

function buildCompareSlots(
  compareResults: Partial<Record<SlotId, SolveResponse>>,
): { slotId: SlotId; z_per_feed: ComplexVal[] }[] {
  // Slot A's markers come from the primary `result` already; B and C
  // ride on top here.
  const out: { slotId: SlotId; z_per_feed: ComplexVal[] }[] = [];
  for (const slotId of ["B", "C"] as SlotId[]) {
    const r = compareResults[slotId];
    if (r) out.push({ slotId, z_per_feed: r.z_per_feed });
  }
  return out;
}

const DEBOUNCE_MS = 50;

// ===========================================================================
// App
// ===========================================================================

export function App() {
  const [builders, setBuilders] = useState<BuilderSchema[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedName, setSelectedName] = useState<string>("dipole");
  const [variant, setVariant] = useState<string>("default");
  const [slots, setSlots] = useState<Slot[]>([
    { id: "A", engineId: "pynec", enabled: true },
    { id: "B", engineId: "pysim-tri", enabled: false },
    { id: "C", engineId: "pysim-sin", enabled: false },
  ]);
  const setSlotEngine = useCallback((slotId: SlotId, engineId: string) => {
    setSlots((ss) => ss.map((s) => (s.id === slotId ? { ...s, engineId } : s)));
  }, []);
  const setSlotEnabled = useCallback((slotId: SlotId, enabled: boolean) => {
    setSlots((ss) => ss.map((s) => (s.id === slotId ? { ...s, enabled } : s)));
  }, []);
  // The primary slot (A) drives wire / currents / FF views; the live
  // single-point solve still uses one engine for those because they
  // don't compare meaningfully across slots.
  const engineId = slots[0].engineId;
  const [ground, setGround] = useState<string>("free");
  const [farField, setFarField] = useState<boolean>(true);
  const [values, setValues] = useState<Record<string, number | ComplexVal>>({});
  const [solving, setSolving] = useState<boolean>(false);
  const [result, setResult] = useState<SolveResponse | null>(null);
  const [solveMs, setSolveMs] = useState<number>(0);
  const [compareResults, setCompareResults] = useState<Partial<Record<SlotId, SolveResponse>>>({});
  const [view, setView] = useState<View>("wire");
  const [z0, setZ0] = useState<number>(50);
  const [projMode, setProjMode] = useState<ProjMode>("auto");
  const [showCurrents, setShowCurrents] = useState<boolean>(true);
  const [azElevDeg, setAzElevDeg] = useState<number>(0);
  const [elPhiDeg, setElPhiDeg] = useState<number>(0);
  const [yaw3d, setYaw3d] = useState<number>(30);
  const [pitch3d, setPitch3d] = useState<number>(25);
  const [zoom3d, setZoom3d] = useState<number>(1.0);
  const [bandStart, setBandStart] = useState<number>(28.0);
  const [bandStop, setBandStop] = useState<number>(29.0);
  const [nSweep, setNSweep] = useState<number>(41);
  const [sweep, setSweep] = useState<SweepResponse | null>(null);
  const [sweeping, setSweeping] = useState<boolean>(false);
  const [converge, setConverge] = useState<ConvergeResponse | null>(null);
  const [converging, setConverging] = useState<boolean>(false);
  const [slideRef, slideSize] = useFillSize<HTMLDivElement>();

  // Orbit-drag handlers for the 3D wire view. Active only when view==='wire3d';
  // attached at the slide div level so dragging anywhere inside the canvas
  // orbits.
  const dragRef = useRef<{ x: number; y: number; yaw: number; pitch: number } | null>(null);
  const onSlidePointerDown = useCallback((e: React.PointerEvent) => {
    if (view !== "wire3d") return;
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    dragRef.current = { x: e.clientX, y: e.clientY, yaw: yaw3d, pitch: pitch3d };
  }, [view, yaw3d, pitch3d]);
  const onSlidePointerMove = useCallback((e: React.PointerEvent) => {
    if (!dragRef.current) return;
    const dx = e.clientX - dragRef.current.x;
    const dy = e.clientY - dragRef.current.y;
    setYaw3d(((dragRef.current.yaw + dx * 0.4) % 360 + 360) % 360);
    setPitch3d(Math.max(-89, Math.min(89, dragRef.current.pitch - dy * 0.4)));
  }, []);
  const onSlidePointerUp = useCallback((e: React.PointerEvent) => {
    (e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId);
    dragRef.current = null;
  }, []);
  const onSlideWheel = useCallback((e: React.WheelEvent) => {
    if (view !== "wire3d") return;
    setZoom3d((z) => Math.max(0.2, Math.min(8, z * (e.deltaY < 0 ? 1.1 : 1 / 1.1))));
  }, [view]);

  // Builder catalogue.
  useEffect(() => {
    fetch("/api/builders")
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`/api/builders → ${r.status}`))))
      .then((data: BuildersResp) => {
        setBuilders(data.builders);
        const first = data.builders.find((b) => b.name === "dipole") ?? data.builders[0];
        if (first) {
          setSelectedName(first.name);
          setValues(defaultValuesFor(first));
        }
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  // Refetch schema on builder/variant change.
  const schema = useMemo(
    () => builders?.find((b) => b.name === selectedName) ?? null,
    [builders, selectedName],
  );
  useEffect(() => {
    if (!selectedName) return;
    const url = `/api/builder/${selectedName}?variant=${encodeURIComponent(variant)}`;
    fetch(url)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${url} → ${r.status}`))))
      .then((s: BuilderSchema) => {
        setValues(defaultValuesFor(s));
        setBuilders((prev) =>
          prev ? prev.map((b) => (b.name === s.name ? { ...b, params: s.params } : b)) : prev,
        );
      })
      .catch((e: Error) => setError(e.message));
  }, [selectedName, variant]);

  const onParamChange = useCallback((key: string, val: number | ComplexVal) => {
    setValues((v) => ({ ...v, [key]: val }));
    // Cached sweep / converge are tied to the antenna being solved; a
    // param edit means they're stale.
    setSweep(null);
    setConverge(null);
  }, []);

  const runSweep = useCallback(async () => {
    if (!schema) return;
    const engineChoice = ENGINE_CHOICES.find((c) => c.id === engineId);
    if (!engineChoice) return;
    if (bandStop <= bandStart) {
      setError("Sweep: band_stop must be > band_start");
      return;
    }
    setSweeping(true);
    try {
      const req: SweepRequest = {
        builder: selectedName,
        variant,
        params: values,
        engine: engineChoice.engine,
        pysim_basis: engineChoice.pysim_basis,
        ground,
        band_start_mhz: bandStart,
        band_stop_mhz: bandStop,
        n_points: nSweep,
      };
      const r = await fetch("/api/sweep", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(req),
      });
      if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
      const j: SweepResponse = await r.json();
      setSweep(j);
      setError(null);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSweeping(false);
    }
  }, [schema, selectedName, variant, values, engineId, ground, bandStart, bandStop, nSweep]);

  const runConverge = useCallback(async () => {
    if (!schema) return;
    const engineChoice = ENGINE_CHOICES.find((c) => c.id === engineId);
    if (!engineChoice) return;
    setConverging(true);
    try {
      const req: ConvergeRequest = {
        builder: selectedName,
        variant,
        params: values,
        engine: engineChoice.engine,
        pysim_basis: engineChoice.pysim_basis,
        ground,
      };
      const r = await fetch("/api/converge", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(req),
      });
      if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
      setConverge(await r.json());
      setError(null);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setConverging(false);
    }
  }, [schema, selectedName, variant, values, engineId, ground]);

  // Debounced auto-solve. Fires one /api/solve per enabled slot, the
  // primary (A) drives the wire/FF/currents views, B/C are stored under
  // compareResults for the Smith chart overlay. AbortController cancels
  // every in-flight request when a newer slider edit comes in.
  const abortRef = useRef<AbortController | null>(null);
  useEffect(() => {
    if (!schema) return;
    const handle = setTimeout(() => {
      abortRef.current?.abort();
      const ac = new AbortController();
      abortRef.current = ac;
      setSolving(true);
      const t0 = performance.now();

      const slotJobs = slots
        .filter((s) => s.enabled)
        .map((s) => {
          const ec = ENGINE_CHOICES.find((c) => c.id === s.engineId);
          if (!ec) return null;
          const req: SolveRequest = {
            builder: selectedName,
            variant,
            params: values,
            engine: ec.engine,
            pysim_basis: ec.pysim_basis,
            ground,
            far_field: s.id === "A" ? farField : false,
          };
          return fetch("/api/solve", {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify(req),
            signal: ac.signal,
          })
            .then(async (r) => {
              if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
              return r.json() as Promise<SolveResponse>;
            })
            .then((j) => ({ slotId: s.id, result: j }));
        })
        .filter((x): x is Promise<{ slotId: SlotId; result: SolveResponse }> => x !== null);

      Promise.allSettled(slotJobs)
        .then((settled) => {
          const out: Partial<Record<SlotId, SolveResponse>> = {};
          let primary: SolveResponse | null = null;
          for (const s of settled) {
            if (s.status === "fulfilled") {
              out[s.value.slotId] = s.value.result;
              if (s.value.slotId === "A") primary = s.value.result;
            } else if ((s.reason as Error)?.name !== "AbortError") {
              setError((s.reason as Error).message);
            }
          }
          if (primary) {
            setResult(primary);
            setSolveMs(performance.now() - t0);
            setError(null);
          }
          setCompareResults(out);
        })
        .finally(() => {
          if (abortRef.current === ac) setSolving(false);
        });
    }, DEBOUNCE_MS);
    return () => clearTimeout(handle);
  }, [schema, selectedName, variant, values, slots, ground, farField]);

  if (error && !builders) {
    return (
      <div className="app">
        <div className="sidebar"><div className="error">{error}</div></div>
      </div>
    );
  }
  if (!builders) {
    return (
      <div className="app">
        <div className="sidebar">
          <h1>antenna_designer</h1>
          <div className="empty-state">Loading builders…</div>
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      <div className="sidebar">
        <h1>antenna_designer</h1>

        <div className="group-label">Antenna</div>
        <div className="geometry-select-row">
          <span className="geometry-select-label">builder</span>
          <select
            className="geometry-select"
            value={selectedName}
            onChange={(e) => { setSelectedName(e.target.value); setVariant("default"); }}
          >
            {builders.map((b) => (<option key={b.name} value={b.name}>{b.name}</option>))}
          </select>
        </div>
        {schema && schema.variants.length > 1 && (
          <div className="geometry-select-row">
            <span className="geometry-select-label">variant</span>
            <select className="geometry-select" value={variant} onChange={(e) => setVariant(e.target.value)}>
              {schema.variants.map((v) => (<option key={v} value={v}>{v}</option>))}
            </select>
          </div>
        )}

        <div className="group-label">Engine slots</div>
        {slots.map((slot) => (
          <div key={slot.id} className="field-row" style={{ gap: 6 }}>
            <label className="checkbox-row" style={{ minWidth: 28, color: "var(--accent)", fontWeight: 600 }}>
              <input
                type="checkbox"
                checked={slot.enabled}
                disabled={slot.id === "A"}
                onChange={(e) => setSlotEnabled(slot.id, e.target.checked)}
              />
              {slot.id}
            </label>
            <div className="backend-tabs" style={{ flex: 1 }}>
              {ENGINE_CHOICES.map((c) => (
                <button
                  key={c.id}
                  className={c.id === slot.engineId ? "backend-tab-btn active" : "backend-tab-btn"}
                  onClick={() => setSlotEngine(slot.id, c.id)}
                  disabled={!slot.enabled}
                  title={c.engine === "pynec" ? "PyNEC (NEC2)" : `pysim — ${c.pysim_basis}`}
                >
                  <span className="slot-letter">{c.letter}</span>
                  <span className="slot-sub">{c.sub}</span>
                </button>
              ))}
            </div>
          </div>
        ))}
        <div className="geometry-select-row">
          <span className="geometry-select-label">ground</span>
          <select className="geometry-select" value={ground} onChange={(e) => setGround(e.target.value)}>
            {GROUND_CHOICES.map((c) => (<option key={c.id} value={c.id}>{c.label}</option>))}
          </select>
        </div>
        <label className="checkbox-row">
          <input type="checkbox" checked={farField} onChange={(e) => setFarField(e.target.checked)} />
          compute far-field
        </label>

        <div className="group-label">Parameters</div>
        {schema && <ParamSliders params={schema.params} values={values} onChange={onParamChange} />}

        {error && <div className="error">{error}</div>}

        <div className="group-label">Sweep</div>
        <div className="field">
          <label><span>band start (MHz)</span><span className="val">{bandStart.toFixed(2)}</span></label>
          <input
            type="number"
            value={bandStart}
            step={0.05}
            onChange={(e) => { setBandStart(parseFloat(e.target.value)); setSweep(null); }}
          />
        </div>
        <div className="field">
          <label><span>band stop (MHz)</span><span className="val">{bandStop.toFixed(2)}</span></label>
          <input
            type="number"
            value={bandStop}
            step={0.05}
            onChange={(e) => { setBandStop(parseFloat(e.target.value)); setSweep(null); }}
          />
        </div>
        <div className="field">
          <label><span>n points</span><span className="val">{nSweep}</span></label>
          <input
            type="range"
            min={5}
            max={101}
            step={2}
            value={nSweep}
            onChange={(e) => { setNSweep(parseInt(e.target.value, 10)); setSweep(null); }}
          />
        </div>
        <div className="field-row">
          <button
            className="backend-tab-btn"
            style={{ borderRadius: 4, flex: 1 }}
            onClick={runSweep}
            disabled={sweeping}
          >
            <span className="slot-letter">{sweeping ? "Sweeping…" : "Sweep"}</span>
          </button>
          <button
            className="backend-tab-btn"
            style={{ borderRadius: 4, flex: 1 }}
            onClick={runConverge}
            disabled={converging}
          >
            <span className="slot-letter">{converging ? "Converging…" : "Converge"}</span>
          </button>
        </div>

        {result && (
          <>
            <div className="group-label">Readout</div>
            <div className="readout">
              <div className="row">
                <span>freq</span>
                <span className="val">{result.freq_mhz.toFixed(3)} MHz</span>
              </div>
              {result.far_field && (
                <div className="row">
                  <span>max gain</span>
                  <span className="val-hot">{result.far_field.max_gain.toFixed(2)} dBi</span>
                </div>
              )}
              <div className="feeds-table">
                <div className="feeds-table-header">
                  <span>feed</span><span>R (Ω)</span><span>X (Ω)</span>
                </div>
                {result.z_per_feed.map((z, i) => (
                  <div className="row" key={i}>
                    <span>
                      <span className="feed-swatch" style={{ background: feedColor(i, result.z_per_feed.length) }} />
                      f{i}
                    </span>
                    <span className="val">{z.re.toFixed(1)}</span>
                    <span className="val">{z.im.toFixed(1)}</span>
                  </div>
                ))}
              </div>
              <div className="field-row" style={{ marginTop: 6 }}>
                <span style={{ color: "var(--muted)" }}>Z₀</span>
                <input
                  type="number"
                  value={z0}
                  step={1}
                  onChange={(e) => setZ0(parseFloat(e.target.value) || 50)}
                  style={{ width: 70 }}
                />
              </div>
            </div>
          </>
        )}
      </div>

      <div className="stage">
        <div className="thumbstrip">
          <button
            className={view === "wire" ? "thumb active" : "thumb"}
            onClick={() => setView("wire")}
          >
            {result
              ? <WireCanvas wires={result.wires} currents={result.currents} size={64} projMode={projMode} showCurrents={showCurrents} thumb />
              : <div className="thumb-canvas" />}
            <span className="thumb-label">Wire 2D</span>
          </button>
          <button
            className={view === "wire3d" ? "thumb active" : "thumb"}
            onClick={() => setView("wire3d")}
          >
            {result
              ? <Wire3DCanvas wires={result.wires} currents={result.currents} size={64} yawDeg={yaw3d} pitchDeg={pitch3d} zoom={zoom3d} showCurrents={showCurrents} thumb />
              : <div className="thumb-canvas" />}
            <span className="thumb-label">Wire 3D</span>
          </button>
          <button
            className={view === "smith" ? "thumb active" : "thumb"}
            onClick={() => setView("smith")}
          >
            {result
              ? <SmithChart
                  z_per_feed={result.z_per_feed}
                  z0={z0}
                  size={64}
                  thumb
                  sweep={sweep}
                  compareSlots={buildCompareSlots(compareResults)}
                />
              : <div className="thumb-canvas" />}
            <span className="thumb-label">Smith</span>
          </button>
          <button
            className={view === "ff-az" ? "thumb active" : "thumb"}
            onClick={() => setView("ff-az")}
            disabled={!result?.far_field}
          >
            {result?.far_field
              ? <FarFieldPlot ff={result.far_field} size={64} cut="azimuth" cutAngleDeg={azElevDeg} thumb />
              : <div className="thumb-canvas" />}
            <span className="thumb-label">FF az</span>
          </button>
          <button
            className={view === "ff-el" ? "thumb active" : "thumb"}
            onClick={() => setView("ff-el")}
            disabled={!result?.far_field}
          >
            {result?.far_field
              ? <FarFieldPlot ff={result.far_field} size={64} cut="elevation" cutAngleDeg={elPhiDeg} thumb />
              : <div className="thumb-canvas" />}
            <span className="thumb-label">FF el</span>
          </button>
          <button
            className={view === "sweep" ? "thumb active" : "thumb"}
            onClick={() => setView("sweep")}
          >
            {sweep
              ? <SwrPlot sweep={sweep} z0={z0} measFreqMhz={result?.freq_mhz ?? 0} size={64} thumb />
              : <div className="thumb-canvas" />}
            <span className="thumb-label">SWR</span>
          </button>
          <button
            className={view === "converge" ? "thumb active" : "thumb"}
            onClick={() => setView("converge")}
          >
            {converge
              ? <ConvergePlot data={converge} size={64} thumb />
              : <div className="thumb-canvas" />}
            <span className="thumb-label">Conv.</span>
          </button>
        </div>

        <div
          className="carousel-slide"
          ref={slideRef}
          onPointerDown={onSlidePointerDown}
          onPointerMove={onSlidePointerMove}
          onPointerUp={onSlidePointerUp}
          onWheel={onSlideWheel}
          style={view === "wire3d" ? { cursor: dragRef.current ? "grabbing" : "grab" } : undefined}
        >
          {!result && <div className="empty-state">Loading…</div>}

          {result && view === "wire" && (
            <>
              <WireCanvas
                wires={result.wires}
                currents={result.currents}
                projMode={projMode}
                showCurrents={showCurrents}
                size={slideSize}
              />
              <div className="antenna-overlay">
                <div className="projection-toggle">
                  {(["auto", "xy", "xz", "yz"] as ProjMode[]).map((m) => (
                    <button
                      key={m}
                      className={m === projMode ? "active" : ""}
                      onClick={() => setProjMode(m)}
                    >{m}</button>
                  ))}
                </div>
                <label className="overlay-checkbox">
                  <input
                    type="checkbox"
                    checked={showCurrents}
                    onChange={(e) => setShowCurrents(e.target.checked)}
                  />
                  currents
                </label>
              </div>
            </>
          )}

          {result && view === "wire3d" && (
            <>
              <Wire3DCanvas
                wires={result.wires}
                currents={result.currents}
                size={slideSize}
                yawDeg={yaw3d}
                pitchDeg={pitch3d}
                zoom={zoom3d}
                showCurrents={showCurrents}
              />
              <div className="antenna-overlay">
                <label className="overlay-checkbox">
                  <input
                    type="checkbox"
                    checked={showCurrents}
                    onChange={(e) => setShowCurrents(e.target.checked)}
                  />
                  currents
                </label>
                <button
                  className="overlay-checkbox"
                  onClick={() => { setYaw3d(30); setPitch3d(25); setZoom3d(1.0); }}
                  style={{ cursor: "pointer", border: "1px solid #2a2f38" }}
                >
                  reset view
                </button>
              </div>
            </>
          )}

          {result && view === "smith" && (
            <SmithChart
              z_per_feed={result.z_per_feed}
              z0={z0}
              size={slideSize}
              sweep={sweep}
              compareSlots={buildCompareSlots(compareResults)}
            />
          )}

          {result && view === "sweep" && (
            <SwrPlot sweep={sweep} z0={z0} measFreqMhz={result.freq_mhz} size={slideSize} />
          )}

          {result && view === "converge" && (
            <ConvergePlot data={converge} size={slideSize} />
          )}

          {result && view === "ff-az" && result.far_field && (
            <>
              <FarFieldPlot ff={result.far_field} size={slideSize} cut="azimuth" cutAngleDeg={azElevDeg} />
              <div className="antenna-overlay">
                <div className="overlay-slider">
                  <span>elev</span>
                  <input
                    type="range"
                    min={0}
                    max={89}
                    step={1}
                    value={azElevDeg}
                    onChange={(e) => setAzElevDeg(parseFloat(e.target.value))}
                  />
                  <span className="val">{azElevDeg.toFixed(0)}°</span>
                </div>
              </div>
            </>
          )}

          {result && view === "ff-el" && result.far_field && (
            <>
              <FarFieldPlot ff={result.far_field} size={slideSize} cut="elevation" cutAngleDeg={elPhiDeg} />
              <div className="antenna-overlay">
                <div className="overlay-slider">
                  <span>az</span>
                  <input
                    type="range"
                    min={0}
                    max={359}
                    step={1}
                    value={elPhiDeg}
                    onChange={(e) => setElPhiDeg(parseFloat(e.target.value))}
                  />
                  <span className="val">{elPhiDeg.toFixed(0)}°</span>
                </div>
              </div>
            </>
          )}
        </div>

        {result && (
          <div className="status">
            {solving && <span className="solving-dot" />}
            {result.builder}
            {result.variant !== "default" && `:${result.variant}`} · {result.engine}
            {" · "}
            {result.freq_mhz.toFixed(3)} MHz · {solveMs.toFixed(0)} ms
          </div>
        )}
      </div>
    </div>
  );
}
