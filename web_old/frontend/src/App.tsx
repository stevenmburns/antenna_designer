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
  wire_radius?: number;
  solver_kwargs?: Record<string, number>;
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
type Slot = {
  id: SlotId;
  engineId: string;
  enabled: boolean;
  wireRadius: number;
  // Solver-basis-specific kwargs. Keyed by the basis they apply to so
  // switching engines doesn't lose the previous basis's settings.
  pysimKwargs: {
    triangular: { n_qp_reg: number; n_qp_off: number };
    sinusoidal: { n_qp_const: number };
    bspline: Record<string, never>;
  };
};

const DEFAULT_SLOT_KWARGS: Slot["pysimKwargs"] = {
  triangular: { n_qp_reg: 4, n_qp_off: 4 },
  sinusoidal: { n_qp_const: 16 },
  bspline: {},
};

function slotSolverKwargs(slot: Slot): Record<string, number> {
  const ec = ENGINE_CHOICES.find((c) => c.id === slot.engineId);
  if (!ec || ec.engine !== "pysim") return {};
  const basis = ec.pysim_basis as keyof Slot["pysimKwargs"];
  return slot.pysimKwargs[basis] as Record<string, number>;
}

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
type View = "wire" | "smith" | "ff-az" | "ff-el";

const VIEW_LABEL: Record<View, string> = {
  wire: "Wire 2D",
  smith: "Smith",
  "ff-az": "FF az",
  "ff-el": "FF el",
};

const VIEW_ORDER: View[] = ["wire", "smith", "ff-az", "ff-el"];

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

function useObservedHeight<T extends HTMLElement>(): [RefObject<T>, number] {
  const ref = useRef<T>(null);
  // Seed with viewport height so the first paint already has sane thumb
  // sizes — without this seed the thumbstrip would be ~80px tall on the
  // initial layout (label-only thumbs collapse the flex container's
  // implicit height) and ResizeObserver settles at that small value
  // instead of growing to the real cell height.
  const [h, setH] = useState(() =>
    typeof window !== "undefined" ? window.innerHeight : 0,
  );
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    setH(el.clientHeight || window.innerHeight);
    const ro = new ResizeObserver((entries) => {
      const obs = entries[0].contentRect.height;
      setH(obs > 0 ? obs : el.clientHeight);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, h];
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
  converge,
  compareSlots,
}: {
  z_per_feed: ComplexVal[];
  z0: number;
  size?: number;
  thumb?: boolean;
  sweep?: SweepResponse | null;
  converge?: ConvergeResponse | null;
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

    // Skip the radius-driven drawing when the canvas hasn't been laid out
    // yet — the ResizeObserver on the thumbstrip fires after the first
    // render, so we briefly see size=0 → R<0 which would throw on
    // ctx.arc.
    if (R <= 0) return;

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

    // Optional convergence trail — one polyline per feed across the
    // segmentation scales, with markers at each scale point. Visually
    // distinct from the sweep trail (dashed, hot-yellow tint) so the two
    // can coexist on the same chart without confusion.
    if (converge && converge.z_per_feed.length > 1) {
      const nFeedsConv = converge.z_per_feed[0]?.length ?? 0;
      for (let f = 0; f < nFeedsConv; f++) {
        ctx.strokeStyle = feedColor(f, nFeedsConv, 0.7);
        ctx.lineWidth = thumb ? 1.0 : 1.4;
        ctx.setLineDash(thumb ? [2, 2] : [4, 3]);
        ctx.beginPath();
        let first = true;
        for (let k = 0; k < converge.z_per_feed.length; k++) {
          const z = converge.z_per_feed[k][f];
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
        ctx.setLineDash([]);
        // small dot per scale
        for (let k = 0; k < converge.z_per_feed.length; k++) {
          const z = converge.z_per_feed[k][f];
          if (!z) continue;
          const zn_r = z.re / z0, zn_x = z.im / z0;
          const dR = (zn_r + 1) * (zn_r + 1) + zn_x * zn_x;
          const gR = ((zn_r - 1) * (zn_r + 1) + zn_x * zn_x) / dR;
          const gI = (2 * zn_x) / dR;
          const px = cx + gR * R, py = cy - gI * R;
          ctx.fillStyle = "var(--feed)";
          ctx.beginPath();
          ctx.arc(px, py, thumb ? 1.5 : 2.5, 0, 2 * Math.PI);
          ctx.fill();
        }
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
    if (R <= 0) return;
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
// Helpers
// ===========================================================================

function defaultValuesFor(schema: BuilderSchema): Record<string, number | ComplexVal> {
  const out: Record<string, number | ComplexVal> = {};
  for (const p of schema.params) out[p.key] = p.default;
  return out;
}

type ViewProps = {
  z0: number;
  projMode: ProjMode;
  showCurrents: boolean;
  azElevDeg: number;
  elPhiDeg: number;
  sweep: SweepResponse | null;
  converge: ConvergeResponse | null;
  compareSlots: { slotId: SlotId; z_per_feed: ComplexVal[] }[];
};

function renderView(
  v: View,
  mode: "thumb" | "main",
  result: SolveResponse,
  p: ViewProps,
  size: number,
) {
  const thumb = mode === "thumb";
  switch (v) {
    case "wire":
      return (
        <WireCanvas
          wires={result.wires}
          currents={result.currents}
          size={size}
          projMode={p.projMode}
          showCurrents={p.showCurrents}
          thumb={thumb}
        />
      );
    case "smith":
      return (
        <SmithChart
          z_per_feed={result.z_per_feed}
          z0={p.z0}
          size={size}
          thumb={thumb}
          sweep={p.sweep}
          converge={p.converge}
          compareSlots={p.compareSlots}
        />
      );
    case "ff-az":
      return result.far_field
        ? <FarFieldPlot ff={result.far_field} size={size} cut="azimuth" cutAngleDeg={p.azElevDeg} thumb={thumb} />
        : <div className="thumb-canvas" style={{ width: size, height: size }} />;
    case "ff-el":
      return result.far_field
        ? <FarFieldPlot ff={result.far_field} size={size} cut="elevation" cutAngleDeg={p.elPhiDeg} thumb={thumb} />
        : <div className="thumb-canvas" style={{ width: size, height: size }} />;
  }
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
    { id: "A", engineId: "pynec",     enabled: true,  wireRadius: 0.0005, pysimKwargs: DEFAULT_SLOT_KWARGS },
    { id: "B", engineId: "pysim-tri", enabled: false, wireRadius: 0.0005, pysimKwargs: DEFAULT_SLOT_KWARGS },
    { id: "C", engineId: "pysim-sin", enabled: false, wireRadius: 0.0005, pysimKwargs: DEFAULT_SLOT_KWARGS },
  ]);
  const updateSlot = useCallback((slotId: SlotId, patch: Partial<Slot>) => {
    setSlots((ss) => ss.map((s) => (s.id === slotId ? { ...s, ...patch } : s)));
  }, []);
  const setSlotEnabled = useCallback((slotId: SlotId, enabled: boolean) => {
    setSlots((ss) => ss.map((s) => (s.id === slotId ? { ...s, enabled } : s)));
  }, []);
  const [settingsSlotId, setSettingsSlotId] = useState<SlotId | null>(null);
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
  const [bandStart, setBandStart] = useState<number>(28.0);
  const [bandStop, setBandStop] = useState<number>(29.0);
  const [nSweep, setNSweep] = useState<number>(41);
  const [sweep, setSweep] = useState<SweepResponse | null>(null);
  const [sweeping, setSweeping] = useState<boolean>(false);
  const [converge, setConverge] = useState<ConvergeResponse | null>(null);
  const [converging, setConverging] = useState<boolean>(false);
  const [slideRef, slideSize] = useFillSize<HTMLDivElement>();
  const [stageRef, stageH] = useObservedHeight<HTMLDivElement>();
  // Thumbstrip distributes 3 thumbs vertically; each thumb reserves ~32px
  // of vertical space for the label + padding around the canvas. Scale
  // purely from the available height — no floor, so a tiny window just
  // gets tiny thumbs rather than triggering scrollbars.
  const thumbSide = Math.max(0, Math.floor(stageH / 3) - 32);

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
            wire_radius: s.wireRadius,
            solver_kwargs: slotSolverKwargs(s),
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
        <div className="backend-tabs">
          {slots.map((slot) => {
            const ec = ENGINE_CHOICES.find((c) => c.id === slot.engineId);
            return (
              <div key={slot.id} className="backend-tab-cell">
                <button
                  className={slot.enabled ? "backend-tab-btn active" : "backend-tab-btn"}
                  onClick={() =>
                    slot.id === "A" ? undefined : setSlotEnabled(slot.id, !slot.enabled)
                  }
                  title={
                    slot.id === "A"
                      ? `Slot A — always on (${ec?.letter ?? "?"})`
                      : `Slot ${slot.id}: ${slot.enabled ? "on" : "off"} — click to toggle`
                  }
                >
                  <span className="slot-letter">{slot.id}</span>
                  <span className="slot-sub">{ec?.letter ?? "?"}</span>
                </button>
                <button
                  className="backend-gear-btn"
                  onClick={() => setSettingsSlotId(slot.id)}
                  title={`Configure slot ${slot.id}`}
                >
                  ⚙
                </button>
              </div>
            );
          })}
        </div>
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

      <div className="stage" ref={stageRef}>
        <div className="thumbstrip" style={{ width: thumbSide + 16 }}>
          {VIEW_ORDER.filter((v) => v !== view).map((v) => {
            const disabled = (v === "ff-az" || v === "ff-el") && !result?.far_field;
            return (
              <button
                key={v}
                className="thumb"
                onClick={() => setView(v)}
                disabled={disabled}
                style={{ width: thumbSide + 8 }}
              >
                {result
                  ? renderView(
                      v,
                      "thumb",
                      result,
                      {
                        z0,
                        projMode,
                        showCurrents,
                        azElevDeg,
                        elPhiDeg,
                        sweep,
                        converge,
                        compareSlots: buildCompareSlots(compareResults),
                      },
                      thumbSide,
                    )
                  : <div className="thumb-canvas" style={{ width: thumbSide, height: thumbSide }} />}
                <span className="thumb-label">{VIEW_LABEL[v]}</span>
              </button>
            );
          })}
        </div>

        <div className="carousel-slide" ref={slideRef}>
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

          {result && view === "smith" && (
            <SmithChart
              z_per_feed={result.z_per_feed}
              z0={z0}
              size={slideSize}
              sweep={sweep}
              converge={converge}
              compareSlots={buildCompareSlots(compareResults)}
            />
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

      {settingsSlotId && (
        <SlotSettingsModal
          slot={slots.find((s) => s.id === settingsSlotId)!}
          onChange={(patch) => updateSlot(settingsSlotId, patch)}
          onClose={() => setSettingsSlotId(null)}
        />
      )}
    </div>
  );
}

// ===========================================================================
// Settings modal — picks the engine for one slot and edits its knobs.
// ===========================================================================

function SlotSettingsModal({
  slot,
  onChange,
  onClose,
}: {
  slot: Slot;
  onChange: (patch: Partial<Slot>) => void;
  onClose: () => void;
}) {
  const ec = ENGINE_CHOICES.find((c) => c.id === slot.engineId);
  const isPysim = ec?.engine === "pysim";
  const basis = ec?.pysim_basis as keyof Slot["pysimKwargs"] | undefined;

  const updateKwarg = (k: string, v: number) => {
    if (!basis) return;
    onChange({
      pysimKwargs: {
        ...slot.pysimKwargs,
        [basis]: { ...(slot.pysimKwargs[basis] as object), [k]: v },
      },
    });
  };

  return (
    <div className="backend-config-overlay" onClick={onClose}>
      <div className="backend-config-modal" onClick={(e) => e.stopPropagation()}>
        <div className="backend-config-header">
          <div>Slot {slot.id} configuration</div>
          <button className="backend-config-close" onClick={onClose} aria-label="close">×</button>
        </div>
        <div className="backend-config-body">
          <div className="group-label">Engine</div>
          <div className="backend-tabs">
            {ENGINE_CHOICES.map((c) => (
              <button
                key={c.id}
                className={c.id === slot.engineId ? "backend-tab-btn active" : "backend-tab-btn"}
                onClick={() => onChange({ engineId: c.id })}
              >
                <span className="slot-letter">{c.letter}</span>
                <span className="slot-sub">{c.sub}</span>
              </button>
            ))}
          </div>

          {isPysim && (
            <>
              <div className="group-label">Pysim knobs</div>
              <div className="field">
                <label>
                  <span>wire_radius (m)</span>
                  <span className="val">{slot.wireRadius.toFixed(4)}</span>
                </label>
                <input
                  type="number"
                  value={slot.wireRadius}
                  step={0.0001}
                  min={1e-5}
                  onChange={(e) => onChange({ wireRadius: parseFloat(e.target.value) || 0.0005 })}
                />
              </div>
              {basis === "triangular" && (
                <>
                  <div className="field">
                    <label>
                      <span>n_qp_reg</span>
                      <span className="val">{slot.pysimKwargs.triangular.n_qp_reg}</span>
                    </label>
                    <input
                      type="range"
                      min={2}
                      max={16}
                      step={1}
                      value={slot.pysimKwargs.triangular.n_qp_reg}
                      onChange={(e) => updateKwarg("n_qp_reg", parseInt(e.target.value, 10))}
                    />
                  </div>
                  <div className="field">
                    <label>
                      <span>n_qp_off</span>
                      <span className="val">{slot.pysimKwargs.triangular.n_qp_off}</span>
                    </label>
                    <input
                      type="range"
                      min={2}
                      max={16}
                      step={1}
                      value={slot.pysimKwargs.triangular.n_qp_off}
                      onChange={(e) => updateKwarg("n_qp_off", parseInt(e.target.value, 10))}
                    />
                  </div>
                </>
              )}
              {basis === "sinusoidal" && (
                <div className="field">
                  <label>
                    <span>n_qp_const</span>
                    <span className="val">{slot.pysimKwargs.sinusoidal.n_qp_const}</span>
                  </label>
                  <input
                    type="range"
                    min={4}
                    max={64}
                    step={1}
                    value={slot.pysimKwargs.sinusoidal.n_qp_const}
                    onChange={(e) => updateKwarg("n_qp_const", parseInt(e.target.value, 10))}
                  />
                </div>
              )}
              {basis === "bspline" && (
                <div className="empty-state">No knobs exposed for B-spline yet.</div>
              )}
            </>
          )}
          {!isPysim && (
            <div className="empty-state">PyNEC has no slot-level knobs yet.</div>
          )}
        </div>
        <div className="backend-config-footer">
          <button
            className="backend-config-reset"
            onClick={() => onChange({
              wireRadius: 0.0005,
              pysimKwargs: DEFAULT_SLOT_KWARGS,
            })}
          >
            reset to defaults
          </button>
        </div>
      </div>
    </div>
  );
}
