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

const GROUND_CHOICES = [
  { id: "free", label: "Free space" },
  { id: "pec", label: "PEC plane" },
  { id: "finite:13,0.005", label: "Finite (ε=13, σ=0.005)" },
];

type ProjMode = "auto" | "xy" | "xz" | "yz";
type Projection = "xy" | "xz" | "yz";
type FFCut = "azimuth" | "elevation";
type View = "wire" | "smith" | "ff-az" | "ff-el";

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
}: {
  z_per_feed: ComplexVal[];
  z0: number;
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
    const n = z_per_feed.length;
    z_per_feed.forEach((z, i) => {
      const zn_r = z.re / z0, zn_x = z.im / z0;
      const denomR = (zn_r + 1) * (zn_r + 1) + zn_x * zn_x;
      const gR = ((zn_r - 1) * (zn_r + 1) + zn_x * zn_x) / denomR;
      const gI = (2 * zn_x) / denomR;
      const px = cx + gR * R, py = cy - gI * R;
      ctx.fillStyle = feedColor(i, n);
      ctx.beginPath();
      ctx.arc(px, py, thumb ? 3 : 6, 0, 2 * Math.PI);
      ctx.fill();
      ctx.strokeStyle = "#0d1015";
      ctx.lineWidth = 1.5;
      ctx.stroke();
      if (!thumb) {
        ctx.fillStyle = "#cfd6e3";
        ctx.font = "10px ui-monospace, monospace";
        ctx.fillText(`f${i}`, px + 10, py + 4);
      }
    });
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
// Helpers
// ===========================================================================

function defaultValuesFor(schema: BuilderSchema): Record<string, number | ComplexVal> {
  const out: Record<string, number | ComplexVal> = {};
  for (const p of schema.params) out[p.key] = p.default;
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
  const [engineId, setEngineId] = useState<string>("pynec");
  const [ground, setGround] = useState<string>("free");
  const [farField, setFarField] = useState<boolean>(true);
  const [values, setValues] = useState<Record<string, number | ComplexVal>>({});
  const [solving, setSolving] = useState<boolean>(false);
  const [result, setResult] = useState<SolveResponse | null>(null);
  const [solveMs, setSolveMs] = useState<number>(0);
  const [view, setView] = useState<View>("wire");
  const [z0, setZ0] = useState<number>(50);
  const [projMode, setProjMode] = useState<ProjMode>("auto");
  const [showCurrents, setShowCurrents] = useState<boolean>(true);
  const [azElevDeg, setAzElevDeg] = useState<number>(0);
  const [elPhiDeg, setElPhiDeg] = useState<number>(0);
  const [slideRef, slideSize] = useFillSize<HTMLDivElement>();

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
  }, []);

  // Debounced auto-solve. A newer slider edit aborts the in-flight request
  // before issuing a fresh one — keeps the UI responsive without flooding
  // the backend during a slider drag.
  const abortRef = useRef<AbortController | null>(null);
  useEffect(() => {
    if (!schema) return;
    const engineChoice = ENGINE_CHOICES.find((c) => c.id === engineId);
    if (!engineChoice) return;
    const handle = setTimeout(() => {
      abortRef.current?.abort();
      const ac = new AbortController();
      abortRef.current = ac;
      const req: SolveRequest = {
        builder: selectedName,
        variant,
        params: values,
        engine: engineChoice.engine,
        pysim_basis: engineChoice.pysim_basis,
        ground,
        far_field: farField,
      };
      setSolving(true);
      const t0 = performance.now();
      fetch("/api/solve", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(req),
        signal: ac.signal,
      })
        .then(async (r) => {
          if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
          return r.json();
        })
        .then((j: SolveResponse) => {
          setResult(j);
          setSolveMs(performance.now() - t0);
          setError(null);
        })
        .catch((e: Error) => {
          if (e.name !== "AbortError") setError(e.message);
        })
        .finally(() => {
          if (abortRef.current === ac) setSolving(false);
        });
    }, DEBOUNCE_MS);
    return () => clearTimeout(handle);
  }, [schema, selectedName, variant, values, engineId, ground, farField]);

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

        <div className="group-label">Engine</div>
        <div className="backend-tabs">
          {ENGINE_CHOICES.map((c) => (
            <button
              key={c.id}
              className={c.id === engineId ? "backend-tab-btn active" : "backend-tab-btn"}
              onClick={() => setEngineId(c.id)}
              title={c.engine === "pynec" ? "PyNEC (NEC2)" : `pysim — ${c.pysim_basis}`}
            >
              <span className="slot-letter">{c.letter}</span>
              <span className="slot-sub">{c.sub}</span>
            </button>
          ))}
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
            <span className="thumb-label">Wire</span>
          </button>
          <button
            className={view === "smith" ? "thumb active" : "thumb"}
            onClick={() => setView("smith")}
          >
            {result
              ? <SmithChart z_per_feed={result.z_per_feed} z0={z0} size={64} thumb />
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
            <SmithChart z_per_feed={result.z_per_feed} z0={z0} size={slideSize} />
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
