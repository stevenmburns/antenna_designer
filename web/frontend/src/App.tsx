import { useCallback, useEffect, useMemo, useRef, useState } from "react";

// ---------------------------------------------------------------------------
// API types — mirror the FastAPI shapes in web/server.py.
// ---------------------------------------------------------------------------

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
  thetas: number[]; // degrees
  phis: number[];   // degrees
  rings: number[][]; // [theta_idx][phi_idx] → dBi
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

// ---------------------------------------------------------------------------
// Engine choices. Maps the UI's flat list to (engine, pysim_basis) tuples.
// ---------------------------------------------------------------------------

type EngineChoice = {
  id: string;
  label: string;
  engine: string;
  pysim_basis?: string;
};

const ENGINE_CHOICES: EngineChoice[] = [
  { id: "pynec", label: "PyNEC", engine: "pynec" },
  { id: "pysim-tri", label: "pysim (triangular)", engine: "pysim", pysim_basis: "triangular" },
  { id: "pysim-sin", label: "pysim (sinusoidal)", engine: "pysim", pysim_basis: "sinusoidal" },
  { id: "pysim-bsp", label: "pysim (bspline)", engine: "pysim", pysim_basis: "bspline" },
];

const GROUND_CHOICES = [
  { id: "free", label: "Free space" },
  { id: "pec", label: "PEC plane" },
  { id: "finite:13,0.005", label: "Finite (ε=13, σ=0.005)" },
];

// ---------------------------------------------------------------------------
// Schema-driven sliders.
// ---------------------------------------------------------------------------

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
            <div className="slider-row" key={p.key}>
              <div className="name">{p.key}</div>
              <div className="complex-input" style={{ gridColumn: "1 / span 2" }}>
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
          <div className="slider-row" key={p.key}>
            <div className="name">{p.key}</div>
            <input
              type="range"
              min={p.min}
              max={p.max}
              step={p.step}
              value={cur}
              onChange={(e) => onChange(p.key, parseFloat(e.target.value))}
            />
            <input
              type="number"
              value={cur}
              step={p.step}
              onChange={(e) => onChange(p.key, parseFloat(e.target.value))}
            />
          </div>
        );
      })}
    </>
  );
}

// ---------------------------------------------------------------------------
// 2D wire renderer — auto-picks XY/XZ/YZ projection by the bounding box.
// ---------------------------------------------------------------------------

type Projection = "xy" | "xz" | "yz";

function project(p: [number, number, number], proj: Projection): [number, number] {
  switch (proj) {
    case "xy":
      return [p[0], p[1]];
    case "xz":
      return [p[0], p[2]];
    case "yz":
      return [p[1], p[2]];
  }
}

function pickProjection(wires: WireGeom[]): Projection {
  // Pick the projection whose two axes have the largest spread combined.
  const mins = [Infinity, Infinity, Infinity];
  const maxs = [-Infinity, -Infinity, -Infinity];
  for (const w of wires) {
    for (const p of [w.p0, w.p1]) {
      for (let i = 0; i < 3; i++) {
        if (p[i] < mins[i]) mins[i] = p[i];
        if (p[i] > maxs[i]) maxs[i] = p[i];
      }
    }
  }
  const spread = [maxs[0] - mins[0], maxs[1] - mins[1], maxs[2] - mins[2]];
  const xy = spread[0] + spread[1];
  const xz = spread[0] + spread[2];
  const yz = spread[1] + spread[2];
  if (xz >= xy && xz >= yz) return "xz";
  if (yz >= xy) return "yz";
  return "xy";
}

// Map normalised |I| in [0,1] to a viridis-like ramp.
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
}: {
  wires: WireGeom[];
  currents?: WireCurrent[] | null;
  size?: number;
}) {
  if (wires.length === 0) {
    return <svg width={size} height={size} />;
  }
  const proj = pickProjection(wires);
  const proj2 = wires.map((w) => [project(w.p0, proj), project(w.p1, proj)] as const);
  const xs = proj2.flatMap((pq) => [pq[0][0], pq[1][0]]);
  const ys = proj2.flatMap((pq) => [pq[0][1], pq[1][1]]);
  const xmin = Math.min(...xs);
  const xmax = Math.max(...xs);
  const ymin = Math.min(...ys);
  const ymax = Math.max(...ys);
  const pad = 0.08;
  const dx = Math.max(xmax - xmin, 1e-6);
  const dy = Math.max(ymax - ymin, 1e-6);
  const scale = Math.min((size * (1 - 2 * pad)) / dx, (size * (1 - 2 * pad)) / dy);
  const ox = (size - scale * dx) / 2 - scale * xmin;
  const oy = (size - scale * dy) / 2 - scale * ymin;
  const tx = (x: number) => ox + scale * x;
  const ty = (y: number) => size - (oy + scale * y);

  // Global peak |I| across all wires, used to normalise the colour ramp.
  let peakI = 0;
  if (currents) {
    for (const w of currents) {
      for (let i = 0; i < w.knot_currents_re.length; i++) {
        const m = Math.hypot(w.knot_currents_re[i], w.knot_currents_im[i]);
        if (m > peakI) peakI = m;
      }
    }
  }

  return (
    <svg width={size} height={size}>
      <rect x={0} y={0} width={size} height={size} fill="var(--panel-2)" />
      <text x={6} y={size - 6} fill="var(--text-dim)" fontSize={11} fontFamily="monospace">
        {`${proj} projection — ${dx.toFixed(2)} × ${dy.toFixed(2)} m`}
      </text>
      {/* Base wires — drawn first, then currents overlaid. */}
      {wires.map((w, i) => {
        const [[a, b], [c, d]] = proj2[i];
        const isFeed = w.feed_voltage !== null;
        return (
          <line
            key={`wire-${i}`}
            x1={tx(a)}
            y1={ty(b)}
            x2={tx(c)}
            y2={ty(d)}
            stroke={isFeed ? "var(--feed)" : "var(--wire)"}
            strokeWidth={isFeed ? 2.5 : 1.0}
            opacity={currents && peakI > 0 ? 0.35 : 1}
          />
        );
      })}
      {/* Current overlay: per-knot polyline coloured by |I|. Each wire's
          knot list is a single polyline; render each (k → k+1) segment
          coloured by the segment-midpoint current magnitude. */}
      {currents && peakI > 0 &&
        currents.flatMap((w, wi) => {
          const knots = w.knot_positions;
          const re = w.knot_currents_re;
          const im = w.knot_currents_im;
          const segs = [];
          for (let k = 0; k < knots.length - 1; k++) {
            const [a, b] = project(knots[k], proj);
            const [c, d] = project(knots[k + 1], proj);
            const m1 = Math.hypot(re[k], im[k]);
            const m2 = Math.hypot(re[k + 1], im[k + 1]);
            const m = 0.5 * (m1 + m2);
            segs.push(
              <line
                key={`cur-${wi}-${k}`}
                x1={tx(a)}
                y1={ty(b)}
                x2={tx(c)}
                y2={ty(d)}
                stroke={magToColor(m / peakI)}
                strokeWidth={3.0}
                strokeLinecap="round"
              />,
            );
          }
          return segs;
        })}
      {/* Legend */}
      {currents && peakI > 0 && (
        <g transform={`translate(${size - 110}, 10)`}>
          <rect x={0} y={0} width={100} height={12} fill="url(#cur-grad)" />
          <text x={0} y={26} fill="var(--text-dim)" fontSize={10} fontFamily="monospace">
            {`0`}
          </text>
          <text
            x={100}
            y={26}
            fill="var(--text-dim)"
            fontSize={10}
            fontFamily="monospace"
            textAnchor="end"
          >
            {`|I|=${peakI.toExponential(1)}A`}
          </text>
          <defs>
            <linearGradient id="cur-grad" x1="0" y1="0" x2="1" y2="0">
              {Array.from({ length: 6 }, (_, i) => (
                <stop key={i} offset={`${(i * 100) / 5}%`} stopColor={magToColor(i / 5)} />
              ))}
            </linearGradient>
          </defs>
        </g>
      )}
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Smith chart — per-feed Z markers at a chosen reference impedance.
// ---------------------------------------------------------------------------

function feedColor(i: number, n: number, alpha = 0.95): string {
  const hue = n <= 1 ? 200 : (i * 360) / n;
  return `hsla(${hue}, 80%, 60%, ${alpha})`;
}

function SmithChart({
  z_per_feed,
  z0,
  size = 480,
}: {
  z_per_feed: ComplexVal[];
  z0: number;
  size: number;
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

    const cx = size / 2;
    const cy = size / 2;
    const R = size / 2 - 12;

    ctx.fillStyle = "#0d1015";
    ctx.fillRect(0, 0, size, size);

    // Constant-r circles in Γ plane: centre = (r/(r+1), 0), radius = 1/(r+1)
    ctx.strokeStyle = "#2a313d";
    ctx.lineWidth = 0.6;
    for (const rn of [0.2, 0.5, 1, 2, 5]) {
      ctx.beginPath();
      ctx.arc(cx + (rn / (rn + 1)) * R, cy, (1 / (rn + 1)) * R, 0, 2 * Math.PI);
      ctx.stroke();
    }

    // Constant-x arcs: centre = (1, 1/x), radius = 1/|x|, clipped to unit disk
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

    // Real axis + outer boundary
    ctx.strokeStyle = "#3a4150";
    ctx.lineWidth = 0.8;
    ctx.beginPath();
    ctx.moveTo(cx - R, cy);
    ctx.lineTo(cx + R, cy);
    ctx.stroke();
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(cx, cy, R, 0, 2 * Math.PI);
    ctx.stroke();

    // Labels
    ctx.fillStyle = "#4a5160";
    ctx.font = "10px ui-monospace, monospace";
    ctx.fillText(`Z₀ = ${z0}Ω`, 8, 14);
    ctx.fillText("+jX", cx + R - 26, cy - R + 14);
    ctx.fillText("−jX", cx + R - 26, cy + R - 4);

    // Per-feed markers — Γ = (Z − Z₀) / (Z + Z₀), normalised.
    const n = z_per_feed.length;
    z_per_feed.forEach((z, i) => {
      const zn_r = z.re / z0;
      const zn_x = z.im / z0;
      const denomR = (zn_r + 1) * (zn_r + 1) + zn_x * zn_x;
      const gammaR = ((zn_r - 1) * (zn_r + 1) + zn_x * zn_x) / denomR;
      const gammaI = (2 * zn_x) / denomR;
      const px = cx + gammaR * R;
      const py = cy - gammaI * R;
      ctx.fillStyle = feedColor(i, n);
      ctx.beginPath();
      ctx.arc(px, py, 6, 0, 2 * Math.PI);
      ctx.fill();
      ctx.strokeStyle = "#0d1015";
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.fillStyle = "#cfd6e3";
      ctx.font = "10px ui-monospace, monospace";
      ctx.fillText(`f${i}`, px + 10, py + 4);
    });
  }, [z_per_feed, z0, size]);

  return <canvas ref={canvasRef} />;
}

// ---------------------------------------------------------------------------
// Far-field polar plots — azimuth cut at chosen θ + elevation cut at φ.
// ---------------------------------------------------------------------------

function FarFieldPlot({
  ff,
  size = 480,
  theta0Deg = 90, // azimuth cut elevation: 90° = horizon
  phi0Deg = 0,    // elevation cut bearing
}: {
  ff: FarField;
  size?: number;
  theta0Deg?: number;
  phi0Deg?: number;
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

    const cx = size / 2;
    const cy = size / 2;
    const R = size / 2 - 18;
    const DBI_TOP = Math.ceil(ff.max_gain / 3) * 3 + 3;
    const DB_SPAN = 30;
    const dbToFrac = (db: number) => Math.max(0, (db - (DBI_TOP - DB_SPAN)) / DB_SPAN);

    // Radial grid: 6 dB steps from outer to centre.
    ctx.strokeStyle = "#2a313d";
    ctx.lineWidth = 0.6;
    ctx.fillStyle = "#4a5160";
    ctx.font = "9px ui-monospace, monospace";
    for (let d = 0; d <= 5; d++) {
      const rad = (R * (5 - d)) / 5;
      ctx.beginPath();
      ctx.arc(cx, cy, rad, 0, 2 * Math.PI);
      ctx.stroke();
      const lbl = `${(DBI_TOP - d * 6).toFixed(0)} dBi`;
      if (d > 0 && d < 5) ctx.fillText(lbl, cx + 3, cy - rad + 10);
    }
    // Angle spokes every 30°
    ctx.strokeStyle = "#252a35";
    for (let a = 0; a < 360; a += 30) {
      const ar = (a * Math.PI) / 180;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + Math.cos(ar) * R, cy - Math.sin(ar) * R);
      ctx.stroke();
    }

    // Azimuth cut: find nearest theta in ff.thetas to theta0Deg, plot all phi.
    const thetaIdx = nearestIdx(ff.thetas, theta0Deg);
    if (thetaIdx >= 0) {
      ctx.strokeStyle = "hsla(200, 80%, 60%, 0.95)";
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      for (let pi = 0; pi < ff.phis.length; pi++) {
        const phi = (ff.phis[pi] * Math.PI) / 180;
        const dbi = ff.rings[thetaIdx][pi];
        const rad = dbToFrac(dbi) * R;
        const x = cx + Math.cos(phi) * rad;
        const y = cy - Math.sin(phi) * rad;
        if (pi === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.stroke();
    }

    // Elevation cut: phi=phi0 and phi=phi0+180, plot dbi vs theta from -90..+90.
    const phiF = nearestIdx(ff.phis, phi0Deg);
    const phiB = nearestIdx(ff.phis, (phi0Deg + 180) % 360);
    if (phiF >= 0 && phiB >= 0) {
      ctx.strokeStyle = "hsla(40, 90%, 60%, 0.95)";
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      // Sweep forward bearing from theta=0 (zenith) down to theta=90 (horizon)
      // → map to angles 90°…0° on the chart. Then back bearing from 90 to 0°
      // → map to 180°…270°.
      for (let ti = 0; ti < ff.thetas.length; ti++) {
        const theta = ff.thetas[ti];
        const dbi = ff.rings[ti][phiF];
        const ang = ((90 - theta) * Math.PI) / 180;
        const rad = dbToFrac(dbi) * R;
        const x = cx + Math.cos(ang) * rad;
        const y = cy - Math.sin(ang) * rad;
        if (ti === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      for (let ti = ff.thetas.length - 1; ti >= 0; ti--) {
        const theta = ff.thetas[ti];
        const dbi = ff.rings[ti][phiB];
        const ang = ((90 + theta) * Math.PI) / 180; // back hemisphere
        const rad = dbToFrac(dbi) * R;
        ctx.lineTo(cx + Math.cos(ang) * rad, cy - Math.sin(ang) * rad);
      }
      ctx.stroke();
    }

    // Compass labels (azimuth cut → +x is φ=0, +y is φ=90, etc.)
    ctx.fillStyle = "#cfd6e3";
    ctx.font = "11px ui-monospace, monospace";
    ctx.fillText("φ=0°", cx + R - 30, cy + 14);
    ctx.fillText("φ=90°", cx + 6, cy - R + 14);
    ctx.fillText("φ=180°", cx - R + 4, cy + 14);
    ctx.fillText("φ=270°", cx + 6, cy + R - 4);

    // Legend
    ctx.fillStyle = "hsla(200, 80%, 60%, 0.95)";
    ctx.fillRect(8, size - 24, 12, 2);
    ctx.fillStyle = "#cfd6e3";
    ctx.fillText(`azimuth @ θ=${ff.thetas[thetaIdx]?.toFixed(0)}°`, 26, size - 20);
    ctx.fillStyle = "hsla(40, 90%, 60%, 0.95)";
    ctx.fillRect(8, size - 10, 12, 2);
    ctx.fillStyle = "#cfd6e3";
    ctx.fillText(`elevation @ φ=${phi0Deg}°/${(phi0Deg + 180) % 360}°`, 26, size - 6);
  }, [ff, size, theta0Deg, phi0Deg]);

  return <canvas ref={canvasRef} />;
}

function nearestIdx(arr: number[], target: number): number {
  let best = -1;
  let bestD = Infinity;
  for (let i = 0; i < arr.length; i++) {
    const d = Math.abs(arr[i] - target);
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  }
  return best;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function fmtC(c: ComplexVal): string {
  const sign = c.im >= 0 ? "+" : "-";
  return `${c.re.toFixed(2)} ${sign} ${Math.abs(c.im).toFixed(2)}j`;
}

function defaultValuesFor(schema: BuilderSchema): Record<string, number | ComplexVal> {
  const out: Record<string, number | ComplexVal> = {};
  for (const p of schema.params) {
    out[p.key] = p.default;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Top-level app.
// ---------------------------------------------------------------------------

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
  const [view, setView] = useState<"wire" | "smith" | "farfield">("wire");
  const [z0, setZ0] = useState<number>(50);

  // 1. On mount: fetch the builder index.
  useEffect(() => {
    fetch("/api/builders")
      .then((r) => {
        if (!r.ok) throw new Error(`/api/builders → ${r.status}`);
        return r.json();
      })
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

  // 2. When the selected builder or variant changes, refetch its schema so
  //    we get the right param defaults for the variant.
  const schema = useMemo(
    () => builders?.find((b) => b.name === selectedName) ?? null,
    [builders, selectedName],
  );

  useEffect(() => {
    if (!selectedName) return;
    const url = `/api/builder/${selectedName}?variant=${encodeURIComponent(variant)}`;
    fetch(url)
      .then((r) => {
        if (!r.ok) throw new Error(`${url} → ${r.status}`);
        return r.json();
      })
      .then((s: BuilderSchema) => {
        setValues(defaultValuesFor(s));
        // Patch the cached entry so the slider list reflects the variant's params.
        setBuilders((prev) =>
          prev ? prev.map((b) => (b.name === s.name ? { ...b, params: s.params } : b)) : prev,
        );
      })
      .catch((e: Error) => setError(e.message));
  }, [selectedName, variant]);

  const onParamChange = useCallback((key: string, val: number | ComplexVal) => {
    setValues((v) => ({ ...v, [key]: val }));
  }, []);

  const solve = useCallback(async () => {
    if (!schema) return;
    const engineChoice = ENGINE_CHOICES.find((c) => c.id === engineId);
    if (!engineChoice) return;
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
    setError(null);
    try {
      const r = await fetch("/api/solve", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(req),
      });
      if (!r.ok) {
        const text = await r.text();
        throw new Error(`${r.status} ${text}`);
      }
      setResult(await r.json());
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSolving(false);
    }
  }, [schema, selectedName, variant, values, engineId, ground, farField]);

  if (error && !builders) {
    return (
      <div className="app">
        <div className="panel">
          <div className="error">{error}</div>
        </div>
      </div>
    );
  }
  if (!builders) {
    return (
      <div className="app">
        <div className="panel">Loading builders…</div>
      </div>
    );
  }

  return (
    <div className="app">
      <div className="panel">
        <h2>Antenna</h2>
        <div className="field">
          <label>Builder</label>
          <select
            value={selectedName}
            onChange={(e) => {
              setSelectedName(e.target.value);
              setVariant("default");
            }}
          >
            {builders.map((b) => (
              <option key={b.name} value={b.name}>
                {b.name}
              </option>
            ))}
          </select>
        </div>
        {schema && schema.variants.length > 1 && (
          <div className="field">
            <label>Variant</label>
            <select value={variant} onChange={(e) => setVariant(e.target.value)}>
              {schema.variants.map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </div>
        )}

        <h2>Engine</h2>
        <div className="field">
          <label>Solver</label>
          <select value={engineId} onChange={(e) => setEngineId(e.target.value)}>
            {ENGINE_CHOICES.map((c) => (
              <option key={c.id} value={c.id}>
                {c.label}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label>Ground</label>
          <select value={ground} onChange={(e) => setGround(e.target.value)}>
            {GROUND_CHOICES.map((c) => (
              <option key={c.id} value={c.id}>
                {c.label}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label>
            <input
              type="checkbox"
              checked={farField}
              onChange={(e) => setFarField(e.target.checked)}
            />{" "}
            Compute far-field
          </label>
        </div>

        <h2>Parameters</h2>
        {schema && (
          <ParamSliders params={schema.params} values={values} onChange={onParamChange} />
        )}

        <button className="solve-button" onClick={solve} disabled={solving}>
          {solving ? "Solving…" : "Solve"}
        </button>

        {error && <div className="error" style={{ marginTop: 8 }}>{error}</div>}
      </div>

      <div className="panel viewport">
        <div className="tab-row">
          <button
            className={view === "wire" ? "tab active" : "tab"}
            onClick={() => setView("wire")}
          >
            Wire + currents
          </button>
          <button
            className={view === "smith" ? "tab active" : "tab"}
            onClick={() => setView("smith")}
          >
            Smith
          </button>
          <button
            className={view === "farfield" ? "tab active" : "tab"}
            onClick={() => setView("farfield")}
            disabled={!result?.far_field}
          >
            Far-field
          </button>
        </div>
        {result ? (
          <>
            <div className="status">
              {result.builder} ({result.variant}) · {result.engine} · {result.freq_mhz} MHz · {result.wires.length} wires
              {result.far_field && (
                <>
                  {" · "}far-field {result.far_field.max_gain.toFixed(2)} dBi max
                </>
              )}
            </div>
            <table className="z-table">
              <thead>
                <tr>
                  <th>Feed</th>
                  <th>R (Ω)</th>
                  <th>X (Ω)</th>
                  <th>|Z|</th>
                </tr>
              </thead>
              <tbody>
                {result.z_per_feed.map((z, i) => (
                  <tr key={i}>
                    <td>
                      <span
                        className="feed-swatch"
                        style={{
                          background: feedColor(i, result.z_per_feed.length),
                        }}
                      />
                      f{i}
                    </td>
                    <td>{z.re.toFixed(2)}</td>
                    <td>{z.im.toFixed(2)}</td>
                    <td>{fmtC(z)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {view === "wire" && (
              <WireCanvas wires={result.wires} currents={result.currents} />
            )}
            {view === "smith" && (
              <>
                <div className="field" style={{ flexDirection: "row", alignItems: "center", gap: 8 }}>
                  <label style={{ margin: 0 }}>Z₀ (Ω)</label>
                  <input
                    type="number"
                    value={z0}
                    step={1}
                    onChange={(e) => setZ0(parseFloat(e.target.value) || 50)}
                    style={{ width: 80 }}
                  />
                </div>
                <SmithChart z_per_feed={result.z_per_feed} z0={z0} size={480} />
              </>
            )}
            {view === "farfield" && result.far_field && (
              <FarFieldPlot ff={result.far_field} size={480} />
            )}
          </>
        ) : (
          <div className="status">Hit Solve to compute.</div>
        )}
      </div>
    </div>
  );
}
