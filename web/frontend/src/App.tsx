import { useCallback, useEffect, useMemo, useState } from "react";

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

type SolveResponse = {
  builder: string;
  variant: string;
  engine: string;
  freq_mhz: number;
  z_per_feed: ComplexVal[];
  wires: WireGeom[];
  currents: unknown[];
  far_field: { max_gain: number; min_gain: number } | null;
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

function WireCanvas({ wires, size = 480 }: { wires: WireGeom[]; size?: number }) {
  if (wires.length === 0) {
    return <canvas width={size} height={size} />;
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
  // Letterbox into the square, preserving aspect ratio.
  const scale = Math.min((size * (1 - 2 * pad)) / dx, (size * (1 - 2 * pad)) / dy);
  const ox = (size - scale * dx) / 2 - scale * xmin;
  const oy = (size - scale * dy) / 2 - scale * ymin;
  const tx = (x: number) => ox + scale * x;
  const ty = (y: number) => size - (oy + scale * y); // flip y for screen coords

  return (
    <svg width={size} height={size}>
      <rect x={0} y={0} width={size} height={size} fill="var(--panel-2)" />
      {/* Axis hint */}
      <text x={6} y={size - 6} fill="var(--text-dim)" fontSize={11} fontFamily="monospace">
        {`${proj} projection — ${dx.toFixed(2)} × ${dy.toFixed(2)} m`}
      </text>
      {wires.map((w, i) => {
        const [[a, b], [c, d]] = proj2[i];
        const isFeed = w.feed_voltage !== null;
        return (
          <line
            key={i}
            x1={tx(a)}
            y1={ty(b)}
            x2={tx(c)}
            y2={ty(d)}
            stroke={isFeed ? "var(--feed)" : "var(--wire)"}
            strokeWidth={isFeed ? 2.5 : 1.5}
          />
        );
      })}
    </svg>
  );
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
        <h2>Result</h2>
        {result ? (
          <>
            <div className="status">
              {result.builder} ({result.variant}) · {result.engine} · {result.freq_mhz} MHz · {result.wires.length} wires
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
                    <td>{i}</td>
                    <td>{z.re.toFixed(2)}</td>
                    <td>{z.im.toFixed(2)}</td>
                    <td>{fmtC(z)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {result.far_field && (
              <div className="status">
                Far-field: max {result.far_field.max_gain.toFixed(2)} dBi, min {result.far_field.min_gain.toFixed(2)} dBi
              </div>
            )}
            <WireCanvas wires={result.wires} />
          </>
        ) : (
          <div className="status">Hit Solve to compute.</div>
        )}
      </div>
    </div>
  );
}
