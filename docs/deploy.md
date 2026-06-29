# Deploy runbook — the live simulator on Fly.io

This deploys the web workbench (`antennaknobs.web.server:app` — the API, the
`/ws` live-solve WebSocket, and the built React SPA) as a single persistent
container on [Fly.io](https://fly.io). It uses the prebuilt PyPI engine
wheels, so the image needs **no C++ toolchain**.

Files involved: [`Dockerfile`](../Dockerfile), [`.dockerignore`](../.dockerignore),
[`fly.toml`](../fly.toml).

## 0. One-time prerequisites

- Docker (to build/test the image locally).
- The Fly CLI: `curl -L https://fly.io/install.sh | sh` (or `brew install flyctl`).
- A Fly account + login. Run this in your shell so the browser auth completes:

  ```
  flyctl auth login
  ```

## 1. Build and test the image locally first

Confirm the container serves the app before touching Fly:

```bash
docker build -t antennaknobs-web .
docker run --rm -p 8000:8000 antennaknobs-web
# open http://127.0.0.1:8000 — the simulator should load and knobs should solve
curl -fsS http://127.0.0.1:8000/healthz   # -> ok
```

If the build fails on the engine wheels, check that `momwire` and
`pynec-accel>=1.7.4.post1` resolve on PyPI for linux (the `pip install`
steps in the `Dockerfile` — momwire with the core, pynec-accel as its own
optional GPL-2.0 step).

## 2. First deploy

`fly.toml` is already written, so you only need an app name and a region.

```bash
# If the name "antennaknobs" in fly.toml is taken, create your own and edit
# the `app =` line to match:
fly apps create antennaknobs        # or your chosen name

# Pick the region nearest you (latency is dominated by the solve, but a short
# network hop still helps — see the latency note below):
fly platform regions                # list; edit primary_region in fly.toml

fly deploy
```

`fly deploy` builds the `Dockerfile`, pushes the image, and boots one machine.
When it finishes:

```bash
fly status                          # machine should be "started" + healthy
fly open                            # opens https://<app>.fly.dev
fly logs                            # tail if anything misbehaves
```

At this point the live simulator is reachable at `https://<app>.fly.dev`. **Test it
remotely from your own machine and confirm dragging a knob feels responsive
before adding the custom domains** — this is the moment to measure the real
round-trip.

## 3. Custom domains (after the app is verified live)

For each of `antennaknobs.com` and `antennaknobs.dev` (or a subdomain like
`app.antennaknobs.dev`):

```bash
fly certs add app.antennaknobs.dev
fly certs show app.antennaknobs.dev   # prints the A/AAAA (or CNAME) target
```

Add the printed record at your registrar's DNS, then re-run `fly certs show`
until the cert validates. (The marketing `.com` landing and `.dev` docs are
static sites built separately — see `docs/website-content-plan.md`; only the
live app needs this Fly service.)

## 4. Tag-to-deploy via GitHub Actions (the release discipline)

`main` is the always-green **integration** line: PRs merge into it freely and CI
runs, but **merging does not deploy anything**. A release is a deliberate act —
push a `v*` tag, and three workflows fire off that one tagged commit:

| Workflow | App / target | Secret |
|---|---|---|
| [`fly-deploy.yml`](../.github/workflows/fly-deploy.yml) | simulator (`antennaknobs`) | `FLY_API_TOKEN_SIMULATOR` |
| [`deploy-docs.yml`](../.github/workflows/deploy-docs.yml) | docs site (`antennaknobs-docs`) | `FLY_API_TOKEN_DOCS` |
| [`publish.yml`](../.github/workflows/publish.yml) | the package → PyPI + GitHub Release | — (Trusted Publishing) |

So the tagged version number always names exactly what's live across the package,
the simulator, and the docs — they ship as one consistent snapshot.

One-time bootstrap (after each app exists from step 2): mint an **app-scoped**
deploy token per app and store each as the matching repo secret (GitHub → repo
Settings → Secrets and variables → Actions):

```bash
fly tokens create deploy -a antennaknobs        # -> FLY_API_TOKEN_SIMULATOR
fly tokens create deploy -a antennaknobs-docs    # -> FLY_API_TOKEN_DOCS
```

Until a secret exists, its deploy job runs green but **skips** with a notice — so
a release never fails CI before the token is set. The first deploy of a brand-new
app should still be the manual `fly deploy` from step 2 (it creates and warms the
machine); the tag workflow handles releases thereafter.

### Cutting a release

```bash
# 1. main is green and has everything you want to ship.
git checkout main && git pull

# 2. Bump the version (single source of truth) and commit it on main via a PR.
#    Edit `version = "..."` in pyproject.toml. The tag must match this.

# 3. Tag the release commit and push the tag — this is what deploys.
git tag v0.5.0
git push origin v0.5.0
```

Watch the three workflows in the Actions tab; when green, the simulator
(`app.antennaknobs.dev`), docs (`antennaknobs.dev`), and the published dist are
all at `v0.5.0`. **Rollback / hotfix:** every deploy workflow keeps a
`workflow_dispatch` button — run it from the Actions tab against any ref (an older
tag to roll back, or a fix branch) without cutting a new version.

## 5. Day-to-day

| Task | Command |
|---|---|
| Ship a release (sim + docs + dist) | bump `pyproject.toml`, then `git tag vX.Y.Z && git push origin vX.Y.Z` |
| Deploy a hotfix / roll back, no tag | Actions tab → the deploy workflow → **Run workflow** on the chosen ref |
| Manual one-off deploy from your machine | `fly deploy` (sim) / `cd site && fly deploy` (docs) |
| Tail logs | `fly logs -a antennaknobs` |
| Open a shell in the machine | `fly ssh console -a antennaknobs` |
| Scale memory / CPU | edit `[[vm]]` in `fly.toml`, then `fly deploy` |
| Add a region replica | `fly scale count 2 --region <r>` |

## Latency note

Perceived lag per knob turn ≈ one network round-trip + the server solve time.
The **solve dominates** (free-space dipole-class solves are ~10–80 ms; real-ground
Sommerfeld is ~100× slower). A regional Fly machine adds only ~5–40 ms RTT over a
persistent WebSocket, so typical tuning totals well under the ~100 ms "feels
instant" threshold. If it ever feels sluggish, the levers are: debounce knob
events client-side, lean on the existing solve cache, default to the fast ground
approximation while dragging, and keep `min_machines_running = 1` so there's no
cold start.

## Live-engine size cap (hosted only)

A solve builds a method-of-moments system whose dimension N ≈ the total wire
segment count. The dense solvers — and PyNEC — form an N×N complex matrix
(memory N²·16 bytes), so an unbounded N — a hand-edited request cranking
"segments / wire", or a very large array — could exhaust a small box's RAM.

This guard is **off by default**, so the package people `pip install` and run
locally is **unlocked** (solve as big as your own machine allows). It turns on
**only** when **`ANTENNAKNOBS_HOSTED=1`** is set — which this repo's `fly.toml`
`[env]` does for the shared instance. Same wheel, unlocked locally and capped
online. When on, the server rejects an oversized solve *before* the matrix is
allocated, with a clear error in the UI.

The caps target a single solve's matrix staying under **~800 MB** on the 2 GB
Fly box (`basis = √(800·2²⁰/16) ≈ 7000` for a dense N×N). They're about
**memory, not time** (PyNEC's ~N³ LU is slow long before it's large — a
responsiveness concern, deliberately not guarded). The numbers come from
measuring `arrays.bowtiearray2x4` (see `scripts/measure_solve_memory.py`): PyNEC
tracks the full dense N×N (~1 GB at basis 8000); `arrayblock`'s block-low-rank
uses ~0.6× of that, so it gets a proportionally higher cap. Override any of them
in `fly.toml`'s `[env]` if the VM grows:

| Env var | Default | Applies to |
|---|---|---|
| `ANTENNAKNOBS_HOSTED` | *(unset → off)* | **master switch** — set to `1` to enforce the caps below |
| `ANTENNAKNOBS_MAX_BASIS` | `7000` | dense momwire (triangular / sinusoidal / bspline) — full N×N |
| `ANTENNAKNOBS_MAX_BASIS_COMPRESSED` | `9000` | `arrayblock` / `hmatrix` (block-low-rank, ~0.6× dense memory) |
| `ANTENNAKNOBS_MAX_BASIS_PYNEC` | `7000` | PyNEC (full dense N×N, same as dense momwire) |
