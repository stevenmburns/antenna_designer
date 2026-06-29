# Deploy runbook — the live simulator on Fly.io

This deploys the web workbench (`antennaknobs.web.server:app` — the API, the
`/ws` live-solve WebSocket, and the built React SPA) as a single persistent
container on [Fly.io](https://fly.io). It uses the prebuilt TestPyPI engine
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
`pynec-accel>=1.7.4.post1` resolve on TestPyPI for linux (the `pip install`
step in the `Dockerfile`).

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

## 4. Push-to-deploy via GitHub Actions (optional, recommended)

Fly has no "connect a repo" dashboard like Render/Vercel — but
[`.github/workflows/fly-deploy.yml`](../.github/workflows/fly-deploy.yml) gives
the same result: every push to `main` that touches the app or its deploy config
builds on Fly's remote builders and releases. One-time bootstrap (after the app
exists from step 2):

```bash
# A scoped deploy token for this app:
fly tokens create deploy -a antennaknobs   # use your app name
```

Copy the printed token and add it as a repo secret named **`FLY_API_TOKEN`**
(GitHub → repo Settings → Secrets and variables → Actions → New repository
secret). Until that secret exists, the workflow runs green but **skips** the
deploy with a notice — so it never fails CI before you're ready. Once set, a
merge to `main` deploys automatically; `workflow_dispatch` lets you trigger a
deploy by hand from the Actions tab.

The first deploy should still be the manual `fly deploy` from step 2 (it creates
and warms the machine); the workflow handles redeploys thereafter.

## 5. Day-to-day

| Task | Command |
|---|---|
| Redeploy after a change | `fly deploy` (or just push to `main`) |
| Tail logs | `fly logs` |
| Open a shell in the machine | `fly ssh console` |
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
