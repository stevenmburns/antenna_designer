# Cloudflare DNS setup for the Fly.io apps

A runbook for pointing the `antennaknobs` domains at the Fly apps via
Cloudflare-managed DNS, and validating the Fly TLS certs.

There are two Fly apps:

| App | What | Public hostname |
| --- | --- | --- |
| `antennaknobs` | the live tuner (FastAPI) | `antennaknobs.fly.dev` |
| `antennaknobs-docs` | the docs site (static, nginx) | `antennaknobs-docs.fly.dev` |

## 0. Prerequisites

- The zone (`antennaknobs.dev` / `antennaknobs.com`) must already be **active on
  Cloudflare** — i.e. the registrar's nameservers point at Cloudflare. Check in
  the Cloudflare dashboard that the zone shows "Active". If not, add the site to
  Cloudflare and update the nameservers at the registrar first.
- `flyctl` installed and logged in (`flyctl auth whoami`).

## 1. Get the current Fly app IPs

Fly IPs rarely change, but always confirm before editing DNS:

```bash
flyctl ips list -a antennaknobs-docs   # docs site
flyctl ips list -a antennaknobs        # tuner
```

As of last setup:

| App | A (shared IPv4) | AAAA (dedicated IPv6) |
| --- | --- | --- |
| `antennaknobs-docs` | `66.241.124.199` | `2a09:8280:1::137:a070:0` |
| `antennaknobs` | `66.241.125.41` | `2a09:8280:1::137:614b:0` |

## 2. Add the Fly cert(s)

Tell Fly which hostnames it should serve, so it provisions a TLS cert:

```bash
flyctl certs add antennaknobs.dev -a antennaknobs-docs
# optional extras:
flyctl certs add www.antennaknobs.dev -a antennaknobs-docs
flyctl certs add app.antennaknobs.dev -a antennaknobs       # tuner on a subdomain
```

`flyctl certs show <hostname> -a <app>` prints the exact records Fly recommends.

## 3. Create the records in Cloudflare

> [!IMPORTANT]
> Set every Fly record to **DNS only (grey cloud), NOT proxied**. Fly does its
> own TLS/SNI routing, and a proxied (orange-cloud) record resolves to
> Cloudflare's IPs — which breaks Fly's ACME cert validation and its routing.
> Keep them grey-cloud (at minimum until the cert is validated).

### Recommended records

| Type | Name | Value | Proxy |
| --- | --- | --- | --- |
| `A` | `antennaknobs.dev` (`@`) | `66.241.124.199` | DNS only |
| `AAAA` | `antennaknobs.dev` (`@`) | `2a09:8280:1::137:a070:0` | DNS only |

Optional:

| Type | Name | Value | Proxy |
| --- | --- | --- | --- |
| `CNAME` | `www` | `antennaknobs-docs.fly.dev` | DNS only |
| `CNAME` | `app` | `antennaknobs.fly.dev` | DNS only |

(For `antennaknobs.com`, repeat the apex `A`/`AAAA` pointing at whichever app
should serve it — docs for now, or the marketing site later.)

### Option A — Cloudflare dashboard (manual)

Zone → **DNS** → **Records** → **Add record**, one per row above. For the apex,
set Name to `@`. Toggle the cloud icon to **grey (DNS only)**.

### Option B — Cloudflare API (scripted)

Create a **scoped** API token: Cloudflare → My Profile → **API Tokens** →
Create Token → **Edit zone DNS** template → Permissions `Zone:DNS:Edit`, Zone
Resources = the specific zone(s). Then:

```bash
# Provide the token WITHOUT putting it in shell history / the terminal scrollback:
#   umask 077 && printf '%s' 'YOUR_TOKEN' > ~/.cf-token
export CF_TOKEN="$(cat ~/.cf-token)"
ZONE="antennaknobs.dev"

# Look up the zone id
ZID=$(curl -s -H "Authorization: Bearer $CF_TOKEN" \
  "https://api.cloudflare.com/client/v4/zones?name=$ZONE" | \
  python3 -c 'import sys,json;print(json.load(sys.stdin)["result"][0]["id"])')

# Apex A + AAAA -> the docs app, DNS only (proxied=false)
for rec in 'A 66.241.124.199' 'AAAA 2a09:8280:1::137:a070:0'; do
  set -- $rec
  curl -s -X POST -H "Authorization: Bearer $CF_TOKEN" -H "Content-Type: application/json" \
    "https://api.cloudflare.com/client/v4/zones/$ZID/dns_records" \
    --data "{\"type\":\"$1\",\"name\":\"$ZONE\",\"content\":\"$2\",\"ttl\":1,\"proxied\":false}"
  echo
done

rm -f ~/.cf-token   # clean up the token when done
```

## 4. Validate

```bash
flyctl certs check antennaknobs.dev -a antennaknobs-docs
```

Re-run until it reports verified (DNS propagation is usually seconds–minutes on
Cloudflare). Then:

```bash
curl -sS -o /dev/null -w "%{http_code}\n" https://antennaknobs.dev/
```

should return `200`.

## 5. After DNS is live

- Update the docs' tuner link if you move the tuner onto a custom domain: it's
  the `TUNER_URL` constant in `site/astro.config.mjs` (plus the literal links in
  the home/welcome/web/catalog pages). A push to `main` redeploys the docs via
  the `deploy-docs.yml` Action.

## Notes / gotchas

- **Apex + shared IPv4:** the docs app uses a *shared* Fly IPv4, which routes
  HTTPS fine via SNI, but plain `http://` on the apex (port 80) can't be routed
  by SNI and won't redirect cleanly. For flawless apex redirects, allocate a
  dedicated IPv4: `flyctl ips allocate-v4 -a antennaknobs-docs` (~$2/mo) and use
  that as the `A` record.
- Keep records **grey-cloud**. If you later want Cloudflare's proxy/CDN in front,
  set the zone's SSL/TLS mode to **Full (strict)** first, and expect to re-check
  the Fly cert.
- IPs: re-fetch with `flyctl ips list -a <app>` if a record ever stops resolving
  to the right place.
