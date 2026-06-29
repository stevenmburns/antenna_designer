# Cloudflare DNS setup for the Fly.io apps

A runbook for pointing the `antennaknobs` domains at the Fly apps via
Cloudflare-managed DNS, and validating the Fly TLS certs. This documents the
**as-built** setup: both zones mirror each other across two Fly apps.

There are two Fly apps:

| App | What | Fly hostname |
| --- | --- | --- |
| `antennaknobs` | the live simulator (FastAPI) | `antennaknobs.fly.dev` |
| `antennaknobs-docs` | the docs site (static, nginx) | `antennaknobs-docs.fly.dev` |

## As-built layout

Two zones, `antennaknobs.dev` and `antennaknobs.com`, are configured **identically
(mirrored)**:

| Hostname | Serves | Fly app |
| --- | --- | --- |
| `antennaknobs.dev` (apex) | docs site | `antennaknobs-docs` |
| `app.antennaknobs.dev` | simulator | `antennaknobs` |
| `antennaknobs.com` (apex) | docs site | `antennaknobs-docs` |
| `app.antennaknobs.com` | simulator | `antennaknobs` |

The apex serves the docs; the simulator lives on the `app.` subdomain. `.com`
is a straight mirror of `.dev`.

## 0. Prerequisites

- Both zones (`antennaknobs.dev`, `antennaknobs.com`) must be **active on
  Cloudflare** — i.e. the registrar's nameservers point at Cloudflare. Check in
  the Cloudflare dashboard that each zone shows "Active". If not, add the site to
  Cloudflare and update the nameservers at the registrar first.
- `flyctl` installed and logged in (`flyctl auth whoami`).

## 1. Get the current Fly app IPs

Fly IPs rarely change, but always confirm before editing DNS:

```bash
flyctl ips list -a antennaknobs-docs   # docs site (serves the apexes)
flyctl ips list -a antennaknobs        # simulator (serves app.*)
```

As of last setup:

| App | A (shared IPv4) | AAAA (dedicated IPv6) |
| --- | --- | --- |
| `antennaknobs-docs` | `66.241.124.199` | `2a09:8280:1::137:a070:0` |
| `antennaknobs` | `66.241.125.41` | `2a09:8280:1::137:614b:0` |

## 2. Add the Fly cert(s)

Tell Fly which hostnames each app should serve, so it provisions a TLS cert. The
as-built set is four hostnames across the two apps:

```bash
# Apexes -> docs app
flyctl certs add antennaknobs.dev -a antennaknobs-docs
flyctl certs add antennaknobs.com -a antennaknobs-docs

# app.* subdomains -> simulator
flyctl certs add app.antennaknobs.dev -a antennaknobs
flyctl certs add app.antennaknobs.com -a antennaknobs

# optional www, if you want it:
# flyctl certs add www.antennaknobs.dev -a antennaknobs-docs
# flyctl certs add www.antennaknobs.com -a antennaknobs-docs
```

`flyctl certs add` prints the exact A/AAAA records Fly wants (they match the
table in §3). `flyctl certs show <hostname> -a <app>` reprints them later.

## 3. Create the records in Cloudflare

> [!IMPORTANT]
> Set every Fly record to **DNS only (grey cloud), NOT proxied**. Fly does its
> own TLS/SNI routing, and a proxied (orange-cloud) record resolves to
> Cloudflare's IPs — which breaks Fly's ACME cert validation and its routing.
> Keep them grey-cloud.

### Records (per zone — create the same four in **both** `.dev` and `.com`)

| Type | Name | Value | Proxy |
| --- | --- | --- | --- |
| `A` | `@` (apex) | `66.241.124.199` | DNS only |
| `AAAA` | `@` (apex) | `2a09:8280:1::137:a070:0` | DNS only |
| `A` | `app` | `66.241.125.41` | DNS only |
| `AAAA` | `app` | `2a09:8280:1::137:614b:0` | DNS only |

`@` is the bare apex (Cloudflare auto-expands it); `app` becomes
`app.<zone>`. That's 8 records total across the two zones.

Optional `www` (point at whichever app you certified it on):

| Type | Name | Value | Proxy |
| --- | --- | --- | --- |
| `CNAME` | `www` | `antennaknobs-docs.fly.dev` | DNS only |

### Option A — Cloudflare dashboard (manual)

For each zone: **DNS** → **Records** → **Add record**, one per row above. For the
apex, set Name to `@`. Toggle the cloud icon to **grey (DNS only)** before
saving. Repeat for the second zone.

### Option B — Cloudflare API (scripted)

Create a **scoped** API token: Cloudflare → My Profile → **API Tokens** →
Create Token → **Edit zone DNS** template → Permissions `Zone:DNS:Edit`
(+ `Zone:Zone:Read` for the zone-id lookup), Zone Resources = **both** zones.
Then, per zone:

```bash
# Provide the token WITHOUT putting it in shell history / the terminal scrollback:
#   umask 077 && printf '%s' 'YOUR_TOKEN' > ~/.cf-token
export CF_TOKEN="$(cat ~/.cf-token)"

for ZONE in antennaknobs.dev antennaknobs.com; do
  # Look up the zone id
  ZID=$(curl -s -H "Authorization: Bearer $CF_TOKEN" \
    "https://api.cloudflare.com/client/v4/zones?name=$ZONE" | \
    python3 -c 'import sys,json;print(json.load(sys.stdin)["result"][0]["id"])')

  # apex -> docs app; app.* -> simulator; all DNS only (proxied=false)
  for rec in \
    "A    @   66.241.124.199" \
    "AAAA @   2a09:8280:1::137:a070:0" \
    "A    app 66.241.125.41" \
    "AAAA app 2a09:8280:1::137:614b:0"; do
    set -- $rec
    TYPE=$1; HOST=$2; VAL=$3
    NAME=$([ "$HOST" = "@" ] && echo "$ZONE" || echo "$HOST.$ZONE")
    curl -s -X POST -H "Authorization: Bearer $CF_TOKEN" -H "Content-Type: application/json" \
      "https://api.cloudflare.com/client/v4/zones/$ZID/dns_records" \
      --data "{\"type\":\"$TYPE\",\"name\":\"$NAME\",\"content\":\"$VAL\",\"ttl\":1,\"proxied\":false}"
    echo
  done
done

rm -f ~/.cf-token   # clean up the token when done
```

## 4. Validate

Certs flip to `Issued` once Fly sees the DNS:

```bash
flyctl certs list -a antennaknobs-docs   # antennaknobs.dev, antennaknobs.com
flyctl certs list -a antennaknobs        # app.antennaknobs.dev, app.antennaknobs.com
```

Re-run until each reports `Issued` (DNS propagation is usually seconds–minutes on
Cloudflare). Then confirm each hostname serves a 200 over HTTPS:

```bash
for u in https://antennaknobs.dev/ https://app.antennaknobs.dev/ \
         https://antennaknobs.com/ https://app.antennaknobs.com/; do
  echo -n "$u -> "; curl -sS -o /dev/null -w "%{http_code}\n" "$u"
done
```

All four should return `200`, the apexes routing to `66.241.124.199` (docs) and
the `app.*` hosts to `66.241.125.41` (simulator).

## 5. After DNS is live

- Update the docs' simulator link if you move the simulator's public hostname: it's
  the `SIMULATOR_URL` constant in `site/astro.config.mjs` (plus the literal links in
  the home/welcome/web/catalog pages). A push to `main` redeploys the docs via
  the `deploy-docs.yml` Action.

## Notes / gotchas

- **Apex + shared IPv4:** both apexes use the docs app's *shared* Fly IPv4, which
  routes HTTPS fine via SNI, but plain `http://` on the apex (port 80) can't be
  routed by SNI and won't redirect cleanly. For flawless apex redirects, allocate
  a dedicated IPv4: `flyctl ips allocate-v4 -a antennaknobs-docs` (~$2/mo) and use
  that as the apex `A` record in both zones.
- Keep records **grey-cloud**. If you later want Cloudflare's proxy/CDN in front,
  set the zone's SSL/TLS mode to **Full (strict)** first, and expect to re-check
  the Fly cert.
- IPs: re-fetch with `flyctl ips list -a <app>` if a record ever stops resolving
  to the right place.
