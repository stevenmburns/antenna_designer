# QRZ.com banner ads

Marketing assets for a [QRZ.com](https://www.qrz.com/page/advertising.html)
**top-slot leaderboard** campaign for AntennaKNoBs.

| File | What | Size |
| --- | --- | --- |
| `antennaknobs_728x90.gif` | Animated banner (3 knob-driven panels) | ~34 KB |
| `antennaknobs_728x90.png` | A frozen hero frame of the GIF (static fallback) | ~15 KB |

## QRZ top-slot spec (verify current values with sales@qrz.com)

- **Dimensions:** 728 × 90 px
- **Formats:** JPG, GIF, or PNG · **max file size 48 KB** (both assets are well under)
- **Rate (as read from QRZ's page, June 2026):** top slot **$200 / month**
  ($540 / 3 mo, $2040 / yr); 10% off for 3+ units, 15% for 1 yr+. Confirm before buying.

## What it shows

Three panels, left to right, all driven by the dial and all computed live by this
repo's `momwire` engine from the catalog `beams.moxon` design:

1. **A labelled dial** ("spacing") with a live readout below it — the Moxon's
   element spacing in wavelengths, ticking as the knob turns.
2. **The Moxon geometry** — its real top view (driven element + feed gap,
   reflector, bent tips) from `build_wires()`, reshaping as the spacing changes.
3. **The radiation pattern** — an azimuth cut from `far_field()`.

The dial sweeps `aspect_ratio` (element spacing): as it widens, the boom deepens
and the back lobe collapses into a clean forward beam — the actual Moxon
spacing/front-to-back tradeoff. Turn a knob, the antenna responds: the point of
AntennaKNoBs, drawn by AntennaKNoBs.

## Messaging (deliberate)

- **No URL** — the banner is clickable, so the address is omitted (it just looked
  cramped). The ad links to **antennaknobs.dev**, the main site, so visitors land
  on the installable open-source tool rather than the throwaway web demo.
- Says **"open source"** in the subline (no "free trial" framing that would imply
  a future paywall).
- Avoids the word **"tune"** — turning a design parameter is not antenna tuning.
- Headline **"Your antennas, as code."**; KK7KNB credit for QRZ cred.

## Typography

IBM Plex Sans + IBM Plex Mono — the same family the web app loads — with the
**numeric readout in Plex Mono**, matching the app's value displays. (Plex Mono
ships without a λ glyph, so the unit is drawn in Plex Sans.) The fonts are **not
committed** (OFL 1.1, ~1 MB); fetch them first:

```bash
docs/ad/fetch_fonts.sh        # downloads IBM Plex into docs/ad/fonts/ (gitignored)
```

## Regenerate

```bash
pip install -e ".[web]"        # needs antennaknobs (for the momwire patterns) + Pillow
docs/ad/fetch_fonts.sh
python docs/ad/generate_animated.py   # writes BOTH the .gif and the hero-frame .png
```

Rendered at 3× and downscaled (LANCZOS) for antialiasing; the GIF is delta-frame
encoded (only the changing left panels are stored per frame), which keeps it
under the 48 KB limit. The PNG is the widest-spacing frame (clean forward beam).
Override the font location with the `AD_FONTS` env var if your TTFs live elsewhere.
