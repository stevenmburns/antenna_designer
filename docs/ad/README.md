# QRZ.com banner ads

Marketing assets for a [QRZ.com](https://www.qrz.com/page/advertising.html)
**top-slot leaderboard** campaign for AntennaKNoBs.

| File | What | Size |
| --- | --- | --- |
| `antennaknobs_728x90.png` | Static banner | ~13 KB |
| `antennaknobs_728x90.gif` | Animated banner (dial drives a live antenna pattern) | ~34 KB |

## QRZ top-slot spec (verify current values with sales@qrz.com)

- **Dimensions:** 728 × 90 px
- **Formats:** JPG, GIF, or PNG · **max file size 48 KB** (both assets are well under)
- **Rate (as read from QRZ's page, June 2026):** top slot **$200 / month**
  ($540 / 3 mo, $2040 / yr); 10% off for 3+ units, 15% for 1 yr+. Confirm before buying.

## The animation is real

The GIF's morphing pattern is **not faked** — each frame is a genuine azimuth
cut of the catalog `beams.moxon` design, computed by this repo's `momwire`
engine (`MomwireEngine.far_field`) as the dial sweeps the Moxon's `t0_factor`.
The beam swings from a clean forward lobe (deep rear null) into a figure-8 and
back: the literal point of AntennaKNoBs — turn a knob, the antenna responds.

## Messaging (intentional)

- **Links to `antennaknobs.dev`** (the main site), so visitors land on the
  installable, open-source tool — not straight into the throwaway web demo.
- Says **"open source"** in the subline (no "free trial" framing that would
  imply a future paywall).
- Avoids the word **"tune"** — turning a design parameter is not antenna tuning.
- Headline **"Your antennas, as code."**; KK7KNB credit for QRZ cred.

## Typography

IBM Plex Sans + IBM Plex Mono — the same family the web app loads — with the
`antennaknobs.dev` URL set in Plex Mono to echo the app's monospace readouts.
The fonts are **not committed** (OFL 1.1, ~1 MB); fetch them first:

```bash
docs/ad/fetch_fonts.sh        # downloads IBM Plex into docs/ad/fonts/ (gitignored)
```

## Regenerate

```bash
pip install -e ".[web]"        # needs antennaknobs (for the momwire patterns) + Pillow
docs/ad/fetch_fonts.sh
python docs/ad/generate_static.py
python docs/ad/generate_animated.py
```

Both are rendered at 3× and downscaled (LANCZOS) for antialiasing. The GIF is
delta-frame encoded (only the dial+pattern region is stored per frame), which
is what keeps it under the 48 KB limit. Override the font location with the
`AD_FONTS` env var if your TTFs live elsewhere.
