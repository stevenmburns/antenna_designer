"""AntennaKNoBs QRZ leaderboard banner — static 728x90 PNG (direction 1,
"Your antennas, as code."). IBM Plex Sans + Mono to match the app's typography.

Run from a venv with the package installed (only PIL is strictly needed here):
    python docs/ad/generate_static.py

Fonts: run ./fetch_fonts.sh first (downloads IBM Plex into docs/ad/fonts/), or
point AD_FONTS at a directory holding the IBMPlex*.ttf files.
"""

import os
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONTS = (
    next(
        (
            p
            for p in (
                os.environ.get("AD_FONTS"),
                os.path.join(HERE, "fonts"),
                "/tmp/fonts",
            )
            if p and os.path.isdir(p)
        ),
        os.path.join(HERE, "fonts"),
    )
    + "/"
)
OUT = os.path.join(HERE, "antennaknobs_728x90.png")

SANS = "IBMPlexSans-Regular.ttf"
SEMI = "IBMPlexSans-SemiBold.ttf"
MONO = "IBMPlexMono-SemiBold.ttf"

S = 3
W, H = 728 * S, 90 * S


def px(v):
    return int(round(v * S))


def font(name, p):
    return ImageFont.truetype(FONTS + name, px(p))


BG, PANEL, ACCENT = (13, 16, 23), (20, 25, 34), (56, 189, 248)
WHITE, MUTED, DARK = (242, 246, 251), (150, 165, 185), (9, 12, 17)

img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)
d.rectangle([0, 0, W, px(2)], fill=ACCENT)


def fit(text, fontfile, start, max_w):
    p = start
    while p > 8:
        f = font(fontfile, p)
        if d.textlength(text, font=f) <= px(max_w):
            return f
        p -= 0.5
    return font(fontfile, 8)


# icon tile: dipole + figure-8 azimuth pattern
ix0, iy0, ix1, iy1 = px(10), px(13), px(74), px(77)
d.rounded_rectangle(
    [ix0, iy0, ix1, iy1], radius=px(12), fill=PANEL, outline=ACCENT, width=S
)
cx, cy = (ix0 + ix1) // 2, (iy0 + iy1) // 2
lw, lh = px(15), px(11)
for sgn in (-1, 1):
    lc = cx + sgn * px(8)
    d.ellipse([lc - lw, cy - lh, lc + lw, cy + lh], outline=ACCENT, width=S)
g = px(2)
d.line([cx, cy - px(20), cx, cy - g], fill=WHITE, width=S + 1)
d.line([cx, cy + g, cx, cy + px(20)], fill=WHITE, width=S + 1)
d.ellipse([cx - px(2), cy - px(2), cx + px(2), cy + px(2)], fill=ACCENT)

# brand line
tx, TEXT_W = px(90), 552 - 90
fb = font(SEMI, 15)
d.text((tx, px(11)), "Antenna", font=fb, fill=WHITE)
wA = d.textlength("Antenna", font=fb)
d.text((tx + wA, px(11)), "KNoBs", font=fb, fill=ACCENT)
wK = d.textlength("KNoBs", font=fb)
d.text((tx + wA + wK + px(8), px(13)), "· by KK7KNB", font=font(SANS, 10.5), fill=MUTED)

# headline, "code." accented
h1, h2 = "Your antennas, as ", "code."
fh = fit(h1 + h2, SEMI, 26, TEXT_W)
d.text((tx, px(31)), h1, font=fh, fill=WHITE)
d.text((tx + d.textlength(h1, font=fh), px(31)), h2, font=fh, fill=ACCENT)

# subline
sub = "Open-source MoM wire-antenna modeling in Python — cross-checked vs NEC."
d.text((tx, px(69)), sub, font=fit(sub, SANS, 12, TEXT_W), fill=MUTED)

# CTA button — mono URL (echoes the app's readouts), vertically centered
bx0, by0, bx1, by1 = px(560), px(30), px(718), px(60)
d.rounded_rectangle([bx0, by0, bx1, by1], radius=px(7), fill=ACCENT)
cta = "antennaknobs.dev"
fcta = fit(cta, MONO, 14, 718 - 560 - 14)
cw = d.textlength(cta, font=fcta)
bb = fcta.getbbox(cta)
d.text(
    ((bx0 + bx1) / 2 - cw / 2, (by0 + by1) / 2 - (bb[3] - bb[1]) / 2 - bb[1]),
    cta,
    font=fcta,
    fill=DARK,
)

out = img.resize((728, 90), Image.LANCZOS).quantize(
    colors=200, dither=Image.Dither.NONE
)
out.save(OUT, optimize=True)
print("saved", OUT, os.path.getsize(OUT), "bytes")
