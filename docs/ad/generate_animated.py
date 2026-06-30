"""AntennaKNoBs animated QRZ banner — 728x90 GIF (direction 1, "Your antennas,
as code."). A dial turns and a REAL Moxon azimuth pattern, computed live by this
repo's momwire engine, morphs forward-beam <-> figure-8.

Rendered at 3x and downscaled (LANCZOS) for antialiasing; delta-frame encoded
(disposal=1, shared palette) so only the dial+pattern zone is stored per frame,
keeping the file under QRZ's 48 KB limit. IBM Plex Sans + Mono match the app.

Run from a venv with antennaknobs installed:
    python docs/ad/generate_animated.py

Fonts: run ./fetch_fonts.sh first, or point AD_FONTS at the IBMPlex*.ttf dir.
"""

import math
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from antennaknobs.designs.beams.moxon import Builder
from antennaknobs.engines.momwire import MomwireEngine

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
OUT = os.path.join(HERE, "antennaknobs_728x90.gif")

SANS, SEMI, MONO = (
    "IBMPlexSans-Regular.ttf",
    "IBMPlexSans-SemiBold.ttf",
    "IBMPlexMono-SemiBold.ttf",
)
S, W, H = 3, 728, 90


def f_(n, p):
    return ImageFont.truetype(FONTS + n, round(p * S))


def s(v):
    return round(v * S)


def lw(v):
    return max(1, round(v * S))


BG, PANEL, ACCENT, LIGHT = (13, 16, 23), (20, 25, 34), (56, 189, 248), (125, 211, 252)
WHITE, MUTED, DARK, GRID = (242, 246, 251), (150, 165, 185), (9, 12, 17), (40, 49, 62)

# --- real Moxon azimuth cuts across a t0_factor sweep (the "knob") ---
IT, DYN = 89, 26.0  # azimuth cut at the peak-gain elevation row; dB dynamic range
T0S = [0.400, 0.410, 0.420, 0.430, 0.438, 0.445]


def cut(t0):
    b = Builder(dict(Builder.default_params, t0_factor=t0))
    ff = MomwireEngine(b).far_field(n_theta=90, n_phi=360, del_theta=1, del_phi=1)
    c = np.array(ff.rings)[IT]
    ph = np.deg2rad(np.array(ff.phis))
    r = np.clip((c - (c.max() - DYN)) / DYN, 0, 1)  # per-frame normalize
    return r, ph


CUTS = [cut(t) for t in T0S]

# --- static base frame (supersampled) ---
base = Image.new("RGB", (W * S, H * S), BG)
d = ImageDraw.Draw(base)
d.rectangle([0, 0, W * S, s(2)], fill=ACCENT)


def fit(text, fontfile, start_px, max_w_px):
    px = start_px
    while px > 8:
        fnt = f_(fontfile, px)
        if d.textlength(text, font=fnt) <= s(max_w_px):
            return fnt
        px -= 0.5
    return f_(fontfile, 8)


tx, TEXT_W = s(196), 552 - 196  # text clears the CTA button at x=560
fb = f_(SEMI, 15)
d.text((tx, s(11)), "Antenna", font=fb, fill=WHITE)
wA = d.textlength("Antenna", font=fb)
d.text((tx + wA, s(11)), "KNoBs", font=fb, fill=ACCENT)
wK = d.textlength("KNoBs", font=fb)
d.text((tx + wA + wK + s(8), s(13)), "· by KK7KNB", font=f_(SANS, 10.5), fill=MUTED)
h1, h2 = "Your antennas, as ", "code."
fh = fit(h1 + h2, SEMI, 25, TEXT_W)
d.text((tx, s(31)), h1, font=fh, fill=WHITE)
d.text((tx + d.textlength(h1, font=fh), s(31)), h2, font=fh, fill=ACCENT)
sub = "Open-source MoM wire-antenna modeling in Python — cross-checked vs NEC."
d.text((tx, s(69)), sub, font=fit(sub, SANS, 12, TEXT_W), fill=MUTED)
bx0, by0, bx1, by1 = s(560), s(30), s(718), s(60)
d.rounded_rectangle([bx0, by0, bx1, by1], radius=s(7), fill=ACCENT)
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

DCX, DCY, DR = s(42), s(47), s(22)  # dial
PCX, PCY, PR = s(120), s(47), s(31)  # pattern


def frame(step, n):
    im = base.copy()
    dr = ImageDraw.Draw(im)
    dr.ellipse(
        [DCX - DR, DCY - DR, DCX + DR, DCY + DR],
        fill=PANEL,
        outline=ACCENT,
        width=lw(2),
    )
    for a in range(-135, 136, 27):
        ra = math.radians(a - 90)
        dr.line(
            [
                DCX + (DR - s(4)) * math.cos(ra),
                DCY + (DR - s(4)) * math.sin(ra),
                DCX + (DR - s(1)) * math.cos(ra),
                DCY + (DR - s(1)) * math.sin(ra),
            ],
            fill=GRID,
            width=lw(1),
        )
    ang = math.radians(-120 + 240 * (step / (n - 1)) - 90)
    px = DCX + (DR - s(6)) * math.cos(ang)
    py = DCY + (DR - s(6)) * math.sin(ang)
    dr.line([DCX, DCY, px, py], fill=LIGHT, width=lw(3))
    dr.ellipse([px - s(3), py - s(3), px + s(3), py + s(3)], fill=ACCENT)
    dr.ellipse([DCX - s(3), DCY - s(3), DCX + s(3), DCY + s(3)], fill=ACCENT)
    for fr in (0.5, 1.0):
        dr.ellipse(
            [PCX - PR * fr, PCY - PR * fr, PCX + PR * fr, PCY + PR * fr],
            outline=GRID,
            width=lw(1),
        )
    dr.line([PCX - PR, PCY, PCX + PR, PCY], fill=GRID, width=lw(1))
    dr.line([PCX, PCY - PR, PCX, PCY + PR], fill=GRID, width=lw(1))
    r, ph = CUTS[step]
    pts = [
        (PCX + PR * rr * math.cos(p), PCY - PR * rr * math.sin(p))
        for rr, p in zip(r, ph)
    ]
    dr.line(pts + [pts[0]], fill=ACCENT, width=lw(2), joint="curve")  # outline only
    return im.resize((W, H), Image.LANCZOS)


order = list(range(len(T0S))) + list(range(len(T0S) - 2, 0, -1))  # ping-pong
frames = [frame(s_, len(T0S)) for s_ in order]

master = Image.new("RGB", (W, H * len(frames)))
for i, fr in enumerate(frames):
    master.paste(fr, (0, i * H))
pal = master.quantize(colors=220, dither=Image.Dither.NONE)
fp = [fr.quantize(palette=pal, dither=Image.Dither.NONE) for fr in frames]
fp[0].save(
    OUT,
    save_all=True,
    append_images=fp[1:],
    duration=130,
    loop=0,
    optimize=True,
    disposal=1,
)
print("saved", OUT, os.path.getsize(OUT), "bytes")
