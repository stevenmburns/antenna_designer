# Cebik (W4RNL) design candidates

Source material: L. B. Cebik's antenna articles and books, archived at
<https://antenna2.github.io/cebik/> (the "Cebik website"), the
*Antennas Made of Wires* Vols 1–3 on the Internet Archive, his antenneX
back-issues, and his ARRL *Antenna Handbook* chapters (LPDA, log-periodic).

## Selection principle

"Interesting" = **different from what's already modeled**. The existing
catalog already covers: dipoles (inv-vee, fan, trap, folded, turnstile,
short-loaded), loops (delta + diamond, slanted, arrays, turnstile),
parasitic beams (Yagi, hexbeam, Moxon), bowties + arrays, verticals
(ground-plane, raised), and horizontally-polarized wire curtains (Sterba,
hentenna, hourglass). The Extended Double Zepp is already present as
variants of `freq_based/invvee.py`.

The candidates below deliberately target categories the catalog lacks.

## Candidates

### Tier 1 — build first (distinct, iconic, clearly modelable)

1. **Half-square** (`half_square.py`) — VERTICALLY polarized wire antenna:
   two λ/4 vertical legs joined by a λ/2 horizontal top wire, end/corner
   fed. ~3.8 dBi bidirectional broadside, low takeoff angle. *Gap filled:*
   first vertically-polarized wire array (existing verticals are single
   ground-plane radiators). Single feed, simple geometry.

2. **Bobtail curtain** (`bobtail.py`) — the iconic W4RNL VP curtain: three
   λ/4 verticals on ~λ/2 spacing, joined by a ~1λ top wire, fed at the
   base of the center vertical. ~5–7 dBi broadside, very low angle. Natural
   extension of the half-square; requires a ground model.

3. **Cubical quad (2-element beam)** (`quad.py`) — a driven full-wave SQUARE
   loop + a parasitic reflector loop. *Gap filled:* the catalog has delta
   (triangle) and diamond loops but no square quad beam — the classic
   loop-beam topology. Horizontally polarized, ~7 dBi forward.

4. **Lazy-H** (`lazy_h.py`) — two collinear elements stacked λ/2 apart and
   fed IN PHASE via a phasing harness. *Gap filled:* vertical stacking of
   collinear radiators (existing arrays phase identical elements
   side-by-side). Exercises `build_tls()`/`build_network()` in a new way.

### Tier 2 — stretch goals (very different design methodology)

5. **LPDA — log-periodic dipole array** (`lpda.py`) — N dipoles scaled by
   τ (tau) and σ (sigma) with a transposed (phase-alternating) feed boom.
   *Gap filled:* frequency-INDEPENDENT wideband array; a completely
   different design math from every resonant antenna here. Most complex
   (transposed feeder phasing across many elements).

6. **HB9CV / ZL-Special** (`hb9cv.py`) — a 2-element ALL-DRIVEN phased beam
   (~135° phasing line between two driven elements). *Gap filled:* contrast
   to the parasitic Yagi — both elements driven, gain from phasing not
   parasitics.

### Tier 3 — traveling-wave / broadband (needs resistive termination)

The framework's `network.Load` supports lumped R loads, so terminated
types are feasible:

7. **Terminated rhombic** (`rhombic.py`) — four long wires in a diamond with
   a terminating resistor for unidirectional, traveling-wave operation.
   *Gap filled:* non-resonant traveling-wave antenna.

8. **T2FD — terminated tilted folded dipole** (`t2fd.py`) — folded dipole
   with a series terminating resistor for broadband, low-Q response.
   *Gap filled:* deliberately broadband (sacrifices gain for bandwidth).

## Status

All eight implemented, verified against the MKL-backed PyNEC solver, and
covered by physics regression tests in `tests/test_cebik_designs.py`. Each
auto-registers in the web examples registry (so it also picks up the generic
schema tests) and is reachable from the CLI as `cebik.<name>`. Verified
free-space figures at the 28.57 MHz family default:

| design        | headline result                                            |
|---------------|------------------------------------------------------------|
| half_square   | 4.7 dBi, ~62 ohm corner feed, VP bidirectional broadside   |
| bobtail       | 6.4 dBi peak, deep end nulls, ~50 ohm current-max tap feed |
| quad          | 7.0 dBi forward, resonant driver ~132 ohm                  |
| lazy_h        | 8.1 dBi (stacking gain), HP broadside, high-Z tuner feed   |
| lpda          | ~6-9 dBi forward across ~24-32 MHz (feed-Z caveat in file) |
| hb9cv         | ~6.8 dBi endfire, ~50 ohm inductive (F/B caveat in file)   |
| rhombic       | 8 dBi, ~18 dB F/B, unidirectional; Z ~ termination         |
| t2fd          | SWR < 1.8 over 14-56 MHz; broadband, low gain by design    |

Two models carry documented modeling caveats (see their module docstrings):
the LPDA's ideal lossless crossed feeder gives an unreliable feedpoint
impedance, and the HB9CV's single-ended crossed TL reaches only ~8 dB F/B
(the deep ZL null wants a true differential transposed line / pysim DiffTL).

**Bobtail feedpoint revision (post-merge):** the bobtail was originally
base-fed at the classic "tank" point (the bottom of the centre vertical). That
is a current null -> a high, strongly-reactive impedance (~5000-8000j) that, a
segment-convergence + multi-basis study showed, neither converges with
segmentation nor agrees between solver bases (NEC2/sinusoidal ~5000, rooftop
~2700, bspl-d2 ~3500 ohm) -- the textbook ill-conditioned high-Z feedpoint
Cebik warned about, and a unun can't fix it because it scales the large
reactance too. We re-fed it like `half_square`: tapped at a current MAXIMUM
(`feed_height_frac` up the centre vertical), giving a coax-direct ~50 ohm that
all four engines agree on (45-49 ohm) at identical gain/pattern. Gain and
pattern were basis-stable throughout; only the feed Z was unreliable -- exactly
Cebik's "model for gain/pattern, treat high-Z feeds as approximate" guidance.

## Batch 2 (PR #91)

A second eight-model set, again chosen to hit categories the catalog (now
including batch 1) still lacked. All free-space, self-contained, MKL-PyNEC
verified, with regression tests in `tests/test_cebik_designs.py`:

| design            | gap filled / headline result                              |
|-------------------|-----------------------------------------------------------|
| w8jk              | 180-deg all-driven flat-top: ~5.8 dBi bidirectional       |
|                   | endfire, deep broadside + overhead nulls (array action)   |
| phased_verticals  | 90-deg cardioid: feed-phase steering, ~5.2 dBi, ~6-7 dB   |
|                   | F/B (current-forcing network deepens it; see caveat)      |
| inverted_l        | bent/top-loaded vertical on elevated radials: resonant    |
|                   | ~18 ohm, VP low-angle                                     |
| ocf_dipole        | off-centre (Windom) feed: R rises ~3x off centre to       |
|                   | ~230 ohm (the defining OCF physics)                       |
| vbeam             | resonant long-wire V: ~5 dBi along the bisector, deep     |
|                   | broadside null (standing-wave sibling of the rhombic)     |
| bisquare          | 2 wl VP loop curtain: VP broadside, ~3.4 dBi free-space   |
|                   | (the gain reputation is a low-mount over-ground effect)   |
| jpole             | stub-matched end-fed half-wave: omni VP ~2 dBi, SWR < 2   |
| discone           | broadband vertical (disc+cone wire cage): SWR < 3 over    |
|                   | ~34-65 MHz above the cone cutoff, low-angle omni          |

Design-method notes learned this round:
- VP/ground-mounted antennas (inverted_l, the verticals) are best modelled
  with a small ELEVATED-RADIAL counterpoise in free space (cf.
  `designs/vertical.py`) rather than a PEC/finite ground card: a wire whose
  base sits exactly on a PEC plane gave a pathological ~-10 kohm feed
  reactance, whereas radials give a clean, deterministic feedpoint.
- A two-element cardioid fed by VOLTAGE sources only reaches ~5-7 dB F/B
  because the two driving-point impedances differ, so equal voltages give
  unequal currents; `phased_verticals` exposes a tuned complex `front_voltage`
  (~-1.2j) to partly compensate. A genuinely deep null needs a current-forcing
  feed network (Christman/Lewallen) — same family of caveat as hb9cv.
- A single-wire Franklin collinear was prototyped and dropped: with only two
  half-wave sections a centre feed already makes them in-phase (it degenerates
  to lazy_h's 1 wl element), and the multi-section phase-reversed curtain is
  already covered by `freq_based/sterba`. The discone took its slot instead.
