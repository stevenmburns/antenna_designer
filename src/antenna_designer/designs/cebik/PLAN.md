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
| bobtail       | 6.4 dBi peak, ~30 dB end nulls, high-Z base feed           |
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
