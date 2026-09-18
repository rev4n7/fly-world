# Fly 3D: flight scoping + build (male-cns:v1.0), 2026-09-19

Scripts: `flight_scan.py`, `flight_wiring.py` (scoping), `fly3d_test.py` (self-test → `fly3d_test_out.txt`).

## What the data has for flight
- **Flight DNs:** 24 known types / 63 cells, incl. DNg02_a-g (29 cells; wing power, Namiki et al. 2022),
  DNa01-10, DNb01/06, DNp03/15 (DNHS1)/20 (DNOVS1)/22 (DNOVS2).
- **Wing motor neurons** (VNC, subclass `wm`): power DLMn/DVMn, steering b1-3, i1-2, iii1/3, hg1-4, tp1-2, tpn, ps1.
- **Wiring:** DNg02 → power MNs directly (~900-1,440 synapses per subtype; 1,930 onto the 24 power MNs
  in the simulated set). Steering DNs → wing steering MNs through VNC cells, mostly same-side
  (DNa03: 179k vs 26k path weight; DNa02: 75k vs 29k).
- **Motion vision:** HS/VS/H2 → DNs exist (VS → DNp20 936 syn, HST → DNg41 688, H2 → DNp15 454).
  **Deferred:** it needs a full optic-flow simulation. So flight height is held by a design rule, not by vision.
- **Looming in 3D:** looming cells have real elevations (−54° to +53°). Tested in the brain alone: a threat
  from above gives a downward escape (−0.44), and one from below an upward escape (+0.39).

## Build (`flypac/fetch_fly3d.py` → data/fly3d; `flypac/fly3d.py`; `flypac/fly3d_server.py`; `web/fly3d/`)
- +94 neurons (29 DNg02, 24 power MNs, 41 steering MNs) → 4,188 simulated (with mirrored ears), 3.52M synapses,
  and 957k real 3D synapse bins (8 um) for the 3D brain and 3D strokes.
- **Wing power = real power-MN firing** (3-7 Hz while flying; real DLM MNs fire ~5-10 Hz), driven by DNg02 through
  real synapses. DESIGN: 40 Hz "keep flying" drive into DNg02 while airborne; the lift reference is set by
  the first 2 s of flight; 0.8 s launch phase after takeoff.
- **Neck cut is a real cut here:** synapses from severed descending neurons onto body (VNC) cells are removed,
  so the wing MNs lose DNg02 input and the fly falls.
- **3D escape direction** uses the real elevation of the looming cells that fired, relative to their population
  average (the looming cells over-represent the upper field). From the ground, the Giant Fiber jump goes up.

## Self-test (2 seeds, fly3d_test.py; injury applied after 3 s of healthy flight)
| scenario | result |
|---|---|
| healthy flight | cruises ~3 units high, lands and takes off now and then |
| neck cut both | falls (wing power 6%), stays down |
| DNg02 removed | falls (wing power 6%) |
| 30% random cell loss | wing power 39%, sinks |
| predator from above | escapes downward (−0.51) |
| predator from below | escapes upward (+0.33) |
| female on the ground | descends to her, stays near 57% of the time |
| female, LC10a removed | 10% |

## Performance fixes, 2026-09-19 (user reported lag)
- **Requests took ~2 s:** the page used `localhost`, and Windows tries IPv6 first, which the server didn't
  listen on. Now uses 127.0.0.1. The sim thread also held a lock that page requests waited on. Now the sim
  thread publishes a JSON snapshot ~22x/s and queues clicks; a request takes ~5 ms.
- **Brain slowed 8x when quiet:** decaying synaptic input fell into subnormal floats (< 1e-308), which x86
  CPUs process ~100x slower. `Brain.step` now flushes |g| < 1e-9 to 0 (no effect on spikes above that).
- **Brain.step 1.9x faster:** vectorised spike propagation, and refractoriness is checked only for cells
  that fired in the last 2.2 ms. Verified: identical spikes to `step_reference` (25,703 vs 25,703).
- Result: the server runs at 1.00x real time while the page polls it.
- **World = closed box:** 40 x 40 floor, walls, and a ceiling at 12 units (the fly and predators are clamped
  inside). The walls face inward, so the free camera can look in from outside.
