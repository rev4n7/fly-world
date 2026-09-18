# Fly World: circuit scoping (male-cns:v1.0), 2026-09-18

Scripts in this folder (neuprint-python, token from `.env`): `types_scan.py`, `props.py`,
`syn_scan.py`, `circuits.py`, `forage.py`, `forage3.py`.

Useful annotations in the dataset: `synonyms` (literature names), `fruDsx`, `dimorphism`,
`class`/`subclass`/`superclass`, `entryNerve`. The VNC is included.

## Escape: already built (Fly-Pacman)
LC4/LPLC2/LC6 → Giant Fiber (DNp01) + DNp02/04/11. No change needed.

## Courtship: WELL SUPPORTED
| step | neurons (dataset type) | real synapses |
|---|---|---|
| sees female (moving fly-sized object) | LC10a (275 cells) → AOTU019/012/015/025 → DNa02 | 6-7k per side into AOTU019 |
| sees female → arousal | LC10a → pC1_1a / pC1_1b (P1-like: male-specific, fru+dsx) | 740 |
| taps female (leg taste) | leg GRNs LgLG5/6/7/8/1a → vAB3 (= AN09B017a-g, synonym "vAB3") | 8,406 |
| pheromone → arousal | vAB3 → pC1 (mostly pC1_3a/b/c, 5b, 11a) | 1,582 |
| pheromone → brake | vAB3 → mAL_m* (GABA) → pC1 | 6,067 (mAL→pC1) |
| arousal → song | pC1_14a/5b/7a/7b/19 → pIP10 (song command DN) → VNC song cells (TN1a, dPR1, vPR9) | 1,300+ → 9,644 |
| arousal → chase | pC1_1a/4b → AOTU012/015 (pursuit steering) | ~400 |

Caveats: which leg taste cells are ppk23 (female-pheromone) cells is not labelled; they are
inferred as "the leg taste cells that feed vAB3". Real P1→pursuit gating is partly
dopaminergic/modulatory, which synapse counts don't capture. PPN1 (AN05B102a) exists but
barely touches pC1 (2 synapses), so it is left out.

## Foraging: NOT CLEANLY AVAILABLE as an approach circuit
- Food-odour receptors exist and are typed with sides: ORN_DM1, VA2, DM4, DP1m, DM2, DM5 → their
  PNs (13-15k synapses, clean).
- But food-odour PNs mostly feed the mushroom body (Kenyon cells, learned memory) and AL local
  neurons. Route to steering neurons (DNa02/DNa03): one thin 3-hop route via CB0683
  (23-46 synapses on the first hop), vs 6,600 for LC10a → AOTU019. Too weak to steer in the LIF.
- MBONs → DNa03/DNa02 exist (MBON31/27/26/32, 100-250 syn) but MB output depends on learned
  weights not in the connectome.
- Eating on contact IS wired: taste GRNs (claw_tpGRN, LB3a-d, PhG2) → GNG interneurons → MN9
  (proboscis-extension motor neuron). Caveat: which GRNs are sugar vs bitter isn't labelled.

## Build log, 2026-09-18 (what the simulation showed)
Data layer: `flypac/fetch_world.py` → data/world (4,412 neurons). Simulated set after the exclusions
below: `flypac/worldbrain.py` → 4,094 neurons, 3.56M synapses. Test scripts: `circuit_test.py`,
`fb_test.py`, `scenarios.py` (results in `scenarios_out.txt`).

1. **Global feedback layer = runaway.** The Fly-Pacman rule (>=100 in / >=100 out) applied to all new
   circuits at once picked ~3,400 cells. A weak LC10a stimulus then ignited ~800 cells at >50 Hz within
   0.4 s. Applied **per circuit** (taste: 569 cells, courtship: 576) it is stable, and the taste layer is
   what switches MN9 off after the fly leaves food. Without it, MN9 stays at ~480 Hz forever.
2. **Smell does not steer. Tested, then left out.** At 10 synapses per hop, 460 cells lie on
   PN→A→B→DNa02/03 routes. Driving ONLY the left antenna, or ONLY the right one, turned the fly LEFT both
   times, and 160+ cells ran hot even at 2 Hz receptor input. No usable odour direction reaches steering in
   this data, so those 460 cells are dropped. The PNs' remaining outputs (484 synapses, onto left-side cells
   only) made the fly circle left whenever food existed, so they are cut too. Smell is display-only
   (ORN → PN). The fly finds food by wandering.
3. **Receptor inputs dropped.** ORN→ORN contacts drove ORNs to ~180 Hz for a 60 Hz stimulus. Receptor
   cells (ORN, taste, leg taste) now fire only from the stimulus. ORN rate is 2 Hz max, because each ORN has
   ~180 synapses onto its PN.
4. **Which taste cells = food.** Unlabelled in the data. Stimulating each of the 31 mouth taste types alone:
   only LB3b/c/d and PhG1b/c drive MN9. All 31 together, or LB3 + PhG1, leave MN9 silent. Food =
   LB3b/c/d (the labellum touches food first).
5. **Touching the female doesn't excite P1.** vAB3 (AN09B017a-g) is predicted **glutamatergic** in this
   dataset. Under the model's sign rule (Shiu et al. 2024: Glu = inhibitory) the pheromone relay inhibits
   P1. The literature describes vAB3 as exciting P1. Not changed by hand. LC10a→pC1 (740 synapses) is too
   weak to recruit P1 either. So in this model the female is chased (LC10a→AOTU→DNa02), but P1/song are
   not recruited by her. Ablating P1 therefore doesn't change chasing.
6. **P1 ↔ AVLP717m loop.** Before the courtship feedback layer, the pC1 network, once kicked, self-sustained
   at 100-240 Hz and drove pIP10 (song). It was kicked by predator activity, not by the female. With the
   feedback layer this is mostly gone: the song neuron fires briefly with predators (0.5 s per 30 s), and
   more with GF ablated.

### Self-test (3 seeds x 30 s each, scenarios.py)
| scenario | result |
|---|---|
| female only | within 2 units of her 65% of the time |
| female only, LC10a ablated | 2%: ignores her |
| predator only | 6.0 takeoffs, 0 caught |
| predator only, GF ablated | 0 takeoffs, 1.7 caught |
| 6 foods | 1.11 food eaten |
| 6 foods, MN9 ablated | 0 eaten (walks over food) |
| female + foods, LC10a ablated | ignores female (8%), still eats 1.11 |
| foods + predator | 0.67 eaten, 4.7 takeoffs (escape interrupts eating) |
| foods + predator, GF ablated | 1.14 eaten, 0 takeoffs, 2.3 caught |
| female + predator | near female 20%, 11 takeoffs |
| female + predator, GF ablated | near female 52%, 0 takeoffs, 1.7 caught |

## Injury lab, 2026-09-18 (`flypac/flyworld.py`, `flypac/injury.py`; test: `injury_test.py` → `injury_out.txt`)
Walking speed now comes from DNa02 L+R activity (resting ~10 Hz = normal walk), so movement needs the brain.
- **Neck cut** = the 146 descending neurons (dataset `superclass`) of one or both sides stop reaching the
  body. The brain still fires them (visible in the brain view). Both sides: speed 0, caught 3x by a predator
  with no takeoffs, but still eats food under its mouth (MN9 is a head motor neuron, not descending).
  Left half: half speed.
- **Random cell loss:** 30% → follows the female 8% of the time (healthy 57%). 60% → barely moves.
- **All synapses 50% weaker:** follows the female 22% of the time.
- **Weaker brakes** (inhibitory synapses): at 20% strength, nothing happens at rest; a female triggers
  runaway activity (seizure 14 s of 20, 41 spasm-like takeoffs). At 50%, the P1 ↔ AVLP717m loop runs away
  and the song neuron sits at ~280 Hz.
- **Stroke:** click the brain picture. A cell dies if >= 15% of its real synapse locations fall in the disc.
  A 64 um stroke on the left lobula kills most left visual cells (100 LC10a, 94 LPLC2, 66 LC4...).
- **Left eye removed:** ignores a female on its left (5% vs 67%).
- Not simulated: memory (no learning in the model yet), dyslexia (no honest fly equivalent).
