# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/himawarii7991-byte/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/himawarii7991-byte/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for every Grand Theft Auto VI trophy on PlayStation 5, the percent
of the players Exophase tracks who will have earned it at 2026-11-26 00:00 UTC,
as five quantiles (5%, 25%, 50%, 75%, 95%) per trophy.

Entry for the make.fun bounty
[gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01).

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

`main.ts` (Node 22, standard library only) reads every `*.in` case in the
input directory and writes one `<id>.out` per case to the output directory.
It also runs with no environment set, reading `/input` and writing `/output`,
as the grader does. Deterministic: two runs on the same case write identical
bytes. Runtime is ~0.3 s per case, far under the 8 s cap.

---

<sub>Entered on [make.fun](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>

## Method

All work is in logit space (logit(p) = ln(p / (100 − p))), because the score
is a pinball loss on that scale.

- **The page only changes when Exophase refreshes it** (hours apart), so the
  history is a step function. Per trophy we keep the distinct snapshots (one
  point per value change) and fit a recency-weighted least-squares curve of
  logit(rate) against `ln(1 + hours since release)`. A concave quadratic
  captures saturation (early trophies flatten as the launch cohort finishes
  the early game; late ones keep rising); a line captures the recent slope.
  The median blends the two (70% quadratic, 30% line): the quadratic alone
  under-shoots a still-rising curve, the line alone over-shoots it.
- **Refresh jumps.** At a refresh the tracked-player count jumps and rates
  mostly fall, because the newly tracked players have earned almost nothing
  yet. The spread combines the fit's prediction variance with a modest
  random-walk floor for structural surprises. The jumps are the trend's
  mechanism, not independent shocks on top of it, so they are not added
  again as a separate variance term.
- **Flat history** (no refresh ever seen, as in the practice cases): the
  honest forecast is a mixture over the number of refreshes before the
  target (Poisson with mean = horizon / refresh cadence). Each refresh
  drops the logit by the pooled jump (prior: −ln(1 + 3000 / players
  tracked), the first refresh's dilution). Quantiles come from inverting
  that mixture's CDF, so a short horizon puts the low quantiles at the
  post-refresh level and the median at the current rate.
- **Edge cases.** A trophy at 100% stays 100%. A trophy with no earners
  yet gets a wide upward spread (its rate can only rise). A trophy whose
  rate collapsed to ~0% after having earners goes through the normal
  trend/mixture path. Quantiles are strictly increasing after rounding
  and clamped to (0, 100].

## Backtest (local, practice cases)

The kit holds no practice answers (the server grades), so the practice
answers were reconstructed from the recorded refresh: at 04:30 UTC on
2026-11-19 the page refreshed to 3,868 players tracked and 4,276 trophies
earned (from 638 / 943). The one trophy at 100% stays 100%; the other 54
trophies' rates sum to 10.55% (from 47.80%), a 78% relative fall — the new
cohort had earned almost nothing yet. This reconstruction reproduces the
house's stated practice figures for the persistence baseline (loss ≈ 1.40
vs the stated ≈ 1.51, typical miss 0.68 pp, 90% coverage 0).

Pinball loss on the logit scale, practice cases p00/p01:

| scenario | this entry | flat baseline | trend baseline |
|---|---|---|---|
| A — uniform new-cohort earn rate, calibrated to the 4,276 total | 0.63 | 1.40 | 1.40 |
| B — tier-varying new-cohort earn rate, same total | 0.63 | 1.40 | 1.40 |
| C — no refresh (answer = last rates) | 0.18 | 0.02 | 0.02 |
| D — new cohort earns nothing at all | 0.80 | 1.69 | 1.69 |
| E — milder dilution (new cohort at 30%) | 0.33 | 0.78 | 0.78 |

The entry beats both baselines by ~2.2× on the realistic refresh
scenarios (A, B, D, E). Scenario C is the price of the refresh hedge:
when no refresh happens, the wide low quantiles cost 0.18 instead of the
baseline's 0.02 — small against the 0.77 saved when a refresh does happen.

On synthetic final-like cases (rich history, saturating rates, refreshes
every ~4 h, 4.5-day horizon; three random seeds) the entry scores
0.12–0.15 against 0.27–0.29 for both baselines.

## Files

- `main.ts` — the entry (Node 22, stdlib only)
- `LICENSE` — MIT
- `.makefun/owner.json` — proves this repository is the entrant's
