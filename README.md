# gta6-trophy-rarity-entry

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/bahrizal/gta6-trophy-rarity-entry/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/bahrizal/gta6-trophy-rarity-entry/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts each GTA VI trophy's earn rate one week after launch from Exophase's page history: per-trophy saturating growth curves, a log-time trend and persistence averaged by fit, with a stale-page correction.

## Run it

```sh
MAKEFUN_INPUT_DIR=path/to/case MAKEFUN_OUTPUT_DIR=path/to/out python3 main.py
```

Reads one case file (schema `makefun/gta6-trophies-case/v1`) from the input
directory and writes five quantiles (5%, 25%, 50%, 75%, 95%) of each
trophy's earn rate at the case's target, as a percent of the players
Exophase tracks for the game. Standard library only, no network,
deterministic, about a third of a second per case.

## Method

The house records Exophase's trophy page on its own clock, so a case's
history is a step function: long stretches where every read is the same
stale page, then a jump when the page refreshes. Two things follow.

First, a trophy's history must be read as its distinct values, not its
raw reads: fitting the raw reads lets a long stale plateau outweigh the
few reads that actually moved. Each distinct rate is taken at the time it
first appeared.

Second, a flat history is not evidence of a flat future. It is evidence
the page has not refreshed: players keep earning trophies while the
printed rates stand still, so the true rates at the checkpoint are
already above the last read, and the longer the page has stood still the
further above they are. The forecast corrects for that with a stale-page
drift: the typical signed logit step at a refresh in the last day, times
how many refresh intervals have passed since the trophy's rate last
changed (capped at three).

Per trophy, three families are fitted at run time from the case's own
history and averaged by how well each fits (mean squared logit error,
scale 0.3 logit):

- **Persistence** — the rate stays where it is.
- **Log-time line** — least squares of logit(rate) on
  ln(1 + hours since the 2026-11-19 00:00 UTC launch) over the last
  24 hours of reads, widening the window until the fit sees at least
  three distinct values.
- **Saturating exponential** — `r(t) = A * (1 - exp(-k*max(0, t-d)))`,
  ceiling `A` fitted per growth rate `k` and opening delay `d`. The
  delay matters: a trophy's content opens when players reach it, so its
  rate sits at zero until then and rises fast after, and a curve
  anchored at the launch cannot bend that way.

The fits run on the interior reads only: a rate reported as exactly
0.00% or 100% is censored, not a measurement, and its logit is an
artefact of the clamp that would outweigh the whole shape of the
history.

The spread is the families' own spreads and their disagreement about
the median, combined with the random walk's (0.5 logit per
square-root day, as the house baseline has it) and a stale-page term:
the typical absolute refresh step times the square root of how many
refresh intervals the page has stood still. Floored at 0.35 logit,
capped at 2.5.

The medians are then kept honest against the game's own structure:
the chapter trophies cannot rise along the story order (Chapter 1 ≥
Chapter 2 ≥ Chapter 3 ≥ Chapter 4 ≥ final mission), and the platinum
and 100% completion sit below every other trophy. Both are enforced
by pooling, never by inventing a rate.

## Validation

On synthetic final-like cases (2.5 days of reads, truth 4.5 days out),
the forecaster beats the house persistence baseline on every seed in
both a fast regime (caps nearly reached by the deadline) and a slow
regime (growth continuing through week one), by 0.16–0.18 and
0.38–0.48 pinball loss respectively, and beats the house log-time
trend baseline by a wide margin. On the two practice cases it sits
just above persistence with a much wider interval, as it should:
the practice histories are flat, so the honest answer is "about
where it is, plus the refresh jump we have not seen yet".

---

<sub>Entered on [make.fun](https://make.fun/?ref=r74f371w&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
