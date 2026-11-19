#!/usr/bin/env python3
"""Forecast GTA VI trophy earn rates on Exophase one week after launch.

Reads one case file (schema makefun/gta6-trophies-case/v1) from the input
directory and writes five quantiles (5%, 25%, 50%, 75%, 95%) of each
trophy's earn rate at the case's target, as a percent of the players
Exophase tracks for the game.

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
further above they are. The forecast corrects for that.

Method, fitted at run time from the case's own history (no network, no
hard-coded rates):

  Per trophy, take the distinct rates in order and fit three families
  to them, then average the families by how well each fits: persistence
  (the rate stays where it is), a least-squares line of logit(rate) on
  ln(1 + hours since the 2026-11-19 00:00 UTC launch) over the last
  24 hours of reads, widening the window until the fit sees at least
  three distinct values, and a saturating exponential r(t) =
  A * (1 - exp(-k*max(0, t-d))) whose ceiling A is fitted per
  growth rate k and opening delay d. The delay matters: a trophy's
  content opens when players reach it, so its rate sits at zero
  until then and rises fast after, and a curve anchored at the
  launch cannot bend that way. The families are weighted by their
  mean squared logit error, with a scale of 0.3 logit: a family
  that explains the history earns its say about the future. The
  fits run on the interior reads only -- a rate reported as exactly
  0.00% or 100% is censored, not a measurement, and its logit is
  an artefact of the clamp that would outweigh the whole shape.
  The log-time axis is the honest shape for launch-week growth: fast
  at first, then flattening, so the line's own slope already damps
  over a long horizon, while the exponential flattens harder toward
  its ceiling. Averaging them is the honest answer where the history
  cannot tell a still-rising trophy from one that has already
  levelled off: the families disagree there, and the disagreement
  widens the spread. The median also carries a stale-page drift: the
  typical signed logit step at a refresh in the last day, times how
  many refresh intervals have passed since this trophy's rate last
  changed (capped at three), floored at zero. Where no refresh was
  ever seen, a prior step of 0.3 logit per refresh and a 6-hour
  interval stand in.

  The spread is the families' own spreads and their disagreement
  about the median, combined with the random walk's (0.5 logit per
  square-root day, as the house baseline has it) and a stale-page
  term: the typical absolute refresh step times the square root of how
  many refresh intervals the page has stood still. The spread is
  floored at 0.35 logit so no interval is ever degenerate, and capped
  at 2.5.

  The medians are then kept honest against the game's own structure.
  Trophies that require other trophies cannot be rarer than what they
  require: the platinum (every trophy) and 100% completion sit below
  every other trophy, and the chapter trophies run in story order, so
  Chapter 1 >= Chapter 2 >= Chapter 3 >= Chapter 4 >= final mission.
  The medians are adjusted to the closest values that respect those
  orderings (pool adjacent violators on each chain), which can only
  move a forecast toward a rate the game itself allows.

Standard library only, deterministic, no randomness, no network.
"""

import json
import math
import os
from datetime import datetime, timezone

# The quantile levels the case asks for, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# The launch, the anchor of the log-time axis.
LAUNCH = datetime(2026, 11, 19, tzinfo=timezone.utc)

# The random walk's spread, in logit units per square-root day.
SIGMA_RW = 0.5
# Smallest and largest spread ever quoted, in logit units.
MIN_SIGMA = 0.35
MAX_SIGMA = 2.5
# Prior step per refresh (logit) and refresh interval (hours) where the
# history never showed a refresh.
PRIOR_STEP = 0.3
PRIOR_JUMP = 0.5
PRIOR_INTERVAL_HOURS = 6.0
# The stale-page drift is capped at this many refreshes' worth.
MAX_STALE_REFRESHES = 3.0
# A fit needs at least this many distinct values to trust its slope.
MIN_DISTINCT_FOR_SLOPE = 3


def parse_time(text):
    """Parse an ISO-8601 UTC timestamp like 2026-11-19T03:00:00Z."""
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def clamp_rate(rate):
    """A rate as a percent, inside the range the logit can take."""
    return min(99.99, max(0.01, rate))


def logit(pct):
    """logit of a percent, clamped so 0% and 100% have a value."""
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def fit_points(series):
    """The reads a fit may use: rates strictly between 0% and 100%.

    A rate reported as exactly 0.00% (or 100%) is censored -- no
    player had the trophy yet (or every player had it) -- and its
    logit is an artefact of the clamp, not a measurement. One such
    point would outweigh the whole shape of the history. Where a
    trophy never left 0% or 100%, there is nothing to fit, and the
    whole series stands in so the fallback still sees it.
    """
    interior = [(t, r) for (t, r) in series if 0.0 < r < 100.0]
    return interior if interior else list(series)


def inv_logit(value):
    """A logit back to a percent, kept inside 0.01..99.99."""
    if value >= 0:
        p = 1.0 / (1.0 + math.exp(-value))
    else:
        e = math.exp(value)
        p = e / (1.0 + e)
    return min(99.99, max(0.01, p * 100.0))


def round4(x):
    """x rounded to 4 decimals, half up, as the house baseline rounds."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def median(values):
    """The median of a non-empty list of numbers."""
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def ols_fit(xs, ys):
    """Least squares y = a + b*x; returns (a, b, residuals)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, []
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0.0:
        return mean_y, 0.0, [y - mean_y for y in ys]
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = mean_y - b * mean_x
    return a, b, [y - (a + b * x) for x, y in zip(xs, ys)]


def log_time(when):
    """ln(1 + hours since the launch), floored at zero before launch."""
    hours = (when - LAUNCH).total_seconds() / 3600.0
    return math.log1p(max(0.0, hours))


def distinct_series(reads):
    """A trophy's distinct rates as (time, rate) pairs, in order.

    `reads` is a list of (time, rate-or-None). Nulls are filled from
    the last rate seen, so a read that lacked the trophy does not break
    the series. Consecutive equal rates collapse to their first
    appearance: the page standing still is one observation, not many.
    Reads before the launch are dropped: a 0% printed before anyone
    could play is not a measurement of the launch trajectory, and its
    logit would be an outlier that drags any fit.
    """
    series = []
    last = None
    for when, rate in reads:
        if when < LAUNCH:
            continue
        if rate is None:
            rate = last
        if rate is None:
            continue
        if last is None or rate != last:
            series.append((when, rate))
            last = rate
    return series


def refresh_stats(series, at):
    """Refresh steps and staleness for one trophy's distinct series.

    Returns (signed_steps, abs_steps, hours_since_change): the signed
    logit step at each refresh, its absolute values, and how many hours
    passed between the last change and `at`. The steps are taken from
    the last 24 hours only: early in a launch the logit steps are
    enormous (a rate moving from 0.00% to 1% is four logit), and
    letting those set the typical step would pretend every refresh
    still moves the rate that far. Where the last day holds fewer
    than two steps, the whole series stands in.
    """
    signed = []
    absolute = []
    window_start = at.timestamp() - 24.0 * 3600.0
    for (_, prev), (when, cur) in zip(series, series[1:]):
        if when.timestamp() < window_start:
            continue
        step = logit(cur) - logit(prev)
        signed.append(step)
        absolute.append(abs(step))
    if len(signed) < 2:
        # Fewer than two steps in the last day is noise, not a
        # typical step: a median of one refresh rests the whole
        # drift on a single jump. Fall back to the whole series so
        # the typical step is measured over more than one refresh.
        signed = []
        absolute = []
        for (_, prev), (_, cur) in zip(series, series[1:]):
            step = logit(cur) - logit(prev)
            signed.append(step)
            absolute.append(abs(step))
    if series:
        hours_since = (at - series[-1][0]).total_seconds() / 3600.0
    else:
        hours_since = 0.0
    return signed, absolute, max(0.0, hours_since)


def pool_adjacent(values):
    """The closest non-increasing sequence to `values` (PAVA).

    `values` runs down a chain that must not rise (Chapter 1 down to
    the final mission). Adjacent violators are pooled to their mean
    until the sequence never rises. Returns a new list.
    """
    blocks = [[v] for v in values]
    i = 0
    while i < len(blocks) - 1:
        # A rise is a block whose mean sits below the next block's mean.
        mean_i = sum(blocks[i]) / len(blocks[i])
        mean_j = sum(blocks[i + 1]) / len(blocks[i + 1])
        if mean_i < mean_j:
            blocks[i : i + 2] = [blocks[i] + blocks[i + 1]]
            i = max(0, i - 1)
        else:
            i += 1
    out = []
    for block in blocks:
        mean = sum(block) / len(block)
        out.extend([mean] * len(block))
    return out


# Grid of growth rates (per day) and opening delays (days) for the
# saturating family.
K_GRID = [0.05 * (1.35 ** i) for i in range(20)]
D_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
# Prior scale for model weighting, in logit units.
SIGMA0 = 0.3


def exp_family(series, target_days):
    """The saturating-exponential family for one trophy.

    Each candidate is r(t) = A * (1 - exp(-k*max(0, t-d))), t in days
    since the launch, A fitted by least squares in rate space for its
    (k, d). The delay d matters: a trophy's content opens when
    players reach it, so its rate sits at zero until then and rises
    fast after; a curve anchored at the launch cannot bend that way,
    and without d it reads a late opening as a low ceiling. The fit
    runs on the interior reads only (see fit_points): censored 0%
    and 100% reads would otherwise outweigh the shape. Returns
    (median_logit, sigma, best_mse): the SSE-weighted average over
    (k, d) of the target logit, the within-family spread (the
    residual plus the spread between the candidates), and the
    family's best mean squared logit error. None when the series is
    empty.
    """
    pts = []
    for (when, rate) in fit_points(series):
        t = max(0.0, (when - LAUNCH).total_seconds() / 86400.0)
        pts.append((t, rate))
    if not pts:
        return None
    candidates = []
    for d in D_GRID:
        for k in K_GRID:
            g = [1.0 - math.exp(-k * max(0.0, t - d)) for (t, _) in pts]
            num = sum(r * gg for (_, r), gg in zip(pts, g))
            den = sum(gg * gg for gg in g)
            if den <= 0:
                continue
            a = min(99.99, max(0.01, num / den))
            sse = 0.0
            for (t, r), gg in zip(pts, g):
                sse += (logit(r) - logit(a * gg)) ** 2
            gt = 1.0 - math.exp(-k * max(0.0, target_days - d))
            candidates.append((sse, a * gt))
    if not candidates:
        return None
    weights = [math.exp(-sse / (2.0 * SIGMA0 * SIGMA0))
               for (sse, _) in candidates]
    total = sum(weights)
    if total <= 0:
        weights = [1.0 / len(candidates)] * len(candidates)
    else:
        weights = [w / total for w in weights]
    logits = [logit(m) for (_, m) in candidates]
    mean_logit = sum(w * l for w, l in zip(weights, logits))
    best_sse = min(sse for (sse, _) in candidates)
    n = len(pts)
    resid_var = best_sse / n if n else 0.0
    between = sum(w * (l - mean_logit) ** 2
                  for w, l in zip(weights, logits))
    sigma = math.sqrt(resid_var + between)
    # The family's fit, as a mean squared error, for weighting it
    # against the other families on the same scale.
    return mean_logit, sigma, (best_sse / n if n else 0.0)


def forecast_trophy(series, at, target, horizon_days, global_step,
                    global_jump, global_interval):
    """Five quantiles for one trophy from its distinct series.

    Three families are fitted to the case's own history and averaged
    by how well each fits it: persistence (the rate stays where it
    is), a log-time line (the house trend baseline's shape), and a
    saturating exponential with an opening delay (growth that starts
    when the trophy's content opens and flattens toward a ceiling).
    Averaging them is the honest answer where the history cannot tell
    a still-rising trophy from one that has already levelled off: the
    families disagree there, and the disagreement widens the spread.
    A stale-page drift is added on top: where the page has stood
    still, the true rates have kept rising unseen.

    `global_step`, `global_jump` and `global_interval` are the typical
    signed refresh step, typical absolute refresh step and typical
    refresh interval across all trophies, for trophies that never moved.
    Returns the median (as a rate) and the five quantiles.
    """
    if not series:
        # No rate was ever seen: a flat 1% with a wide spread.
        med = 1.0
        sigma = MAX_SIGMA
        return med, [round4(inv_logit(logit(med) + z * sigma)) for z in Z]

    last_rate = series[-1][1]
    last_logit = logit(last_rate)
    target_days = max(0.0, (target - LAUNCH).total_seconds() / 86400.0)
    rw_var = (SIGMA_RW * math.sqrt(horizon_days)) ** 2

    # Family 1: persistence. Its fit is measured like the others':
    # how well a constant at the last rate explains the history. A
    # trophy that has been rising is not explained by staying put,
    # and the weight says so -- and neither is its spread: the
    # family's own misfit widens the interval it quotes, so a
    # rising trophy does not get a narrow band around a flat median.
    pers_mse = (sum((logit(r) - last_logit) ** 2 for (_, r) in series)
                / len(series))
    families = [(last_logit, math.sqrt(rw_var + pers_mse), pers_mse)]

    # Family 2: the log-time line over the last 24 hours of distinct
    # values, widening the window until it holds enough to trust.
    slope = 0.0
    intercept = last_logit
    residual = 0.0
    line_sse = 0.0
    for window_hours in (24.0, 48.0, 120.0, 1e9):
        window = [(t, r) for (t, r) in series
                  if (at - t).total_seconds() / 3600.0 <= window_hours]
        if not window:
            continue
        if len(window) >= MIN_DISTINCT_FOR_SLOPE or window_hours >= 1e9:
            xs = [log_time(t) for (t, _) in window]
            ys = [logit(r) for (_, r) in window]
            intercept, slope, residuals = ols_fit(xs, ys)
            line_sse = sum(r * r for r in residuals)
            if residuals:
                residual = math.sqrt(line_sse / len(residuals))
            break
    line_logit = intercept + slope * log_time(target)
    # The line's own spread is the random walk's plus its residual.
    line_sigma = math.sqrt(rw_var + residual ** 2)
    # Its fit, as a mean squared error over the whole distinct
    # series, so it is weighed against the other families on the
    # same scale: a line fitted to the last day must still explain
    # the earlier reads.
    line_mse = (sum((logit(r) - (intercept + slope * log_time(t))) ** 2
                    for (t, r) in series) / len(series)
                if series else 0.0)
    families.append((line_logit, line_sigma, line_mse))

    # Family 3: the saturating exponential.
    exp = exp_family(series, target_days)
    if exp is not None:
        exp_logit, exp_sigma, exp_sse = exp
        families.append((exp_logit, math.sqrt(rw_var + exp_sigma ** 2),
                         exp_sse))

    # Weight the families by their mean squared fit, with a scale of
    # SIGMA0 logit: a family that explains the history earns its say
    # about the future, and one that does not is weighted down.
    weights = [math.exp(-mse / (2.0 * SIGMA0 * SIGMA0))
               for (_, _, mse) in families]
    total = sum(weights)
    weights = [w / total for w in weights]
    median_logit = sum(w * m for w, (m, _, _) in zip(weights, families))

    # Stale-page drift: the page may have stood still since the last
    # change, while players kept earning. Shift the median up by the
    # typical signed step per refresh interval elapsed.
    signed, absolute, hours_since = refresh_stats(series, at)
    step = median(signed) if signed else global_step
    jump = median(absolute) if absolute else global_jump
    interval = global_interval
    refreshes = min(hours_since / interval, MAX_STALE_REFRESHES)
    median_logit += max(0.0, step) * refreshes

    # Spread: the families' own spreads and their disagreement about
    # the median, plus the stale-page term, combined in quadrature.
    var = sum(w * (s ** 2 + (m - median_logit) ** 2)
              for w, (m, s, _) in zip(weights, families))
    stale_sigma = jump * math.sqrt(refreshes)
    sigma = math.sqrt(var + stale_sigma ** 2)
    sigma = min(MAX_SIGMA, max(MIN_SIGMA, sigma))

    quantiles = []
    previous = 0.0
    for z in Z:
        value = round4(inv_logit(median_logit + z * sigma))
        if value < previous:
            value = previous
        quantiles.append(value)
        previous = value
    median_rate = round4(inv_logit(median_logit))
    return median_rate, quantiles


def story_chains(trophies):
    """The orderings the game's own structure imposes, as index lists.

    Chapter trophies run in story order and must not rise along it;
    the platinum and 100% completion sit below every other trophy.
    A chain is a list of trophy positions (into the case's trophy
    list) in the order they must not rise.
    """
    by_desc = {}
    for i, trophy in enumerate(trophies):
        by_desc[trophy.get("description", "")] = i
    chain = []
    for desc in ("Complete Chapter 1.", "Complete Chapter 2.",
                 "Complete Chapter 3.", "Complete Chapter 4.",
                 "Complete the final mission."):
        if desc in by_desc:
            chain.append(by_desc[desc])
    chains = []
    if len(chain) >= 2:
        chains.append(chain)
    # The platinum and 100% completion cap: each sits below every
    # other trophy, expressed as a chain of [cap, every other trophy]
    # handled separately in enforce_caps.
    caps = []
    for i, trophy in enumerate(trophies):
        name = trophy.get("name", "")
        desc = trophy.get("description", "")
        if name == "Vice Got a Price!" or desc == "Attain 100% Completion.":
            caps.append(i)
    return chains, caps


def forecast_case(case):
    """Five quantiles per trophy for one case; returns the output object."""
    trophies = case["trophies"]
    history = case.get("history") or []
    at = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0

    # Per-trophy series of (time, rate), with nulls filled.
    series_list = []
    for i in range(len(trophies)):
        reads = []
        for read in history:
            when = parse_time(read["at"])
            rates = read.get("rates") or []
            rate = rates[i] if i < len(rates) else None
            reads.append((when, rate))
        reads.sort(key=lambda item: item[0])
        series_list.append(distinct_series(reads))

    # Global refresh statistics across trophies that did move, for the
    # trophies that never did.
    all_signed = []
    all_absolute = []
    gaps = []
    for series in series_list:
        signed, absolute, _ = refresh_stats(series, at)
        all_signed.extend(signed)
        all_absolute.extend(absolute)
        for (prev_t, _), (cur_t, _) in zip(series, series[1:]):
            gaps.append((cur_t - prev_t).total_seconds() / 3600.0)
    global_step = median(all_signed) if all_signed else PRIOR_STEP
    global_jump = median(all_absolute) if all_absolute else PRIOR_JUMP
    global_interval = median(gaps) if gaps else PRIOR_INTERVAL_HOURS
    if global_interval <= 0:
        global_interval = PRIOR_INTERVAL_HOURS

    medians = []
    rows = []
    for i, series in enumerate(series_list):
        med, quantiles = forecast_trophy(
            series, at, target, horizon_days,
            global_step, global_jump, global_interval)
        medians.append(med)
        rows.append({"id": trophies[i]["id"], "q": quantiles})

    # Enforce the story orderings on the medians, then rebuild each
    # affected row's quantiles around its adjusted median, keeping the
    # row's own spread.
    chains, caps = story_chains(trophies)
    for chain in chains:
        values = [medians[i] for i in chain]
        adjusted = pool_adjacent(values)
        for pos, idx in enumerate(chain):
            if adjusted[pos] == medians[idx]:
                continue
            old_med = medians[idx]
            new_med = adjusted[pos]
            medians[idx] = new_med
            shift = logit(new_med) - logit(old_med)
            rows[idx]["q"] = _shift_row(rows[idx]["q"], shift)
    for cap in caps:
        others = [medians[i] for i in range(len(trophies)) if i != cap]
        if not others:
            continue
        ceiling = min(others)
        if medians[cap] > ceiling:
            shift = logit(ceiling) - logit(medians[cap])
            medians[cap] = ceiling
            rows[cap]["q"] = _shift_row(rows[cap]["q"], shift)

    return {"trophies": rows}


def _shift_row(quantiles, shift):
    """A row's quantiles shifted by `shift` logit, re-rounded, monotone."""
    out = []
    previous = 0.0
    for q in quantiles:
        value = round4(inv_logit(logit(q) + shift))
        if value < previous:
            value = previous
        out.append(value)
        previous = value
    return out


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(output_dir, exist_ok=True)
    names = sorted(name for name in os.listdir(input_dir)
                   if name.endswith(".in"))
    for name in names:
        case_id = name[: -len(".in")]
        out_path = os.path.join(output_dir, case_id + ".out")
        try:
            with open(os.path.join(input_dir, name), "r",
                      encoding="utf-8") as handle:
                case = json.load(handle)
            result = forecast_case(case)
        except Exception:
            # Never leave a case without a valid forecast: a flat 1%
            # with a wide spread for every trophy the case names.
            try:
                with open(os.path.join(input_dir, name), "r",
                          encoding="utf-8") as handle:
                    case = json.load(handle)
                trophies = case.get("trophies") or []
            except Exception:
                trophies = []
            rows = []
            for trophy in trophies:
                quantiles = [round4(inv_logit(logit(1.0) + z * 2.0))
                             for z in Z]
                rows.append({"id": trophy.get("id"), "q": quantiles})
            result = {"trophies": rows}
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, separators=(",", ":"))


if __name__ == "__main__":
    main()
