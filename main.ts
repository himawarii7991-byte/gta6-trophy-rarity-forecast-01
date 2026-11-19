// gta6-trophy-rarity-forecast-01 entry (Node 22, standard library only).
//
// Forecasts, for every GTA VI trophy, five quantiles of its Exophase earn rate
// (percent of tracked players) at the case's target, from the house's reads.
//
// Method (logit space throughout; logit(p) = ln(p / (100 - p))):
//   * The page only changes when Exophase refreshes it (hours apart), so the
//     history is a step function. Per trophy we keep the distinct snapshots
//     (one point per value change) and fit a recency-weighted least-squares
//     curve of logit(rate) against u = ln(1 + hours since release). A concave
//     quadratic (c <= 0) captures saturation: early trophies flatten as the
//     launch cohort finishes the early game, late ones keep rising. Fewer
//     than 5 distinct points falls back to a line; fewer than 3 to the
//     refresh mixture below.
//   * Refresh jumps: at a refresh the tracked-player count jumps and rates
//     mostly fall (new players have earned almost nothing yet). We pool the
//     logit jumps across trophies and refreshes (median / MAD) and the
//     refresh cadence (median gap between distinct snapshots). The spread
//     combines the fit's prediction variance, the variance of the refreshes
//     expected before the target, and a random-walk floor.
//   * Flat history (no refresh seen, e.g. the practice cases): the honest
//     forecast is a mixture over the number of refreshes before the target
//     (Poisson with mean = horizon / cadence). Each refresh drops the logit
//     by the pooled jump (prior: -ln(1 + 3000 / players_tracked), the first
//     refresh's dilution). Quantiles come from inverting that mixture's CDF
//     on a grid, so a short horizon puts the low quantiles at the
//     post-refresh level and the median at the current rate.
//   * A trophy at 100% stays 100%. A trophy with no earners yet gets a wide
//     upward spread (its rate can only rise).
// Deterministic: no randomness, no clock, no network. Output rows sorted by id.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Random-walk spread, logit units per square-root day (the house baseline's). */
const SIGMA_RW = 0.5;
/** Random-walk floor for the trend branch: structural surprises only. */
const SIGMA_TREND = 0.25;
/** Weight of the concave-quadratic median vs the linear median (0..1). */
const QUAD_W = 0.7;
/** Recency half-life for the trend fit, hours. */
const HALF_LIFE_H = 24;
/** Trend fit window: distinct points from the last this many hours (or all). */
const FIT_WINDOW_H = 48;
/** Prior refresh cadence, hours (the first refresh's gap on launch night). */
const CADENCE_PRIOR_H = 4.5;
/** Prior new players per refresh (first refresh added ~3230). */
const NEW_PER_REFRESH_PRIOR = 3000;
/** Prior logit spread of one refresh jump when no refresh was ever seen. */
const JUMP_SD_PRIOR = 0.6;
/** Cap on the total refresh drift applied to a flat-history median, logit. */
const MAX_DRIFT = 3;

interface Trophy { id: number; tier: string }
interface Read { at: string; players_tracked: number; rates: Array<number | null> }
interface Case {
  id: string; at: string; horizon_minutes: number; target: string;
  levels: number[]; game: { release_date: string };
  trophies: Trophy[]; history: Read[];
}

const clamp = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => Math.log(clamp(pct) / (100 - clamp(pct)));
const invLogit = (l: number): number => 100 / (1 + Math.exp(-l));
const round4 = (x: number): number => Math.round(x * 10000) / 10000;
/** Normal CDF (Abramowitz-Stegun 7.1.26, |err| < 1.5e-7). */
function normCdf(x: number): number {
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const d = 0.3989423 * Math.exp(-x * x / 2);
  const p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
  return x > 0 ? 1 - p : p;
}
function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const n = s.length;
  return n === 0 ? 0 : n % 2 === 1 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
}

/** Solve a small symmetric linear system by Gauss-Jordan; null when singular. */
function solve(A: number[][], b: number[]): number[] | null {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    if (Math.abs(M[piv][col]) < 1e-12) return null;
    [M[col], M[piv]] = [M[piv], M[col]];
    const d = M[col][col];
    for (let j = col; j <= n; j++) M[col][j] /= d;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = M[r][col];
      for (let j = col; j <= n; j++) M[r][j] -= f * M[col][j];
    }
  }
  return M.map((row) => row[n]);
}
/** Invert a small symmetric matrix via Gauss-Jordan; null when singular. */
function invert(A: number[][]): number[][] | null {
  const n = A.length;
  const M = A.map((row, i) => [...row, ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    if (Math.abs(M[piv][col]) < 1e-12) return null;
    [M[col], M[piv]] = [M[piv], M[col]];
    const d = M[col][col];
    for (let j = col; j < 2 * n; j++) M[col][j] /= d;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = M[r][col];
      for (let j = col; j < 2 * n; j++) M[r][j] -= f * M[col][j];
    }
  }
  return M.map((row) => row.slice(n));
}

/**
 * Weighted least squares of y on basis functions of u. Returns the fitted
 * coefficients, the residual std, and the prediction variance factor
 * x_t' (X'WX)^-1 x_t at the target point, or null when it cannot be fit.
 */
function wls(
  pts: Array<{ u: number; y: number; w: number }>,
  basis: (u: number) => number[],
  xTarget: number[],
): { coef: number[]; sigRes: number; leverage: number } | null {
  const p = xTarget.length;
  const XtWX: number[][] = Array.from({ length: p }, () => new Array<number>(p).fill(0));
  const XtWy: number[] = new Array<number>(p).fill(0);
  for (const pt of pts) {
    const x = basis(pt.u);
    for (let i = 0; i < p; i++) {
      XtWy[i] += pt.w * x[i] * pt.y;
      for (let j = 0; j < p; j++) XtWX[i][j] += pt.w * x[i] * x[j];
    }
  }
  const coef = solve(XtWX, XtWy);
  if (coef === null) return null;
  const Minv = invert(XtWX);
  if (Minv === null) return null;
  let rss = 0;
  let wsum = 0;
  for (const pt of pts) {
    const x = basis(pt.u);
    const fit = x.reduce((s, xi, i) => s + xi * coef[i], 0);
    rss += pt.w * (pt.y - fit) * (pt.y - fit);
    wsum += pt.w;
  }
  const dof = Math.max(pts.length - p, 1);
  const sigRes = Math.sqrt(rss / dof / Math.max(wsum / pts.length, 1e-9));
  let lev = 0;
  for (let i = 0; i < p; i++) for (let j = 0; j < p; j++) lev += xTarget[i] * Minv[i][j] * xTarget[j];
  return { coef, sigRes, leverage: Math.max(lev, 0) };
}

function forecastCase(c: Case): Array<{ id: number; q: number[] }> {
  const release = new Date(`${c.game.release_date}T00:00:00Z`).getTime();
  const hoursSince = (iso: string): number => (new Date(iso).getTime() - release) / 3600000;
  const uOf = (iso: string): number => Math.log(1 + Math.max(0, hoursSince(iso)));
  const hAt = hoursSince(c.at);
  const uTarget = Math.log(1 + Math.max(0, hAt + c.horizon_minutes / 60));
  const horizonH = c.horizon_minutes / 60;
  const horizonD = c.horizon_minutes / 1440;

  // Distinct snapshots: a read counts as new when its rates vector or the
  // tracked count differs from the previous kept read.
  const snaps: Read[] = [];
  for (const r of c.history) {
    const prev = snaps[snaps.length - 1];
    if (!prev) { snaps.push(r); continue; }
    const sameRates = r.rates.length === prev.rates.length &&
      r.rates.every((v, i) => v === prev.rates[i]);
    if (!sameRates || r.players_tracked !== prev.players_tracked) snaps.push(r);
  }
  const lastRead = c.history[c.history.length - 1];
  const nLast = lastRead.players_tracked;

  // Refresh cadence: median gap (hours) between distinct snapshots.
  const gaps: number[] = [];
  for (let k = 1; k < snaps.length; k++) {
    gaps.push((new Date(snaps[k].at).getTime() - new Date(snaps[k - 1].at).getTime()) / 3600000);
  }
  const cadence = gaps.length > 0 ? Math.max(0.5, median(gaps)) : CADENCE_PRIOR_H;
  const expectedRefreshes = horizonH / cadence;

  // Pooled refresh jumps (logit) across trophies and refreshes.
  const jumps: number[] = [];
  for (let k = 1; k < snaps.length; k++) {
    for (let i = 0; i < c.trophies.length; i++) {
      const a = snaps[k - 1].rates[i];
      const b = snaps[k].rates[i];
      if (a === null || b === null || a >= 99.99 || b >= 99.99) continue;
      jumps.push(logit(b) - logit(a));
    }
  }
  let jumpMean: number;
  let jumpSd: number;
  if (jumps.length >= 5) {
    jumpMean = median(jumps);
    const mad = median(jumps.map((d) => Math.abs(d - jumpMean)));
    jumpSd = Math.min(1.5, Math.max(0.3, 1.4826 * mad));
  } else {
    // No refresh ever seen: prior dilution of one refresh (new players have
    // earned almost nothing yet), sized by the tracked base.
    jumpMean = -Math.log(1 + NEW_PER_REFRESH_PRIOR / Math.max(1, nLast));
    jumpSd = JUMP_SD_PRIOR;
  }

  const out: Array<{ id: number; q: number[] }> = [];
  for (let i = 0; i < c.trophies.length; i++) {
    const t = c.trophies[i];
    const rLast = lastRead.rates[i];
    // Saturated: everyone tracked holds it; it stays 100%.
    if (rLast !== null && rLast >= 99.99) { out.push({ id: t.id, q: [100, 100, 100, 100, 100] }); continue; }
    // No earners yet anywhere in the history: rate can only rise; wide spread.
    // (A trophy whose rate merely collapsed to 0.01 after having earners goes
    // through the trend/mixture below instead.)
    const ever = c.history.some((r) => r.rates[i] !== null && (r.rates[i] as number) > 0);
    if (!ever || rLast === null) {
      out.push({ id: t.id, q: [0.01, 0.01, 0.05, 0.5, 2] });
      continue;
    }
    // Per-trophy distinct points (first occurrence of each value), last 48h.
    const pts: Array<{ u: number; y: number; w: number; t: number }> = [];
    const cutoff = hAt - FIT_WINDOW_H;
    for (const r of c.history) {
      const v = r.rates[i];
      if (v === null) continue;
      const hh = hoursSince(r.at);
      if (hh < cutoff && c.history.length > 5) continue;
      const y = logit(v);
      const prev = pts[pts.length - 1];
      if (prev && Math.abs(prev.y - y) < 1e-12) continue;
      pts.push({ u: uOf(r.at), y, w: Math.pow(0.5, Math.max(0, hAt - hh) / HALF_LIFE_H), t: hh });
    }
    const yLast = logit(rLast);
    if (pts.length >= 3) {
      // Trend branch. Fit on v = u - u(last read), so the basis is centred
      // at the present (well conditioned) and v_target is the extrapolation
      // distance. A concave quadratic captures saturation; a line captures
      // the recent slope. The median averages the two: the quadratic alone
      // under-shoots a still-rising curve, the line alone over-shoots it.
      const uLast = uOf(lastRead.at);
      const vT = uTarget - uLast;
      const vpts = pts.map((p) => ({ u: p.u - uLast, y: p.y, w: p.w }));
      let mu: number | null = null;
      let varPred = 0;
      let nFits = 0;
      let muQ: number | null = null;
      let muL: number | null = null;
      if (pts.length >= 5) {
        const fq = wls(vpts, (v) => [1, v, v * v], [1, vT, vT * vT]);
        if (fq !== null && fq.coef[2] <= 0) {
          muQ = fq.coef[0] + fq.coef[1] * vT + fq.coef[2] * vT * vT;
          const sigQ = Math.min(1.5, Math.max(0.1, fq.sigRes));
          varPred += sigQ * sigQ * (1 + Math.min(5, fq.leverage));
          nFits++;
        }
      }
      const fl = wls(vpts, (v) => [1, v], [1, vT]);
      if (fl !== null) {
        muL = fl.coef[0] + fl.coef[1] * vT;
        const sigL = Math.min(1.5, Math.max(0.1, fl.sigRes));
        varPred += sigL * sigL * (1 + Math.min(5, fl.leverage));
        nFits++;
      }
      if (muQ !== null && muL !== null) mu = QUAD_W * muQ + (1 - QUAD_W) * muL;
      else mu = muQ !== null ? muQ : muL;
      if (mu !== null && nFits > 0) {
        varPred /= nFits;
        mu = Math.min(yLast + 2.5, Math.max(yLast - 2.5, mu));
        mu = Math.min(9.21, Math.max(-9.21, mu));
        // Spread: the fit's prediction variance plus a modest random-walk
        // floor for structural surprises. The refresh jumps are the trend's
        // mechanism, not independent shocks on top of it, so they are not
        // added again here (their in-sample deviation is inside sigRes).
        const s = Math.min(1.5, Math.max(0.35, Math.sqrt(
          varPred + SIGMA_TREND * SIGMA_TREND * horizonD,
        )));
        const q = Z.map((z) => round4(invLogit(mu + z * s)));
        // Enforce non-decreasing and (0, 100) after rounding.
        for (let k = 1; k < q.length; k++) if (q[k] < q[k - 1]) q[k] = q[k - 1];
        out.push({ id: t.id, q: q.map((v) => Math.min(100, Math.max(0.01, v))) });
        continue;
      }
    }
    // Flat-history branch: Poisson mixture over refresh counts before target.
    const m = expectedRefreshes;
    const sig0 = SIGMA_RW * Math.sqrt(horizonD);
    const lo = -12;
    const hi = 10;
    // Poisson weights for j = 0..J.
    const J = Math.min(40, Math.ceil(m + 5 * Math.sqrt(m) + 3));
    const wPois: number[] = [];
    let wSum = 0;
    for (let j = 0; j <= J; j++) {
      const w = Math.exp(-m + j * Math.log(Math.max(m, 1e-9)) - lgamma(j + 1));
      wPois.push(w);
      wSum += w;
    }
    const mus: number[] = [];
    const sds: number[] = [];
    for (let j = 0; j <= J; j++) {
      mus.push(Math.max(yLast - MAX_DRIFT, yLast + j * jumpMean));
      sds.push(Math.sqrt(sig0 * sig0 + j * jumpSd * jumpSd));
    }
    const qs = LEVELS.map((tau) => {
      // Binary search the mixture CDF for tau.
      let a = lo;
      let b = hi;
      for (let it = 0; it < 60; it++) {
        const mid = (a + b) / 2;
        let cdf = 0;
        for (let j = 0; j <= J; j++) cdf += (wPois[j] / wSum) * normCdf((mid - mus[j]) / sds[j]);
        if (cdf < tau) a = mid; else b = mid;
      }
      return round4(invLogit((a + b) / 2));
    });
    for (let k = 1; k < qs.length; k++) if (qs[k] < qs[k - 1]) qs[k] = qs[k - 1];
    out.push({ id: t.id, q: qs.map((v) => Math.min(100, Math.max(0.01, v))) });
  }
  out.sort((a, b) => a.id - b.id);
  return out;
}

/** ln Gamma via Lanczos (for Poisson weights). */
function lgamma(x: number): number {
  const p = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
  let z = x - 1;
  let a = p[0];
  for (let i = 1; i < 9; i++) a += p[i] / (z + i);
  const t = z + 7.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(a);
}

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  if (c.history.length === 0) throw new Error(`${file} has no history`);
  const trophies = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
