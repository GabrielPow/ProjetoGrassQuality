"""First-pass: does AlphaEarth embedding change separate MapBiomas-confirmed disturbances
from normal land better than spectral change? Fully unsupervised scores; labels only for evaluation.
numpy-only (sklearn/scipy are blocked by Windows App Control on this machine).
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import STRATA, ANOMALY_STRATA, HIST, SEED

d = np.load('data/amazon_firstpass.npz', allow_pickle=True)
AE, S2, years, T, strat, split, block = d['AE'], d['S2'], d['years'], d['T'], d['strat'], d['split'], d['block']
s2_names = list(d['s2_names'])
jT = np.searchsorted(years, T)
rows = np.arange(len(T))


def at(arr, off):
    return arr[rows, jT + off]


# ---- representations: history mean (T-2, T-1), event year T, lagged T+1 ----
ae_hist = np.mean([at(AE, -k) for k in range(1, HIST + 1)], axis=0)
s2_hist = np.mean([at(S2, -k) for k in range(1, HIST + 1)], axis=0)
ae_T, ae_T1, s2_T, s2_T1 = at(AE, 0), at(AE, 1), at(S2, 0), at(S2, 1)

ok = ~(np.isnan(ae_hist).any(1) | np.isnan(ae_T).any(1) | np.isnan(ae_T1).any(1)
       | np.isnan(s2_hist).any(1) | np.isnan(s2_T).any(1) | np.isnan(s2_T1).any(1))
print(f'complete rows: {ok.sum()}/{len(ok)}  (dropped by stratum: '
      f'{dict(zip(*np.unique(strat[~ok], return_counts=True)))})')

y = np.isin(strat, ANOMALY_STRATA).astype(int)
train_norm = ok & (split == 'train') & (y == 0)


def unit(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def cos_change(cur, hist):
    return 1 - np.sum(unit(cur) * unit(hist), axis=1)


# per-feature scale of year-over-year change, estimated on train normals only
s2_scale = np.nanstd((s2_T - s2_hist)[train_norm], axis=0) + 1e-6
ndvi, nbr = s2_names.index('NDVI'), s2_names.index('NBR')


def knn_score(F, k=10):
    ref = F[train_norm]
    mu, sd = ref.mean(0), ref.std(0) + 1e-6
    ref, Fz = (ref - mu) / sd, (F - mu) / sd
    Fz = np.nan_to_num(Fz)
    out = np.empty(len(F))
    for i in range(0, len(F), 256):
        dist = np.sqrt(((Fz[i:i + 256, None, :] - ref[None]) ** 2).sum(-1))
        dist[np.isclose(dist, 0)] = np.inf          # exclude self-match for train points
        out[i:i + 256] = np.sort(dist, axis=1)[:, :k].mean(1)
    return out


SCORES = {
    # name: (family, score)  -- higher = more anomalous
    'AE cosine change (T)':       ('AE', cos_change(ae_T, ae_hist)),
    'AE cosine change (T+1)':     ('AE', cos_change(ae_T1, ae_hist)),
    'AE kNN on delta (T)':        ('AE', knn_score(ae_T - ae_hist)),
    'AE kNN static, no time (T)': ('AE', knn_score(ae_T)),
    'NDVI drop (T)':              ('S2', s2_hist[:, ndvi] - s2_T[:, ndvi]),
    'NBR drop (T)':               ('S2', s2_hist[:, nbr] - s2_T[:, nbr]),
    'S2 band distance (T)':       ('S2', np.linalg.norm((s2_T - s2_hist) / s2_scale, axis=1)),
    'S2 band distance (T+1)':     ('S2', np.linalg.norm((s2_T1 - s2_hist) / s2_scale, axis=1)),
    'S2 kNN on delta (T)':        ('S2', knn_score(s2_T - s2_hist)),
}


def auroc(y, s):
    # Mann-Whitney U with average ranks for ties
    order = np.argsort(s, kind='mergesort')
    ss = s[order]
    r = np.empty(len(s))
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def avg_precision(y, s):
    o = np.argsort(-s); yy = y[o]
    prec = np.cumsum(yy) / np.arange(1, len(yy) + 1)
    return (prec * yy).sum() / yy.sum()


def tpr_at_fpr(y, s, fpr=0.05):
    thr = np.quantile(s[y == 0], 1 - fpr)
    return (s[y == 1] > thr).mean()


def block_bootstrap(mask, ylab, s, fn, n=300):
    rng = np.random.default_rng(SEED)
    b = block[mask]; ub = np.unique(b); idx = {u: np.where(b == u)[0] for u in ub}
    yl, sc, vals = ylab[mask], s[mask], []
    for _ in range(n):
        pick = np.concatenate([idx[u] for u in rng.choice(ub, len(ub))])
        if 0 < yl[pick].sum() < len(pick):
            vals.append(fn(yl[pick], sc[pick]))
    return np.percentile(vals, [2.5, 97.5])


test = ok & (split == 'test')
TASKS = {
    'all anomalies vs all normals': (test, y),
    'deforest vs stable forest':    (test & np.isin(strat, [0, 2]), (strat == 2).astype(int)),
    'deforest vs stable pasture':   (test & np.isin(strat, [1, 2]), (strat == 2).astype(int)),
    'mining vs all normals':        (test & np.isin(strat, [0, 1, 3]), (strat == 3).astype(int)),
}

res = []
for task, (m, yl) in TASKS.items():
    for name, (fam, s) in SCORES.items():
        a = auroc(yl[m], s[m]); lo, hi = block_bootstrap(m, yl, s, auroc)
        res.append(dict(task=task, score=name, family=fam, n=int(m.sum()), n_pos=int(yl[m].sum()),
                        AUROC=a, AUROC_lo=lo, AUROC_hi=hi, AP=avg_precision(yl[m], s[m]),
                        TPR_at_5FPR=tpr_at_fpr(yl[m], s[m])))
res = pd.DataFrame(res)
res.to_csv('results/metrics.csv', index=False)
pd.set_option('display.width', 200)
for task, g in res.groupby('task', sort=False):
    print(f'\n== {task}  (n={g.n.iloc[0]}, positives={g.n_pos.iloc[0]})')
    print(g[['score', 'AUROC', 'AUROC_lo', 'AUROC_hi', 'AP', 'TPR_at_5FPR']].round(3).to_string(index=False))

# per event year, headline task
per_year = []
for t in np.unique(T):
    m = test & (T == t)
    for name in ['AE cosine change (T)', 'S2 band distance (T)', 'NDVI drop (T)']:
        per_year.append(dict(T=int(t), score=name, AUROC=auroc(y[m], SCORES[name][1][m])))
per_year = pd.DataFrame(per_year).pivot(index='T', columns='score', values='AUROC')
print('\n== AUROC by event year (all anomalies vs all normals)\n', per_year.round(3))
per_year.to_csv('results/auroc_by_year.csv')

# ---- figures ----
BLUE, ORANGE = '#2a78d6', '#eb6834'
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': GRID, 'axes.labelcolor': MUTED,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False,
                     'axes.spines.right': False})

fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
for ax, (task, (m, yl)) in zip(axes, TASKS.items()):
    for name, color in [('AE cosine change (T)', BLUE), ('S2 band distance (T)', ORANGE)]:
        s = SCORES[name][1][m]; yy = yl[m]
        o = np.argsort(-s)
        tpr = np.r_[0, np.cumsum(yy[o]) / yy.sum()]
        fpr = np.r_[0, np.cumsum(1 - yy[o]) / (1 - yy).sum()]
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name.split(' (')[0]}  AUROC {auroc(yy, s):.2f}")
    ax.plot([0, 1], [0, 1], color=GRID, lw=1, ls='--')
    ax.set_title(task, color=INK, fontsize=9); ax.set_xlabel('False positive rate')
    ax.legend(frameon=False, fontsize=7.5, loc='lower right'); ax.grid(color=GRID, lw=0.5)
axes[0].set_ylabel('True positive rate')
fig.suptitle('Held-out spatial blocks: AlphaEarth embedding change vs Sentinel-2 spectral change', color=INK)
fig.tight_layout(); fig.savefig('results/roc.png', dpi=150)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for ax, name, color in [(axes[0], 'AE cosine change (T)', BLUE), (axes[1], 'S2 band distance (T)', ORANGE)]:
    s = SCORES[name][1]
    data = [s[test & (strat == k)] for k in STRATA]
    ax.boxplot(data, orientation='horizontal', widths=0.5, patch_artist=True, showfliers=False,
               medianprops=dict(color=INK), boxprops=dict(facecolor=color, alpha=0.35, edgecolor=color),
               whiskerprops=dict(color=color), capprops=dict(color=color))
    ax.set_yticks(range(1, len(STRATA) + 1), [STRATA[k] for k in STRATA])
    ax.set_title(name, color=INK, fontsize=9); ax.set_xlabel('anomaly score'); ax.grid(axis='x', color=GRID, lw=0.5)
fig.tight_layout(); fig.savefig('results/score_by_stratum.png', dpi=150)

json.dump({'n_complete': int(ok.sum()), 'n_test': int(test.sum()),
           'n_train_normals': int(train_norm.sum())}, open('results/summary.json', 'w'))
