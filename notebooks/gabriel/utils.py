"""Sementes, log de execuções e métricas (numpy-only, portadas de amazon_anomaly_firstpass/analyze.py)."""
import json
import platform
import random
import subprocess
from datetime import datetime
from importlib import metadata

import numpy as np

import config

KEY_PACKAGES = ['numpy', 'pandas', 'torch', 'terratorch', 'scikit-learn', 'scipy', 'rasterio',
                'xarray', 'zarr', 'geopandas', 'earthengine-api']


# ---------------- reprodutibilidade ----------------

def set_seed(seed=config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    return seed


def git_info():
    def run(*args):
        return subprocess.run(['git', *args], cwd=config.REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    try:
        return {'commit': run('rev-parse', 'HEAD'), 'branch': run('rev-parse', '--abbrev-ref', 'HEAD'),
                'dirty': bool(run('status', '--porcelain'))}
    except (OSError, subprocess.CalledProcessError):
        return {'commit': None, 'branch': None, 'dirty': None}


def package_versions(names=KEY_PACKAGES):
    out = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = None
    return out


def new_run(name, cfg=None, seed=config.SEED):
    """Cria results/<name>/<timestamp>/ e salva config, semente, commit e versões em run.json."""
    run_dir = config.RESULTS_DIR / name / datetime.now().strftime('%Y%m%d-%H%M%S')
    run_dir.mkdir(parents=True, exist_ok=True)
    info = {'name': name, 'seed': seed, 'created': datetime.now().isoformat(timespec='seconds'),
            'git': git_info(), 'platform': platform.platform(), 'python': platform.python_version(),
            'in_colab': config.IN_COLAB, 'packages': package_versions(),
            'config': cfg if cfg is not None else config.as_dict()}
    (run_dir / 'run.json').write_text(json.dumps(info, indent=2, default=str), encoding='utf-8')
    return run_dir


# ---------------- métricas (maior score = mais anômalo) ----------------

def auroc(y, s):
    # Mann-Whitney U com ranks médios para empates
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


def tpr_at_fpr(y, s, fpr=(0.01, 0.05)):
    """TPR no limiar que deixa passar `fpr` dos normais. Aceita um valor ou uma lista."""
    def one(f):
        thr = np.quantile(s[y == 0], 1 - f)
        return (s[y == 1] > thr).mean()
    return one(fpr) if np.isscalar(fpr) else {f: one(f) for f in fpr}


def precision_at_k(y, s, k):
    """Das k amostras mais anômalas, a fração que é evento real."""
    return y[np.argsort(-s)[:k]].mean()


def block_bootstrap(blocks, y, s, fn, n=300, seed=config.SEED, q=(2.5, 97.5)):
    """IC por bootstrap reamostrando blocos espaciais inteiros."""
    rng = np.random.default_rng(seed)
    ub = np.unique(blocks); idx = {u: np.where(blocks == u)[0] for u in ub}
    vals = []
    for _ in range(n):
        pick = np.concatenate([idx[u] for u in rng.choice(ub, len(ub))])
        if 0 < y[pick].sum() < len(pick):
            vals.append(fn(y[pick], s[pick]))
    return np.percentile(vals, q)
