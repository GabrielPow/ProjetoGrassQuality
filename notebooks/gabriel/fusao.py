"""Fusão de scores (PLANO_JUNTAR_DEEP_LEARNING.md, Fase 3; regras em PROTOCOLO.md, seção 6).

Cada método é padronizado em relação aos scores das amostras NORMAIS de validação (`val_normal`), sem rótulos
de anomalia:
- principal: z-score robusto, z = (score − mediana) / IQR; score combinado = média dos z;
- ablações: percentil p = fração dos normais de validação abaixo do score; média ou máximo dos p.

Funciona no patch (vetores (n,)) e na célula (matrizes (n, 49)); na célula, o score do patch é o máximo das
células depois da fusão. NaN = método sem score naquela célula/amostra; a fusão usa os métodos disponíveis.
"""
import numpy as np
import pandas as pd

RULES = {'z_media': 'média dos z-scores robustos',
         'p_media': 'média dos percentis',
         'p_max': 'máximo dos percentis'}
PRINCIPAL = 'z_media'


class Ref:
    """Distribuição de referência de um método: scores das amostras normais de validação."""

    def __init__(self, ref_scores):
        r = np.asarray(ref_scores, dtype=np.float64).ravel()
        r = np.sort(r[~np.isnan(r)])
        assert len(r) > 10, 'poucos scores normais de validação para a referência'
        self.sorted = r
        self.med = np.median(r)
        q1, q3 = np.percentile(r, [25, 75])
        self.iqr = (q3 - q1) if q3 > q1 else (r.std() + 1e-12)

    def z(self, x):
        return (np.asarray(x, np.float64) - self.med) / self.iqr

    def p(self, x):
        x = np.asarray(x, np.float64)
        out = np.searchsorted(self.sorted, x, side='right') / len(self.sorted)
        return np.where(np.isnan(x), np.nan, out)


def fuse(scores, refs, rule=PRINCIPAL):
    """scores: {método: array}; refs: {método: Ref}. Devolve o array combinado (mesma forma)."""
    if rule == 'z_media':
        stack = np.stack([refs[m].z(v) for m, v in scores.items()])
        agg = np.nanmean
    elif rule in ('p_media', 'p_max'):
        stack = np.stack([refs[m].p(v) for m, v in scores.items()])
        agg = np.nanmean if rule == 'p_media' else np.nanmax
    else:
        raise ValueError(rule)
    with np.errstate(all='ignore'), _quiet():
        return agg(stack, axis=0)


def patch_from_cells(fused_cells):
    """(n, 49) -> máximo (principal) e média das células, ignorando NaN."""
    with np.errstate(all='ignore'), _quiet():
        return np.nanmax(fused_cells, axis=1), np.nanmean(fused_cells, axis=1)


def fuse_patch(df, methods, ref_mask, name, rules=tuple(RULES)):
    """J1: fusão de scores de patch já agregados. df tem uma coluna por método; ref_mask = linhas val_normal."""
    refs = {m: Ref(df.loc[ref_mask, m]) for m in methods}
    sc = {m: df[m].values for m in methods}
    return pd.DataFrame({f'{name} [{r}]': fuse(sc, refs, r) for r in rules}, index=df.index)


def fuse_cells(cells, methods, ref_mask, name, rules=tuple(RULES)):
    """J2: fusão célula a célula. cells = {método: DataFrame (n, 49)} com o mesmo índice; ref = todas as células
    das linhas val_normal. Devolve os scores de patch (máximo e média das células) por regra."""
    refs = {m: Ref(cells[m].values[ref_mask]) for m in methods}
    sc = {m: cells[m].values for m in methods}
    out = {}
    for r in rules:
        mx, mean = patch_from_cells(fuse(sc, refs, r))
        out[f'{name} [{r}]'] = mx
        out[f'{name} [{r}] [média]'] = mean
    idx = next(iter(cells.values())).index
    return pd.DataFrame(out, index=idx)


class _quiet:
    """Silencia o aviso 'All-NaN slice' do numpy (célula sem nenhum método disponível vira NaN, como esperado)."""

    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter('ignore', RuntimeWarning)

    def __exit__(self, *a):
        self._w.__exit__(*a)
