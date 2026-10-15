"""Protocolo de avaliação da Etapa 2.4 (ver PROTOCOLO.md). Score maior = mais anômalo.

Entradas comuns:
- scores:  DataFrame patch_id, <key>, score        (key = 'step' trimestral ou 'year' anual)
- labels:  saída de rotulos.quarterly_labels / annual_labels
- grade:   patch_id, block, split
"""
import numpy as np
import pandas as pd

import config
import rotulos
import utils

STEP_INDEX = {s: i for i, s in enumerate(config.STEPS)}


def make_task(df, tipo=None):
    """Positivos: evento (do tipo, se dado) >= EVENT_MIN_FRAC. Negativos: patch-passos 'limpo'.
    Ambíguos e eventos de outros tipos ficam de fora. Sem rebalancear: prevalência real."""
    pos = (df['estado'] == 'evento') if tipo is None else (df[f'frac_{tipo}'] >= config.EVENT_MIN_FRAC)
    keep = pos | (df['estado'] == 'limpo')
    return df[keep].assign(y=pos[keep].astype(int).values)


def _merge(scores, labels, grade, key):
    df = labels.merge(scores[['patch_id', key, 'score']], on=['patch_id', key], how='inner')
    return df.merge(grade[['patch_id', 'block', 'split']], on='patch_id', how='left')


def evaluate(scores, labels, grade, key='step', split='test', periodo='teste',
             tipos=config.EVENT_TYPES, fprs=(0.01, 0.05), ks=(100, 500), n_boot=300):
    """Uma linha por tarefa ('todos' + cada tipo com positivos), com IC 95% por bootstrap de blocos."""
    df = _merge(scores, labels, grade, key)
    if split is not None:
        df = df[df['split'] == split]
    if periodo is not None:
        df = df[df['periodo'] == periodo]
    rows = []
    for tipo in [None, *tipos]:
        t = make_task(df, tipo)
        y, s, b = t['y'].values, t['score'].values, t['block'].values
        if y.sum() == 0 or y.sum() == len(y):
            continue
        lo, hi = utils.block_bootstrap(b, y, s, utils.auroc, n=n_boot)
        plo, phi = utils.block_bootstrap(b, y, s, utils.avg_precision, n=n_boot)
        r = {'tarefa': tipo or 'todos', 'n': len(y), 'n_pos': int(y.sum()), 'prevalencia': y.mean(),
             'AUROC': utils.auroc(y, s), 'AUROC_lo': lo, 'AUROC_hi': hi,
             'AUPRC': utils.avg_precision(y, s), 'AUPRC_lo': plo, 'AUPRC_hi': phi}
        r.update({f'TPR@{f:.0%}FPR': v for f, v in utils.tpr_at_fpr(y, s, fprs).items()})
        r.update({f'prec@{k}': utils.precision_at_k(y, s, k) for k in ks if k <= len(y)})
        rows.append(r)
    return pd.DataFrame(rows)


def threshold_at_fpr(scores, labels, grade, fpr=0.01, key='step', split='val'):
    """Limiar operacional: quantil dos scores 'limpo' da validação. Nunca escolher no teste."""
    df = _merge(scores, labels, grade, key)
    clean = df[(df['split'] == split) & (df['estado'] == 'limpo')]['score']
    return float(np.quantile(clean, 1 - fpr))


def _event_steps(eventos):
    """Eventos vetoriais com data (DETER, PRODES) -> passo trimestral do evento."""
    ev = eventos[eventos['event_id'].fillna('') != ''].copy()
    ev['step'] = rotulos.step_of(ev['date_end'])
    ev = ev[ev['step'].isin(STEP_INDEX)]
    ev['i_event'] = ev['step'].map(STEP_INDEX)
    return ev


def _event_hits(scores, eventos, thr, lags, min_frac=0.0):
    """Para cada (evento, patch, lag) em `lags`: o score do patch passou do limiar no passo evento+lag?"""
    ev = _event_steps(eventos)
    ev = ev[ev['frac'] >= min_frac] if min_frac > 0 else ev
    sc = scores.assign(i=scores['step'].map(STEP_INDEX))
    rows = [ev.assign(lag=lag, i=ev['i_event'] + lag) for lag in lags]
    x = pd.concat(rows).merge(sc[['patch_id', 'i', 'score']], on=['patch_id', 'i'], how='inner')
    x['hit'] = x['score'] > thr
    return x


def event_recall(scores, eventos, poligonos, thr, max_lag=2, size_bins=(0, 5, 25, 100, np.inf)):
    """Fração dos polígonos detectados: algum patch que o polígono toca passa do limiar entre o
    trimestre do evento e `max_lag` trimestres depois. Por fonte, tipo e tamanho (ha).
    `eventos` = formato longo (evento x patch); `poligonos` = rotulos.event_table (com area_ha)."""
    x = _event_hits(scores, eventos, thr, range(0, max_lag + 1))
    det = x.groupby('event_id')['hit'].any()
    out = poligonos[poligonos['event_id'].isin(det.index)].copy()
    out['detectado'] = out['event_id'].map(det).values
    out['tamanho_ha'] = pd.cut(out['area_ha'], size_bins).astype(str)
    return (out.groupby(['source', 'tipo', 'tamanho_ha'])['detectado']
            .agg(n='size', recall='mean').reset_index())


def lead_time(scores, eventos, labels, thr, window=4, min_frac=config.EVENT_MIN_FRAC):
    """Antecedência em relação ao DETER: primeiro trimestre (de -window a +window) em que um patch
    com >= min_frac do alerta passa do limiar, menos o trimestre do alerta. Negativo = antes do DETER.

    Antes do alerta, só contam patch-trimestres 'limpo' em `labels` (rótulos trimestrais): um disparo
    causado por outro evento anterior no mesmo patch (ex.: fogo um ano antes) não é antecedência.
    Atenção: com FPR f por patch-passo, um alerta com p patches tem chance ~1-(1-f)^(p*window) de um
    "aviso antecipado" por acaso. Compare sempre com `chance_rate` (mesma conta em patches limpos)."""
    deter = eventos[eventos['source'] == 'deter']
    x = _event_hits(scores, deter, thr, range(-window, window + 1), min_frac=min_frac)
    clean = labels.loc[labels['estado'] == 'limpo', ['patch_id', 'step']].assign(
        i=lambda d: d['step'].map(STEP_INDEX), limpo=True)
    x = x.merge(clean[['patch_id', 'i', 'limpo']], on=['patch_id', 'i'], how='left')
    x = x[(x['lag'] >= 0) | x['limpo'].fillna(False).astype(bool)]
    first = x[x['hit']].groupby('event_id')['lag'].min()
    meta = deter.drop_duplicates('event_id').set_index('event_id')
    out = meta.loc[first.index, ['tipo']].assign(lag_trimestres=first.values)
    return out.reset_index()


def chance_rate(scores, labels, thr, key='step'):
    """Taxa de passagem do limiar em patch-passos 'limpo': a linha de base do acaso para lead_time."""
    df = labels.merge(scores[['patch_id', key, 'score']], on=['patch_id', key])
    return float((df.loc[df['estado'] == 'limpo', 'score'] > thr).mean())
