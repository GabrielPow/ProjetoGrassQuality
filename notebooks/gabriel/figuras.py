"""Figuras e tabelas da apresentação (PLANO_JUNTAR_DEEP_LEARNING.md, Fase 6), reaproveitáveis nos mapas da Etapa 8.

Só lê o que já está salvo: cubo da sub-AOI, células e tokens do TerraMind, checkpoints e `results/`. Nada de Earth
Engine nem treino. Cada figura também é salva em `results/apresentacao/<nome>.png`.
"""
import glob
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle

import config
import embeddings
import features_originais as fo

OUT = config.RESULTS_DIR / 'apresentacao'
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e1e0d9'
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#8a5cd1', '#d1495b', '#6b7a8f']
SPLIT_COR = {'train': '#2a78d6', 'val': '#eda100', 'test': '#d1495b'}

# exemplos fixos (todos em blocos `val`, que formam o teste limitado; PROTOCOLO.md, seção 7)
EXEMPLOS = {'desmatamento': ('tapajos_252_4125', '2024Q3'),
            'fogo': ('tapajos_244_4120', '2024Q3'),
            'mineracao': ('tapajos_245_4122', '2023Q1')}

# colunas dos scores salvos -> nome curto nos slides
FUSAO = 'Fusão J2 | AE + Original IForest [z_media]'
METODOS = {'Transformer AE | masked': 'AE (TerraMind)',
           'Transformer AE-ORIG | masked': 'AE (features originais)',
           'Transformer AE-TM+ORIG | masked': 'AE (TerraMind + originais)',
           FUSAO: 'Fusão de scores (AE + IForest orig.)',
           'Original | Deep SVDD (20 features + delta)': 'Deep SVDD (features orig.)',
           'TerraMind S2S1 | kNN no delta': 'kNN no delta (TerraMind)',
           'Espectral | queda de NDVI': 'Queda de NDVI'}


def estilo():
    plt.rcParams.update({'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': MUTED, 'xtick.color': MUTED,
                         'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titlecolor': INK, 'figure.dpi': 100})


def _fim(fig, nome):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f'{nome}.png', dpi=130, bbox_inches='tight')
    plt.show()


def _ultimo(nome):
    runs = sorted(glob.glob(str(config.RESULTS_DIR / nome / '*')))
    assert runs, f'nenhum run em results/{nome}'
    return Path(runs[-1])


# ---------------- dados ----------------

def contexto():
    import zarr
    cube = zarr.open_group(str(config.CUBE_DIR / config.CUBE_PATH_NAME), mode='r')
    pids, steps = list(cube.attrs['patch_id']), list(cube.attrs['steps'])
    d = config.LABELS_DIR / 'tapajos'
    grade = pd.read_csv(d / 'grade.csv')
    g = grade.set_index('patch_id').loc[pids]
    emb = config.EMBEDDINGS_DIR / 'tapajos_sub.npz'
    return SimpleNamespace(
        cube=cube, pids=pids, steps=steps, grade=grade, g=g,
        lin=(g['row'].max() - g['row']).values, col=(g['col'] - g['col'].min()).values,   # posição no mosaico (topo = norte)
        trim=pd.read_csv(d / 'rotulos_trimestrais.csv.gz'), anual=pd.read_csv(d / 'rotulos_anuais.csv.gz'),
        cells=zarr.open_group(str(embeddings.cells_path(emb)), mode='r'),
        tokens=zarr.open_group(str(emb.with_name(emb.stem + '_tokens.zarr')), mode='r'))


def _i(c, pid):
    return c.pids.index(pid)


def _t(c, step, delta=0):
    return c.steps.index(step) + delta


def s2(c, pid, step, delta=0):
    i, t = _i(c, pid), _t(c, step, delta)
    return c.cube['s2'][i, t], c.cube['s2_nobs'][i, t]


def rgb(x, ganho=3.5):
    """(12, H, W) DN -> (H, W, 3) em [0, 1], bandas B4, B3, B2."""
    return np.clip(x[[3, 2, 1]].astype(np.float32).transpose(1, 2, 0) / 10000 * ganho, 0, 1)


def indices(x):
    ndvi, gndvi, bsi = fo._indices(x[None].astype(np.float32) / 10000)
    return ndvi[0], gndvi[0], bsi[0]


def _grade_celulas(ax, cor='white', lw=0.5):
    for k in range(1, config.CELL_GRID):
        p = k * 32 - 0.5
        ax.axhline(p, color=cor, lw=lw, alpha=0.7); ax.axvline(p, color=cor, lw=lw, alpha=0.7)


def _evento(c, pid, step):
    r = c.trim[(c.trim['patch_id'] == pid) & (c.trim['step'] == step)]
    if r.empty:
        return '—'
    r = r.iloc[0]
    tipos = [f'{k[5:]} {r[k]:.0%}' for k in r.index if k.startswith('frac_') and r[k] >= config.EVENT_MIN_FRAC]
    return ', '.join(tipos) if tipos else r['estado']


def _off(*axes):
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)


# ---------------- diagramas ----------------

def _caixa(ax, x, y, w, h, texto, cor, fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.06', fc=cor, ec='none', alpha=0.9))
    ax.text(x + w / 2, y + h / 2, texto, ha='center', va='center', fontsize=fs, color='white', wrap=True)


def _seta(ax, x0, y0, x1, y1):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='->', color=MUTED, lw=1.4))


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.6); ax.axis('off')
    ax.text(0.1, 4.35, 'Implementação A (notebooks/)', color=CAT[1], fontsize=10, weight='bold')
    ax.text(0.1, 1.95, 'Implementação B (notebooks/gabriel/)', color=CAT[0], fontsize=10, weight='bold')
    _caixa(ax, 0.1, 3.0, 2.3, 1.1, 'Sentinel-2\n(composição)', CAT[1])
    _caixa(ax, 2.9, 3.0, 2.6, 1.1, 'Índices e textura\nNDVI, GNDVI, BSI,\ndesvios, Fourier', CAT[1])
    _caixa(ax, 0.1, 0.6, 2.3, 1.1, 'Sentinel-2 + S1\ntrimestral 2019–2025', CAT[0])
    _caixa(ax, 2.9, 0.6, 2.6, 1.1, 'TerraMind Small\n(congelado)', CAT[0])
    _caixa(ax, 6.1, 1.8, 2.4, 1.1, 'Célula de 320 m\n20 features + 384-d\npor trimestre', MUTED)
    _caixa(ax, 9.0, 3.0, 2.0, 1.1, 'Caminho 1\ndetectores +\nfusão de scores', CAT[2])
    _caixa(ax, 9.0, 0.6, 2.0, 1.1, 'Caminho 2\nTransformer AE,\nentrada combinada', CAT[2])
    _caixa(ax, 11.4, 1.8, 1.5, 1.1, 'Score de\nanomalia\n→ mapa', INK)
    for a in [(2.4, 3.55, 2.9, 3.55), (2.4, 1.15, 2.9, 1.15), (5.5, 3.55, 6.1, 2.6), (5.5, 1.15, 6.1, 2.1),
              (8.5, 2.6, 9.0, 3.4), (8.5, 2.1, 9.0, 1.3), (11.0, 3.4, 11.4, 2.6), (11.0, 1.3, 11.4, 2.1)]:
        _seta(ax, *a)
    _fim(fig, '03_pipeline')


def fig_ae():
    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.set_xlim(0, 13); ax.set_ylim(0, 3.6); ax.axis('off')
    for k in range(9):
        cor = INK if k == 8 else CAT[0]
        _caixa(ax, 0.1 + k * 0.62, 2.4, 0.55, 0.8, 'máscara' if k == 8 else f't−{8 - k}', cor, fs=7)
    ax.text(2.9, 3.35, 'sequência de 9 trimestres de uma célula: Z(t−8) … Z(t)', ha='center', color=MUTED, fontsize=9)
    _caixa(ax, 0.1, 0.5, 2.6, 1.2, 'Projeção linear\n+ codificação de ano\ne trimestre (sen/cos)', CAT[0])
    _caixa(ax, 3.3, 0.5, 2.6, 1.2, 'Transformer encoder\n4 camadas, 4 cabeças\nd = 128, máscara p/ falta', CAT[0])
    _caixa(ax, 6.5, 0.5, 2.0, 1.2, 'Gargalo\n128 → 64 → 128', CAT[2])
    _caixa(ax, 9.1, 0.5, 1.8, 1.2, 'Reconstrução\nẐ(t)', CAT[0])
    _caixa(ax, 11.4, 0.5, 1.5, 1.2, 'Score =\n‖Ẑ(t) − Z(t)‖²', INK)
    _seta(ax, 2.9, 2.35, 1.4, 1.75)
    for a in [(2.7, 1.1, 3.3, 1.1), (5.9, 1.1, 6.5, 1.1), (8.5, 1.1, 9.1, 1.1), (10.9, 1.1, 11.4, 1.1)]:
        _seta(ax, *a)
    ax.text(9.0, 2.75, 'O passo t fica escondido: o modelo prevê Z(t) só com o contexto,\n'
                       'então não consegue "copiar" uma anomalia. Treino só com sequências normais.',
            fontsize=9, color=INK)
    _fim(fig, '11_arquitetura_ae')


# ---------------- área e imagens ----------------

def fig_area(c):
    import json
    ro = pd.read_csv(config.LABELS_DIR / 'norte_ro' / 'grade.csv')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={'width_ratios': [1.25, 1]})
    ax = axes[0]
    for sp, cor in SPLIT_COR.items():
        m = c.grade['split'] == sp
        ax.scatter(c.grade.loc[m, 'lon'], c.grade.loc[m, 'lat'], s=1.2, color=cor, label=f'{sp} ({m.sum()})')
    lo, la = c.g['lon'], c.g['lat']
    pad = 0.012
    ax.add_patch(Rectangle((lo.min() - pad, la.min() - pad), lo.max() - lo.min() + 2 * pad, la.max() - la.min() + 2 * pad,
                           fill=False, ec=INK, lw=2))
    ax.text(lo.max() + 0.05, la.max(), 'sub-AOI\n16 × 16 patches\n(única processada)', fontsize=9, va='top')
    ax.set_title('Tapajós / BR-163 (PA): 9.801 patches de 2,24 km, blocos de 4 × 4 por partição')
    ax.legend(markerscale=6, frameon=False, fontsize=8, loc='lower left'); ax.set_aspect('equal')
    ax2 = axes[1]
    ax2.scatter(ro['lon'], ro['lat'], s=1.2, color='#b9b7af')
    ax2.set_title('Norte de Rondônia: 9.644 patches\n(só teste de generalização; ainda não processado)')
    ax2.set_aspect('equal')
    _fim(fig, '04_area')


def fig_patch_celulas(c, pid=EXEMPLOS['desmatamento'][0], step='2023Q3'):
    x, _ = s2(c, pid, step)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    axes[0].imshow(rgb(x)); axes[0].set_title(f'Patch 224 × 224 px (10 m) = 2,24 km\n{pid}, {step}')
    axes[1].imshow(rgb(x))
    for k in range(1, 14):
        axes[1].axhline(k * 16 - 0.5, color='white', lw=0.3); axes[1].axvline(k * 16 - 0.5, color='white', lw=0.3)
    axes[1].set_title('Tokens do TerraMind: 14 × 14\n(16 px = 160 m cada)')
    axes[2].imshow(rgb(x)); _grade_celulas(axes[2], lw=1.2)
    axes[2].set_title('Células de análise: 7 × 7\n(2 × 2 tokens = 320 m)')
    _off(*axes)
    _fim(fig, '04_patch_tokens_celulas')


def fig_serie(c, pid, step_evento, n=8, nome='05_serie'):
    t0 = _t(c, step_evento) - n + 2
    passos = c.steps[t0:t0 + n]
    fig, axes = plt.subplots(2, n, figsize=(1.75 * n, 4.1))
    for k, st in enumerate(passos):
        x, nobs = s2(c, pid, st)
        axes[0, k].imshow(rgb(x))
        if (nobs == 0).any():
            axes[0, k].imshow(np.ma.masked_where(nobs > 0, np.ones_like(nobs)), cmap='autumn', alpha=0.6)
        vv = c.cube['s1'][_i(c, pid), _t(c, st), 0].astype(np.float32)
        axes[1, k].imshow(vv, cmap='gray', vmin=-22, vmax=-2)
        ev = _evento(c, pid, st)
        cor = CAT[5] if ev not in ('limpo', '—', 'ambiguo') else MUTED
        axes[0, k].set_title(f'{st}\n{ev}', fontsize=8, color=cor)
    axes[0, 0].set_ylabel('Sentinel-2\nRGB', fontsize=9); axes[1, 0].set_ylabel('Sentinel-1\nVV (dB)', fontsize=9)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f'{pid}: 8 trimestres. Laranja = pixel sem observação limpa. Vermelho = evento rotulado.', fontsize=10)
    fig.tight_layout()
    _fim(fig, nome)


def fig_antes_depois(c, tipo):
    pid, st = EXEMPLOS[tipo]
    (xa, _), (xd, _) = s2(c, pid, st, -4), s2(c, pid, st, +1)
    na, _, ba = indices(xa); nd, _, bd = indices(xd)
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.4))
    axes[0, 0].imshow(rgb(xa)); axes[0, 0].set_title(f'antes: {c.steps[_t(c, st, -4)]}')
    axes[0, 1].imshow(rgb(xd)); axes[0, 1].set_title(f'depois: {c.steps[_t(c, st, 1)]}')
    for ax, v, nome, cm, lim in [(axes[1, 0], na, 'NDVI antes', 'RdYlGn', (0, 0.9)), (axes[1, 1], nd, 'NDVI depois', 'RdYlGn', (0, 0.9)),
                                 (axes[0, 2], ba, 'BSI antes (solo exposto)', 'YlOrBr', (-0.4, 0.2)),
                                 (axes[0, 3], bd, 'BSI depois', 'YlOrBr', (-0.4, 0.2))]:
        im = ax.imshow(v, cmap=cm, vmin=lim[0], vmax=lim[1]); ax.set_title(nome)
    im = axes[1, 2].imshow(nd - na, cmap='RdBu', vmin=-0.6, vmax=0.6); axes[1, 2].set_title('ΔNDVI (depois − antes)')
    fig.colorbar(im, ax=axes[1, 2], fraction=0.046)
    axes[1, 3].axis('off')
    axes[1, 3].text(0, 0.85, f'{tipo.upper()}\n\npatch {pid}\nevento em {st}:\n{_evento(c, pid, st)} do patch\n\n'
                             f'"antes" = mesmo trimestre do ano\nanterior (evita confundir com\nestação seca)',
                    va='top', fontsize=10)
    _off(*[a for a in axes.ravel() if a is not axes[1, 3]])
    fig.tight_layout()
    _fim(fig, f'06_antes_depois_{tipo}')


def fig_features(c, tipo='desmatamento'):
    pid, st = EXEMPLOS[tipo]
    i = _i(c, pid)
    F = {}
    for rot, d in [('antes', -4), ('depois', 1)]:
        t = _t(c, st, d)
        F[rot] = fo.original_features(c.cells['spec_cells'][i, t].astype(np.float32),
                                      c.cells['orig_cells'][i, t].astype(np.float32))
    sel = ['NDVI_media', 'NDVI_desvio', 'GNDVI_media', 'BSI_media', 'B8_desvio', 'FFT_NDVI_anel4']
    fig, axes = plt.subplots(2, len(sel), figsize=(14, 5))
    for k, nome in enumerate(sel):
        j = fo.ORIGINAL_NAMES.index(nome)
        a, b = F['antes'][:, j].reshape(7, 7), F['depois'][:, j].reshape(7, 7)
        lo, hi = np.nanmin([a, b]), np.nanmax([a, b])
        for r, v in enumerate([a, b]):
            im = axes[r, k].imshow(v, cmap='viridis', vmin=lo, vmax=hi)
        axes[0, k].set_title(nome.replace('_', ' '), fontsize=9)
        fig.colorbar(im, ax=axes[:, k], fraction=0.05, shrink=0.8)
    axes[0, 0].set_ylabel('antes', fontsize=10); axes[1, 0].set_ylabel('depois', fontsize=10)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f'Grade 7 × 7 de células, {pid}', fontsize=10)
    _fim(fig, '08_features_celulas')


def fig_fourier(c, tipo='desmatamento'):
    pid, st = EXEMPLOS[tipo]
    i = _i(c, pid)
    ta, td = _t(c, st, -4), _t(c, st, 1)
    ndvi = {}
    for rot, t in [('antes', ta), ('depois', td)]:
        ndvi[rot] = fo._to_cells(indices(c.cube['s2'][i, t])[0][None])[0]          # (49, 32, 32)
    j = int(np.argmax(np.abs(ndvi['depois'].mean((1, 2)) - ndvi['antes'].mean((1, 2)))))
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.4))
    for k, rot in enumerate(['antes', 'depois']):
        axes[k].imshow(ndvi[rot][j], cmap='RdYlGn', vmin=0, vmax=0.9); axes[k].set_title(f'NDVI da célula {j}, {rot}')
    a = (ndvi['depois'][j] - ndvi['depois'][j].mean()) * fo._HANN
    P = np.abs(np.fft.fftshift(np.fft.fft2(a))) ** 2 / (fo._HANN ** 2).sum()
    axes[2].imshow(np.log10(P + 1e-8), cmap='magma'); axes[2].set_title('log |FFT|² depois\n(janela de Hann; anéis)')
    for r in fo.RING_EDGES[1:-1]:
        axes[2].add_patch(Circle((16, 16), r * 32, fill=False, ec='white', lw=0.8))
    _off(*axes[:3])
    m = np.ones((1, 32, 32), bool)
    pa, pdp = fo.ring_energy(ndvi['antes'][j][None], m)[0], fo.ring_energy(ndvi['depois'][j][None], m)[0]
    x = np.arange(4)
    axes[3].bar(x - 0.2, pa, 0.4, color=CAT[0], label='antes'); axes[3].bar(x + 0.2, pdp, 0.4, color=CAT[1], label='depois')
    axes[3].set_xticks(x, ['anel 1\n(baixa)', '2', '3', '4\n(alta)']); axes[3].set_ylabel('log10 energia')
    axes[3].legend(frameon=False, fontsize=8); axes[3].set_title('Perfil radial (4 números/célula)')
    ch = (pdp[2:] - pa[2:]) ** 2
    axes[4].bar(['anel 3', 'anel 4'], ch, color=CAT[2]); axes[4].set_title(f'Mudança de textura\nΣ(Δ log E)² = {ch.sum():.2f}')
    fig.tight_layout()
    _fim(fig, '09_fourier')


def fig_terramind(c, tipo='desmatamento'):
    pid, st = EXEMPLOS[tipo]
    i, ta, td = _i(c, pid), _t(c, st, -4), _t(c, st, 1)
    def cosd(a, b):
        a, b = a.astype(np.float32), b.astype(np.float32)
        return 1 - (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-9)
    tok = cosd(c.tokens['tok_s2s1'][i, ta], c.tokens['tok_s2s1'][i, td]).reshape(14, 14)
    cel = cosd(c.cells['c_s2s1'][i, ta], c.cells['c_s2s1'][i, td]).reshape(7, 7)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.7))
    axes[0].imshow(rgb(s2(c, pid, st, -4)[0])); axes[0].set_title(f'antes ({c.steps[ta]})')
    axes[1].imshow(rgb(s2(c, pid, st, 1)[0])); axes[1].set_title(f'depois ({c.steps[td]})')
    im = axes[2].imshow(tok, cmap='magma', vmin=0); axes[2].set_title('mudança do embedding\npor token (14 × 14)')
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    im = axes[3].imshow(cel, cmap='magma', vmin=0); axes[3].set_title('mudança por célula\n(7 × 7, unidade do modelo)')
    fig.colorbar(im, ax=axes[3], fraction=0.046)
    _off(*axes)
    fig.suptitle('Distância cosseno entre os embeddings TerraMind (S2 + S1) de antes e depois', fontsize=10)
    _fim(fig, '10_terramind')


# ---------------- treino ----------------

def fig_curvas():
    e6, e9 = _ultimo('etapa6_transformer_ae_tapajos_sub'), _ultimo('etapa_ae_combinado_tapajos_sub')
    cur = {'AE (TerraMind)': e6 / 'curva_masked.csv', 'AE (features originais)': e9 / 'curva_orig.csv',
           'AE (TerraMind + originais)': e9 / 'curva_tm_orig.csv'}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.3))
    for ax, (nome, p), cor in zip(axes, cur.items(), CAT):
        cv = pd.read_csv(p)
        ax.plot(cv['epoca'], cv['treino'], color=cor, ls='--', lw=1.5, label='treino')
        ax.plot(cv['epoca'], cv['val'], color=cor, lw=2, label='validação (erro em t)')
        b = cv['val'].idxmin()
        ax.scatter(cv.loc[b, 'epoca'], cv.loc[b, 'val'], color=INK, zorder=3, s=18)
        ax.set_title(f'{nome}\nmelhor época {int(cv.loc[b, "epoca"])} de {len(cv)}', fontsize=9.5)
        ax.set_xlabel('época'); ax.grid(axis='y', color=GRID, lw=0.5)
    axes[0].set_ylabel('erro quadrático (padronizado)'); axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _fim(fig, '11_curvas')


# ---------------- teste final ----------------

def tabela_scores():
    """Scores de patch (todas as amostras da sub-AOI) dos 7 métodos dos slides, dos runs mais recentes."""
    a = pd.read_csv(_ultimo('etapa_ae_combinado_tapajos_sub') / 'scores_trimestrais.csv.gz')
    b = pd.read_csv(_ultimo('etapa_fusao_tapajos_sub') / 'scores_trimestrais.csv.gz')
    cols_a = [m for m in METODOS if m in a.columns]
    cols_b = [m for m in METODOS if m in b.columns and m not in cols_a]
    return a[['patch_id', 'step'] + cols_a].merge(b[['patch_id', 'step'] + cols_b], on=['patch_id', 'step'])


def _tarefa(tab, col, c, split='val', periodo='teste', tipo=None):
    import avaliacao
    sc = tab[['patch_id', 'step']].assign(score=tab[col].values).dropna(subset=['score'])
    df = avaliacao._merge(sc, c.trim, c.grade, 'step')
    df = df[(df['split'] == split) & (df['periodo'] == periodo)]
    return avaliacao.make_task(df, tipo)


def teste_limitado(c, tab, n_boot=300):
    """Teste limitado (PROTOCOLO.md, seção 7): blocos val da sub-AOI em 2023–2025."""
    import avaliacao
    out = []
    for col, nome in METODOS.items():
        sc = tab[['patch_id', 'step']].assign(score=tab[col].values).dropna(subset=['score'])
        r = avaliacao.evaluate(sc, c.trim, c.grade, key='step', split='val', periodo='teste', n_boot=n_boot)
        out.append(r.assign(metodo=nome))
    return pd.concat(out, ignore_index=True)


def diagnostico_treino():
    """Mesmas métricas no split train (166 eventos, holdout só temporal), dos runs salvos."""
    a = pd.read_csv(_ultimo('etapa_ae_combinado_tapajos_sub') / 'metricas.csv')
    a = a[(a['agregacao'] == 'máximo') & (a['resolucao'] == 'trimestral')]
    a = a[a['metodo'].isin(METODOS)].assign(metodo=lambda d: d['metodo'].map(METODOS))
    return a


def fig_roc(c, tab):
    from sklearn.metrics import roc_curve
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    for (col, nome), cor in zip(METODOS.items(), CAT):
        t = _tarefa(tab, col, c)
        fpr, tpr, _ = roc_curve(t['y'], t['score'])
        import utils
        ax.plot(fpr, tpr, color=cor, lw=1.8, label=f'{nome} ({utils.auroc(t["y"].values, t["score"].values):.2f})')
    ax.plot([0, 1], [0, 1], color=MUTED, ls='--', lw=1); ax.text(0.6, 0.52, 'acaso', color=MUTED, fontsize=8, rotation=38)
    ax.set_xlabel('taxa de falso positivo'); ax.set_ylabel('taxa de verdadeiro positivo')
    ax.set_title('ROC no teste limitado (val, 2023–2025)'); ax.legend(frameon=False, fontsize=7.5, loc='lower right')
    _fim(fig, '15_roc')


def matrizes(c, tab, col='Transformer AE-TM+ORIG | masked', fprs=(0.01, 0.05)):
    """Uma matriz por limiar, lado a lado, e a tabela com acurácia, precisão e recall de cada uma."""
    linhas = []
    fig, axes = plt.subplots(1, len(fprs), figsize=(4.6 * len(fprs), 3.9))
    for ax, f in zip(np.atleast_1d(axes), fprs):
        r = matriz_confusao(c, tab, col, f, ax=ax)
        linhas.append({'limiar (FPR alvo)': f'{f:.0%}', **{k: v for k, v in r.items() if k != 'limiar'}})
    fig.suptitle(f'{METODOS[col]}: limiar escolhido nos limpos de val 2021–22, aplicado em val 2023–25', fontsize=10)
    fig.tight_layout()
    _fim(fig, '15_matriz_confusao')
    return pd.DataFrame(linhas).set_index('limiar (FPR alvo)')


def matriz_confusao(c, tab, col='Transformer AE-TM+ORIG | masked', fpr=0.01, ax=None):
    """Limiar = quantil dos limpos de val em 2021–2022 (FPR alvo); aplicado em val 2023–2025."""
    ref = _tarefa(tab, col, c, periodo='treino')
    thr = float(np.quantile(ref.loc[ref['y'] == 0, 'score'], 1 - fpr))
    t = _tarefa(tab, col, c)
    y, p = t['y'].values, (t['score'].values > thr).astype(int)
    tp, fp = int(((p == 1) & (y == 1)).sum()), int(((p == 1) & (y == 0)).sum())
    fn, tn = int(((p == 0) & (y == 1)).sum()), int(((p == 0) & (y == 0)).sum())
    n = len(y)
    res = {'limiar': thr, 'VP': tp, 'FP': fp, 'FN': fn, 'VN': tn, 'n': n, 'eventos': int(y.sum()),
           'acurácia': (tp + tn) / n, 'precisão': tp / max(tp + fp, 1), 'recall': tp / max(tp + fn, 1),
           'acurácia "tudo normal"': (fp + tn) / n, 'FPR no teste': fp / max(fp + tn, 1)}
    m = np.array([[tn, fp], [fn, tp]])
    ax.imshow(m, cmap='Blues')
    for (r, k), v in np.ndenumerate(m):
        ax.text(k, r, v, ha='center', va='center', fontsize=14, color=INK if v < m.max() / 2 else 'white')
    ax.set_xticks([0, 1], ['previsto normal', 'previsto anomalia']); ax.set_yticks([0, 1], ['limpo', 'evento'])
    ax.set_title(f'limiar para FPR {fpr:.0%}', fontsize=9.5)
    return res


# ---------------- mapa, série e erros ----------------

def _celulas_ae(c, step, ckpt_nome='transformer_ae_masked_tm_orig_tapajos_sub.pt'):
    import baselines
    import modelo
    s, ctx, _ = baselines.load('tapajos_sub', 'tapajos', k=modelo.K_SEQ)
    tr = modelo.Trainer.load(config.CHECKPOINTS_DIR / ckpt_nome)
    alvo = s[(s['step'] == step) | s['val_normal']]
    err = modelo.score_all(tr, alvo, ctx, log=False)
    rows = s.loc[err.index]
    ref = err[rows['val_normal'].values].values.ravel()
    ref = ref[~np.isnan(ref)]
    med, iqr = np.median(ref), np.subtract(*np.percentile(ref, [75, 25]))
    z = (err[(rows['step'] == step).values] - med) / iqr
    return z, rows[rows['step'] == step]


def fig_mapa(c, step='2024Q3'):
    z, rows = _celulas_ae(c, step)
    G = config.CELL_GRID
    n = int(c.lin.max()) + 1
    M = np.full((n * G, n * G), np.nan)
    pos = {p: (c.lin[k], c.col[k]) for k, p in enumerate(c.pids)}
    for idx, r in rows.iterrows():
        li, co = pos[r['patch_id']]
        M[li * G:(li + 1) * G, co * G:(co + 1) * G] = z.loc[idx].values.reshape(G, G)
    t = _t(c, step)
    x = c.cube['s2'][:, t, 1:4]                                          # B2, B3, B4
    k = 4
    R = np.zeros((n * 224 // k, n * 224 // k, 3), np.float32)
    for q, p in enumerate(c.pids):
        li, co = pos[p]
        R[li * 56:(li + 1) * 56, co * 56:(co + 1) * 56] = np.clip(x[q, ::-1, ::k, ::k].transpose(1, 2, 0) / 10000 * 3.5, 0, 1)
    ev = c.trim[(c.trim['step'] == step) & (c.trim['estado'] == 'evento') & c.trim['patch_id'].isin(c.pids)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.3))
    axes[0].imshow(R); axes[0].set_title(f'Sentinel-2 RGB, {step} (sub-AOI, 35,8 km)')
    im = axes[1].imshow(M, cmap='magma', vmin=0, vmax=np.nanpercentile(M, 99.5))
    axes[1].set_title('Score do AE (TerraMind + originais) por célula\n(z robusto vs. normais de validação)')
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    for _, r in ev.iterrows():
        li, co = pos[r['patch_id']]
        for ax, s_ in [(axes[0], 56), (axes[1], G)]:
            ax.add_patch(Rectangle((co * s_ - 0.5, li * s_ - 0.5), s_, s_, fill=False, ec='#3fd0ff', lw=1.2))
    _off(*axes)
    fig.suptitle(f'Contorno azul = patch com evento rotulado em {step} ({len(ev)} patches)', fontsize=10)
    _fim(fig, '16_mapa_anomalia')


def fig_serie_score(c, tab, tipo='desmatamento', cols=('Transformer AE-TM+ORIG | masked', FUSAO, 'Espectral | queda de NDVI')):
    pid, st = EXEMPLOS[tipo]
    fig, ax = plt.subplots(figsize=(12, 3.4))
    lab = c.trim[c.trim['patch_id'] == pid].set_index('step')
    for k, s_ in enumerate(c.steps):
        if s_ in lab.index and lab.loc[s_, 'estado'] == 'evento':
            ax.axvspan(k - 0.5, k + 0.5, color=CAT[5], alpha=0.15)
    for col, cor in zip(cols, CAT):
        ref = _tarefa(tab, col, c, periodo='treino')
        r = ref.loc[ref['y'] == 0, 'score']
        med, iqr = r.median(), r.quantile(0.75) - r.quantile(0.25)
        v = tab[tab['patch_id'] == pid].set_index('step')[col].reindex(c.steps)
        ax.plot(range(len(c.steps)), (v - med) / iqr, marker='o', ms=3, color=cor, label=METODOS[col])
    ax.set_xticks(range(0, len(c.steps), 2), c.steps[::2], rotation=45, fontsize=8)
    ax.set_ylabel('score (z robusto)'); ax.grid(axis='y', color=GRID, lw=0.5)
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    ax.set_title(f'{pid}: score ao longo do tempo. Faixa vermelha = trimestre com evento rotulado')
    _fim(fig, '16_serie_score')


def fig_erros(c, tab, col='Transformer AE-TM+ORIG | masked', n=2):
    t = _tarefa(tab, col, c)
    fp = t[t['y'] == 0].nlargest(n, 'score')
    fn = t[t['y'] == 1].nsmallest(n, 'score')
    fig, axes = plt.subplots(2, 2 * n, figsize=(2.9 * 2 * n, 6.0))
    for r, (df, rot) in enumerate([(fp, 'FALSO POSITIVO'), (fn, 'FALSO NEGATIVO')]):
        for k, (_, e) in enumerate(df.iterrows()):
            for d, ax in [(-4, axes[r, 2 * k]), (0, axes[r, 2 * k + 1])]:
                ax.imshow(rgb(s2(c, e['patch_id'], e['step'], d)[0])); ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f'{c.steps[_t(c, e["step"], d)]}' + (' (t)' if d == 0 else ' (t−4)'), fontsize=8.5)
            axes[r, 2 * k].set_ylabel(f'{rot}\n{e["patch_id"]}\n{_evento(c, e["patch_id"], e["step"])}', fontsize=8)
    fig.suptitle('Cada par: mesmo trimestre do ano anterior (t−4) × trimestre avaliado (t)', fontsize=10)
    fig.tight_layout()
    _fim(fig, '17_erros')
    return fp, fn
