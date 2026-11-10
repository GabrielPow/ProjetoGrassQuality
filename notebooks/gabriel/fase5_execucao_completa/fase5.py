"""Fase 5 (PLANO_JUNTAR_DEEP_LEARNING.md, seção 8): AOIs completas, do download à avaliação, sem notebooks.

Feito para rodar dias seguidos na máquina dedicada (passo a passo em SERVIDOR.md). Tudo é retomável:
- stream_<aoi>: cada (bloco, trimestre) pronto fica marcado; um download que falha é pulado e refeito numa
  nova passada (até 5 passadas por execução);
- ajuste: células de ajuste e detectores salvos em disco; dados de treino de cada AE salvos; cada AE grava um
  checkpoint por época (modelo.Trainer, resume_path) e continua da época seguinte;
- pontuacao: em partes de PART_SLOTS slots (40 blocos); cada parte é salva ao terminar;
- fusao / avaliacao: uma saída por AOI, pulada se já existe;
- cada etapa concluída grava checkpoints/fase5_execucao_completa/<etapa>.ok e é pulada nas execuções seguintes.
Se o processo cair (rede, energia, memória), rode o MESMO comando de novo: continua de onde parou.
`rodar_fase5.bat` faz isso sozinho.

Esta pasta é a EXECUÇÃO COMPLETA (AOIs inteiras); os notebooks de notebooks/gabriel são a sub-AOI.
Os módulos (config, baselines, modelo, ...) continuam em notebooks/gabriel e são importados de lá.

Uso (em notebooks/gabriel/fase5_execucao_completa, com o venv):
    python fase5.py --status           # o que já está pronto
    python fase5.py --teste            # cadeia inteira em 9 blocos, saída separada (fase5_teste)
    python fase5.py                    # tudo, na ordem
    python fase5.py --etapas ajuste    # só essas etapas (as anteriores precisam estar prontas)
    python fase5.py --refazer ajuste   # apaga a saída da etapa (e das seguintes) e refaz

Decisões fixas (PROTOCOLO.md, seções 6 e 7; nada aqui olha rótulos de anomalia antes da avaliação):
- detectores e os 3 AEs (TerraMind, originais, as duas; formulação masked, d = 128) ajustados só com normais
  de train / val do Tapajós; Rondônia só é pontuada;
- referências da fusão (z robusto) e da normalização local: células val_normal do Tapajós, nas duas AOIs;
- avaliação: Tapajós em `test`, Rondônia em `test_gen`, período 2023–2025, só nas amostras (patch, trimestre)
  que têm score em todos os métodos (histórico de 8 trimestres do AE).

Códigos de saída: 0 ok; 1 erro (tentar de novo resolve rede/energia); 3 configuração (não adianta repetir);
130 interrompido com Ctrl+C.
"""
import argparse
import json
import math
import os
import pickle
import platform
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # módulos de notebooks/gabriel
import config  # noqa: E402

ETAPAS = ['stream_tapajos', 'stream_norte_ro', 'ajuste', 'pontuacao', 'fusao', 'avaliacao']
REQUER = {'stream_tapajos': [], 'stream_norte_ro': [], 'ajuste': ['stream_tapajos'],
          'pontuacao': ['stream_tapajos', 'stream_norte_ro', 'ajuste'], 'fusao': ['pontuacao'],
          'avaliacao': ['fusao']}
AOI_TREINO = 'tapajos'
AOIS = ['tapajos', 'norte_ro']
SPLIT_AVAL = {'tapajos': 'test', 'norte_ro': 'test_gen'}
ENTRADAS = ['tm', 'orig', 'tm+orig']
MODE, D_MODEL = 'masked', 128
PART_SLOTS = 640                 # múltiplo de 64 (lote de leitura): as partes dão o mesmo resultado que uma passada só
EXIT_ERRO, EXIT_CONFIG, EXIT_CTRL_C = 1, 3, 130
DISCO_GB = {False: 60, True: 3}  # espaço livre recomendado (células das duas AOIs ~40 GB + caches de treino)

PARAMS = {False: dict(epochs=40, n_train=200_000, n_val=50_000, n_boot=200, n_boot_par=300),
          True: dict(epochs=2, n_train=20_000, n_val=5_000, n_boot=20, n_boot_par=20)}

NOMES = {'tm': 'Transformer AE | masked',            # mesmos nomes do 08/09, para casar as tabelas
         'orig': 'Transformer AE-ORIG | masked',
         'tm+orig': 'Transformer AE-TM+ORIG | masked'}
NDVI = 'Espectral | queda de NDVI'
KNN_TM = 'TerraMind S2S1 | kNN no delta'


def _orig():
    import baselines
    return baselines.ORIG_IFOREST, baselines.ORIG_FOURIER, f'{baselines.ORIG_TAG} | Deep SVDD (20 features + delta)'


def pares_fusao():
    """Mesmas combinações do 08_fusao; a principal é AE + Original IForest (PROTOCOLO, seção 6)."""
    iforest, fourier, _ = _orig()
    ae = NOMES['tm']
    return {'AE + Original IForest': [ae, iforest], 'AE + queda de NDVI': [ae, NDVI],
            'AE + Fourier': [ae, fourier], 'AE + kNN no delta TerraMind': [ae, KNN_TM]}


def principal():
    return 'Fusão J2 | AE + Original IForest [z_media]'


def keep_cells():
    iforest, fourier, _ = _orig()
    return [iforest, fourier, NDVI, KNN_TM]


class ConfigErro(Exception):
    """Erro que não se resolve rodando de novo (arquivo faltando, GPU pedida e ausente, ...)."""


# ---------------- caminhos, log, arquivos ----------------

class Caminhos:
    def __init__(self, teste):
        self.teste = teste
        tag = 'fase5_teste' if teste else 'fase5_execucao_completa'
        self.prefixo = 'teste_' if teste else ''
        self.work = config.CHECKPOINTS_DIR / tag      # intermediários (grandes): modelos, partes, caches
        self.res = config.RESULTS_DIR / tag           # resultados (pequenos): scores, métricas, curvas, logs
        for d in (self.work, self.res, self.res / 'logs', config.EMBEDDINGS_DIR):
            d.mkdir(parents=True, exist_ok=True)

    def emb(self, aoi):
        return config.EMBEDDINGS_DIR / f'{self.prefixo}{aoi}.npz'

    def ok(self, etapa):
        return self.work / f'{etapa}.ok'

    def partes(self, nome):
        return self.work / 'partes' / nome


def nome_arq(entrada):
    return entrada.replace('+', '_')


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, x):
        for s in self.streams:
            s.write(x)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def log(msg):
    print(f'[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}', flush=True)


def salvar(obj, path):
    """pickle com escrita atômica (um corte no meio não deixa arquivo pela metade)."""
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def ler(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def salvar_csv(df, path):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    df.to_csv(tmp, index=False, compression='gzip' if path.name.endswith('.gz') else None)
    os.replace(tmp, path)


def salvar_json(obj, path):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    os.replace(tmp, path)


# ---------------- blocos do teste rápido ----------------

def blocos_teste(aoi):
    """Teste: poucos blocos completos (16 patches), sorteados com semente fixa e por split, para a cadeia inteira
    ter normais de treino e de validação e algo a avaliar."""
    grade = pd.read_csv(config.LABELS_DIR / aoi / 'grade.csv')
    cheios = grade.groupby('block').filter(lambda b: len(b) == 16).groupby('block')['split'].first()
    quer = {'train': 3, 'val': 2, 'test': 2} if aoi == AOI_TREINO else {'test_gen': 2}
    rng = np.random.default_rng(config.SEED)
    out = []
    for sp, n in quer.items():
        b = sorted(cheios[cheios == sp].index)
        out += [str(x) for x in rng.choice(b, min(n, len(b)), replace=False)]
    return out


def pendentes(aoi, store, blocks=None):
    """Patch-trimestres ainda sem embedding (nos blocos pedidos)."""
    import embeddings
    occ = store['patch_id'] != ''
    if blocks is not None:
        grade = pd.read_csv(config.LABELS_DIR / aoi / 'grade.csv')
        sel = np.zeros(len(occ), bool)
        for b, L in enumerate(embeddings.block_layout(grade)):
            if L['block'] in set(blocks):
                sel[b * embeddings.NB:(b + 1) * embeddings.NB] = True
        occ &= sel
    return int((~store['done'][occ]).sum())


# ---------------- etapas ----------------

_TM = {}


def etapa_stream(aoi, a, p):
    import dados
    import embeddings
    log(f'Earth Engine: {dados.init_ee()}')
    if 'm' not in _TM:
        _TM['m'] = embeddings.load_model(a.threads, a.device)
        log(f'TerraMind em {next(_TM["m"].parameters()).device}')
    blocks = blocos_teste(aoi) if p.teste else None
    if blocks:
        log(f'teste: blocos {blocks}')
    for passada in range(1, 6):
        t0 = time.time()
        store = embeddings.stream_aoi(aoi, p.emb(aoi), workers=a.workers, model=_TM['m'], blocks=blocks,
                                      max_tasks=a.max_tasks)
        falta = pendentes(aoi, store, blocks)
        log(f'{aoi}: passada {passada} em {(time.time() - t0) / 3600:.2f} h; {falta} patch-trimestres pendentes')
        if falta == 0:
            return {'passadas': passada, 'arquivo': str(p.emb(aoi))}
        if a.max_tasks:
            log('--max-tasks: parada de propósito (medição de velocidade); a etapa não é marcada como pronta')
            return None
        time.sleep(60)
    raise RuntimeError(f'{aoi}: ainda há {falta} pendentes depois de 5 passadas (veja as falhas de download acima)')


def etapa_ajuste(a, p):
    """Detectores (06/08) e os 3 AEs (07/09), só com normais de train / val do Tapajós."""
    import baselines
    import modelo
    pr = PARAMS[p.teste]
    fonte = p.prefixo + AOI_TREINO
    info = {}

    f_mod = p.work / 'baselines_modelos.pkl'
    if f_mod.exists():
        log('detectores: já ajustados')
    else:
        s4, ctx4, _ = baselines.load(fonte, AOI_TREINO)
        log(f'detectores: {int(s4["treino_normal"].sum())} amostras normais de treino, {int(s4["val_normal"].sum())} de validação')
        f_norm = p.work / 'baselines_normais.pkl'
        if f_norm.exists():
            fit = ler(f_norm)
        else:
            t0 = time.time()
            fit = baselines.collect_normals(s4, ctx4, a.cfgs)
            salvar(fit, f_norm)
            log(f'células de ajuste sorteadas em {(time.time() - t0) / 60:.1f} min: treino {len(fit["tr"]["spec"])}, val {len(fit["va"]["spec"])}')
        t0 = time.time()
        models = baselines.fit_all(fit, a.cfgs)
        salvar(models, f_mod)
        hist = {k: m[2].hist for k, m in models.items() if hasattr(m[2], 'hist')}
        salvar_json({'epocas_deep_svdd': {k: len(v) for k, v in hist.items()}}, p.res / 'detectores.json')
        log(f'detectores ajustados em {(time.time() - t0) / 60:.1f} min')
    info['detectores'] = str(f_mod)

    s8 = ctx8 = None
    for e in ENTRADAS:
        final = p.work / f'ae_{nome_arq(e)}.pt'
        if final.exists():
            log(f'AE {e}: já treinado')
            continue
        if s8 is None:
            s8, ctx8, _ = baselines.load(fonte, AOI_TREINO, k=modelo.K_SEQ)
        f_dados = p.work / f'dados_treino_{nome_arq(e)}.npz'
        if f_dados.exists():
            with np.load(f_dados) as d:
                tr = (d['Xtr'], d['Vtr'], d['TFtr'])
                va = (d['Xva'], d['Vva'], d['TFva'])
        else:
            t0 = time.time()
            tr = modelo.gather_training(s8, ctx8, 'treino_normal', pr['n_train'], entrada=e)
            va = modelo.gather_training(s8, ctx8, 'val_normal', pr['n_val'], seed=config.SEED + 1, entrada=e)
            tmp = f_dados.with_name(f_dados.stem + '.tmp.npz')
            np.savez(tmp, Xtr=tr[0], Vtr=tr[1], TFtr=tr[2], Xva=va[0], Vva=va[1], TFva=va[2])
            os.replace(tmp, f_dados)
            log(f'AE {e}: sequências sorteadas em {(time.time() - t0) / 60:.1f} min')
        parcial = final.with_name(final.stem + '.parcial.pt')
        log(f'AE {e}: treinando (treino {tr[0].shape}, val {va[0].shape})' + (' - retomando' if parcial.exists() else ''))
        t0 = time.time()
        t = modelo.Trainer(*tr, *va, mode=MODE, d_model=D_MODEL, epochs=pr['epochs'], entrada=e,
                           resume_path=parcial, device=a.device, threads=a.threads)
        t.save(final)
        salvar_csv(pd.DataFrame(t.curve), p.res / f'curva_ae_{nome_arq(e)}.csv')
        parcial.unlink(missing_ok=True)
        f_dados.unlink(missing_ok=True)          # 1–1,5 GB cada; o modelo final já está salvo
        log(f'AE {e}: {(time.time() - t0) / 60:.1f} min nesta sessão, melhor val {t.best_val:.4f}, {len(t.curve)} épocas')
        info[f'val_{e}'] = float(t.best_val)
    return info


def por_partes(pasta, s, n_slots, fn, rotulo):
    """Roda fn(s_parte) para cada faixa de PART_SLOTS slots e salva cada parte; partes prontas são puladas."""
    pasta.mkdir(parents=True, exist_ok=True)
    n = math.ceil(n_slots / PART_SLOTS)
    t0, feitas = time.time(), 0
    for k in range(n):
        f = pasta / f'parte_{k:04d}.pkl'
        if f.exists():
            continue
        sp = s[(s['i'] >= k * PART_SLOTS) & (s['i'] < (k + 1) * PART_SLOTS)]
        salvar(fn(sp) if len(sp) else None, f)
        feitas += 1
        prontas = len(list(pasta.glob('parte_*.pkl')))
        el = (time.time() - t0) / 60
        log(f'  {rotulo}: parte {prontas}/{n} ({el:.1f} min nesta sessão, ~{el / feitas * (n - prontas):.0f} min restantes)')


def juntar(pasta):
    return [x for x in (ler(f) for f in sorted(pasta.glob('parte_*.pkl'))) if x is not None]


def etapa_pontuacao(a, p):
    import baselines
    import modelo
    models = ler(p.work / 'baselines_modelos.pkl')
    aes = {e: modelo.Trainer.load(p.work / f'ae_{nome_arq(e)}.pt', threads=a.threads, device=a.device) for e in ENTRADAS}
    keep = keep_cells()
    for aoi in AOIS:
        fonte = p.prefixo + aoi
        s4, ctx4, _ = baselines.load(fonte, aoi)
        log(f'{aoi}: detectores em {len(s4)} amostras')
        por_partes(p.partes(f'baselines_{aoi}'), s4, ctx4['n_slots'],
                   lambda sp: baselines.score_all(sp, ctx4, models, a.cfgs, log=False, keep_cells=keep),
                   f'{aoi} detectores')
        s8, ctx8, _ = baselines.load(fonte, aoi, k=modelo.K_SEQ)
        for e, t in aes.items():
            log(f'{aoi}: AE {e}')
            por_partes(p.partes(f'ae_{nome_arq(e)}_{aoi}'), s8, ctx8['n_slots'],
                       lambda sp: modelo.score_all(t, sp, ctx8, log=False) if (sp['t'] >= modelo.K_SEQ).any() else None,
                       f'{aoi} AE {e}')
    return {}


def _chave(s, idx):
    return pd.MultiIndex.from_frame(s.loc[idx, ['patch_id', 'step']])


def etapa_fusao(a, p):
    """Tabela de scores de patch por AOI (detectores + AEs + fusão J2). Referências: val_normal do Tapajós."""
    import baselines
    import fusao
    import modelo
    pares = pares_fusao()
    ref_err, refs, resumo = {}, None, {}
    for aoi in AOIS:                                  # Tapajós primeiro: dá as referências
        fonte = p.prefixo + aoi
        s4, _, _ = baselines.load(fonte, aoi)
        s8, _, _ = baselines.load(fonte, aoi, k=modelo.K_SEQ)
        b = juntar(p.partes(f'baselines_{aoi}'))
        b_patch = pd.concat([x[0] for x in b]).dropna(how='all')
        b_patch = b_patch.set_axis(_chave(s4, b_patch.index))
        b_cells = {m: pd.concat([x[1][m] for x in b]) for m in keep_cells()}
        b_cells = {m: v.set_axis(_chave(s4, v.index)) for m, v in b_cells.items()}

        ae_patch, ae_cells = [], {}
        for e in ENTRADAS:
            err = pd.concat(juntar(p.partes(f'ae_{nome_arq(e)}_{aoi}')))
            if aoi == AOI_TREINO:
                ref_err[e] = (err, s8)
            ae_patch.append(modelo.patch_scores(err, s8, NOMES[e], ref=None if aoi == AOI_TREINO else ref_err[e]))
            ae_cells[NOMES[e]] = err.set_axis(_chave(s8, err.index))
        ae_patch = pd.concat(ae_patch, axis=1)
        ae_patch = ae_patch.set_axis(_chave(s8, ae_patch.index))

        comum = ae_patch.index.intersection(b_patch.index)
        cells_all = {m: v.loc[comum] for m, v in {**ae_cells, **b_cells}.items()}
        if aoi == AOI_TREINO:
            meta = s8.set_index(['patch_id', 'step']).loc[comum]
            ref_mask = meta['val_normal'].values
            metodos = sorted({m for ms in pares.values() for m in ms})
            refs = {m: fusao.Ref(cells_all[m].values[ref_mask]) for m in metodos}
            resumo['referencia'] = {'amostras_val_normal': int(ref_mask.sum()),
                                    'metodos': {m: {'mediana': r.med, 'iqr': r.iqr} for m, r in refs.items()}}
        fus = {}
        for nome, ms in pares.items():
            for r in fusao.RULES:
                mx, mean = fusao.patch_from_cells(fusao.fuse({m: cells_all[m].values for m in ms}, refs, r))
                fus[f'Fusão J2 | {nome} [{r}]'] = mx
                fus[f'Fusão J2 | {nome} [{r}] [média]'] = mean
        tab = pd.concat([ae_patch.loc[comum], b_patch.loc[comum], pd.DataFrame(fus, index=comum)], axis=1)
        tab = tab.loc[:, ~tab.columns.duplicated()].reset_index()
        salvar_csv(tab, p.res / f'scores_{aoi}.csv.gz')
        resumo[aoi] = {'amostras': len(tab), 'amostras_detectores': len(b_patch), 'amostras_ae': len(ae_patch)}
        log(f'{aoi}: {len(tab)} amostras com todos os métodos -> scores_{aoi}.csv.gz')
    salvar_json(resumo, p.res / 'fusao.json')
    return resumo


def _rotulos(aoi):
    d = config.LABELS_DIR / aoi
    return {'anual': pd.read_csv(d / 'rotulos_anuais.csv.gz'), 'trim': pd.read_csv(d / 'rotulos_trimestrais.csv.gz'),
            'grade': pd.read_csv(d / 'grade.csv')}


def _eval_all(df, labels, grade, key, split, n_boot):
    import avaliacao
    out = []
    for col in [c for c in df.columns if '|' in c and not c.endswith('[média]')]:   # [média]: ablação já feita na sub-AOI
        sc = df[['patch_id', key]].assign(score=df[col].values).dropna(subset=['score'])
        r = avaliacao.evaluate(sc, labels, grade, key=key, split=split, n_boot=n_boot)
        if len(r):
            out.append(r.assign(metodo=col))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def etapa_avaliacao(a, p):
    import avaliacao
    pr = PARAMS[p.teste]
    _, _, dsvdd = _orig()
    princ = principal()
    pares = [(princ, NOMES['tm']), (princ, _orig()[0]), (princ, NDVI), (princ, KNN_TM),
             (NOMES['tm+orig'], NOMES['tm']), (NOMES['tm+orig'], princ), (NOMES['orig'], NOMES['tm']),
             (NOMES['tm+orig'], NOMES['orig']), (NOMES['orig'], dsvdd), (NOMES['tm'], NDVI)]
    resumo = {}
    for aoi in AOIS:
        split = SPLIT_AVAL[aoi]
        lab = _rotulos(aoi)
        tab = pd.read_csv(p.res / f'scores_{aoi}.csv.gz')
        f_met = p.res / f'metricas_{aoi}.csv'
        if f_met.exists():
            res = pd.read_csv(f_met)
            log(f'{aoi}: métricas já calculadas')
        else:
            t0 = time.time()
            rq = _eval_all(tab, lab['trim'], lab['grade'], 'step', split, pr['n_boot']).assign(resolucao='trimestral')
            anual = (tab.assign(year=tab['step'].str[:4].astype(int)).drop(columns='step')
                     .groupby(['patch_id', 'year']).max().reset_index())
            ra = _eval_all(anual, lab['anual'], lab['grade'], 'year', split, pr['n_boot']).assign(resolucao='anual')
            res = pd.concat([rq, ra], ignore_index=True)
            salvar_csv(res, f_met)
            log(f'{aoi}: métricas em {(time.time() - t0) / 60:.1f} min')
        f_cmp = p.res / f'comparacao_pareada_{aoi}.csv'
        if not f_cmp.exists():
            comp = []
            for x, y in pares:
                try:
                    c = avaliacao.compare(tab[['patch_id', 'step']].assign(score=tab[x].values),
                                          tab[['patch_id', 'step']].assign(score=tab[y].values),
                                          lab['trim'], lab['grade'], split=split, n_boot=pr['n_boot_par'])
                    comp.append(c.assign(a=x, b=y))
                except Exception as ex:       # ex.: sem eventos no split (teste rápido)
                    log(f'  comparação {x} x {y} não calculada: {type(ex).__name__}: {ex}')
            salvar_csv(pd.concat(comp, ignore_index=True) if comp else pd.DataFrame(), f_cmp)
        comp = pd.read_csv(f_cmp) if f_cmp.stat().st_size > 2 else pd.DataFrame()
        todos = res[res['tarefa'] == 'todos'] if len(res) else res
        resumo[aoi] = {
            'split': split, 'amostras': len(tab),
            'auroc_todos': {r: todos[todos['resolucao'] == r].set_index('metodo')['AUROC'].round(4).to_dict()
                            for r in ['trimestral', 'anual']} if len(todos) else {},
            'eventos': ({r: int(todos.loc[todos['resolucao'] == r, 'n_pos'].fillna(0).max())
                         for r in ['trimestral', 'anual'] if (todos['resolucao'] == r).any()} if len(todos) else {}),
            'pareado_todos': (comp[comp['tarefa'] == 'todos'][['a', 'b', 'dAUROC', 'dAUROC_lo', 'dAUROC_hi']]
                              .round(4).to_dict('records') if len(comp) else []),
        }
        if len(todos):
            t = todos[todos['resolucao'] == 'trimestral'].set_index('metodo')
            log(f'{aoi} ({split}), AUROC trimestral, 10 melhores:\n'
                + t['AUROC'].sort_values(ascending=False).head(10).round(3).to_string())
    salvar_json(resumo, p.res / 'resumo.json')
    return {'resumo': str(p.res / 'resumo.json')}


FUNCS = {'stream_tapajos': lambda a, p: etapa_stream('tapajos', a, p),
         'stream_norte_ro': lambda a, p: etapa_stream('norte_ro', a, p),
         'ajuste': etapa_ajuste, 'pontuacao': etapa_pontuacao, 'fusao': etapa_fusao, 'avaliacao': etapa_avaliacao}


def saidas(etapa, p):
    """Arquivos que `--refazer` apaga (as embeddings baixadas nunca são apagadas por aqui)."""
    w, r = p.work, p.res
    return {'ajuste': [*w.glob('baselines_*.pkl'), *w.glob('ae_*.pt'), *w.glob('dados_treino_*.npz'),
                       *r.glob('curva_ae_*.csv'), r / 'detectores.json'],
            'pontuacao': [w / 'partes'],
            'fusao': [*r.glob('scores_*.csv.gz'), r / 'fusao.json'],
            'avaliacao': [*r.glob('metricas_*.csv'), *r.glob('comparacao_pareada_*.csv'), r / 'resumo.json'],
            }.get(etapa, [])


def refazer(etapa, p):
    if etapa.startswith('stream'):
        raise ConfigErro('--refazer não vale para o download (apagaria dias de trabalho). '
                         'Para baixar de novo, apague à mão o arquivo em embeddings/.')
    for e in ETAPAS[ETAPAS.index(etapa):]:          # a etapa e todas as que dependem dela
        for f in [p.ok(e), *saidas(e, p)]:
            if f.is_dir():
                shutil.rmtree(f)
            elif f.exists():
                f.unlink()
    log(f'--refazer {etapa}: saídas de {etapa} e das etapas seguintes apagadas')


# ---------------- verificações e status ----------------

def verificar(a, p, etapas):
    import torch
    log(f'Python {platform.python_version()} | torch {torch.__version__} | {platform.platform()}')
    if torch.cuda.is_available():
        log(f'GPU: {torch.cuda.get_device_name(0)}')
    else:
        log('GPU: nenhuma (CUDA indisponível): TerraMind e AEs vão rodar em CPU, bem mais devagar')
        if a.device == 'cuda':
            raise ConfigErro('--device cuda pedido, mas o torch não vê GPU (instale o torch com CUDA, ver SERVIDOR.md)')
    log(f'dados em {config.DATA_ROOT}')
    for aoi in AOIS:
        for f in ['grade.csv', 'rotulos_anuais.csv.gz', 'rotulos_trimestrais.csv.gz']:
            if not (config.LABELS_DIR / aoi / f).exists():
                raise ConfigErro(f'falta {config.LABELS_DIR / aoi / f}: copie a pasta rotulos/ do notebook (ver SERVIDOR.md)')
    if any(e.startswith('stream') and not p.ok(e).exists() for e in etapas):
        import dados
        try:
            key = dados.service_account_key()
        except FileNotFoundError as ex:
            raise ConfigErro(str(ex))
        log(f'Earth Engine: {"conta de serviço " + key if key else f"login de usuário (nenhuma chave em {config.CHAVES_DIR})"}')
    livre = shutil.disk_usage(config.DATA_ROOT).free / 1e9
    log(f'disco livre: {livre:.0f} GB')
    if livre < DISCO_GB[p.teste]:
        log(f'AVISO: recomendado >= {DISCO_GB[p.teste]} GB livres para esta execução')


def status(p):
    import embeddings
    for e in ETAPAS:
        ok = p.ok(e)
        extra = ''
        if ok.exists():
            extra = json.loads(ok.read_text(encoding='utf-8')).get('fim', '')
        elif e.startswith('stream'):
            aoi = e.split('_', 1)[1]
            if p.emb(aoi).exists():
                st = embeddings.load_store(p.emb(aoi))
                occ = st['patch_id'] != ''
                extra = f'{st["done"][occ].mean():.1%} dos patch-trimestres baixados'
                if p.teste:
                    tot = len(st['done'][occ].ravel())
                    extra += f' (teste: faltam {pendentes(aoi, st, blocos_teste(aoi))} de {tot})'
        elif e == 'ajuste':
            feitos = [f.name for f in p.work.glob('*.pt') if not f.name.endswith('.parcial.pt')]
            parc = [f.name for f in p.work.glob('*.parcial.pt')]
            extra = f'prontos: {feitos or "-"}; em treino: {parc or "-"}'
        elif e == 'pontuacao' and (p.work / 'partes').exists():
            extra = '; '.join(f'{d.name}: {len(list(d.glob("parte_*.pkl")))} partes'
                              for d in sorted((p.work / 'partes').iterdir()))
        print(f'  {"OK " if ok.exists() else "..."} {e:<16} {extra}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--etapas', nargs='+', choices=ETAPAS, help='só estas etapas (padrão: todas, na ordem)')
    ap.add_argument('--refazer', choices=ETAPAS[2:], help='apaga a saída desta etapa e das seguintes e refaz')
    ap.add_argument('--teste', action='store_true', help='cadeia inteira em 9 blocos, 2 épocas; saída em *_teste')
    ap.add_argument('--status', action='store_true', help='mostra o que já está pronto e sai')
    ap.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    ap.add_argument('--workers', type=int, default=10, help='downloads simultâneos do Earth Engine')
    ap.add_argument('--threads', type=int, default=8, help='threads de CPU do PyTorch')
    ap.add_argument('--cfgs', nargs='+', default=['s2s1'], choices=['s2s1', 's2'],
                    help="embeddings dos detectores TerraMind ('s2' dobra o tempo; os slides usam só s2s1)")
    ap.add_argument('--max-tasks', type=int, default=None,
                    help='download: para depois de N (bloco, trimestre), para medir a velocidade (não marca a etapa)')
    a = ap.parse_args()

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding='utf-8', errors='replace')     # console do Windows (cp1252) e acentos
        except AttributeError:
            pass
    p = Caminhos(a.teste)
    if a.status:
        status(p)
        return 0
    f_log = open(p.res / 'logs' / f'fase5_{datetime.now():%Y%m%d-%H%M%S}.log', 'a', encoding='utf-8')
    sys.stdout = _Tee(sys.stdout, f_log)
    sys.stderr = _Tee(sys.stderr, f_log)
    versao = config.REPO_ROOT / 'VERSAO.txt'
    salvar_json({'inicio': datetime.now().isoformat(timespec='seconds'), 'argv': sys.argv,
                 'versao': versao.read_text(encoding='utf-8').strip() if versao.exists() else None,
                 'config': config.as_dict(), 'params': PARAMS[a.teste]},
                p.res / 'logs' / f'execucao_{datetime.now():%Y%m%d-%H%M%S}.json')
    etapas = [e for e in ETAPAS if not a.etapas or e in a.etapas]
    try:
        import utils
        utils.set_seed(config.SEED)
        verificar(a, p, etapas)
        if a.refazer:
            refazer(a.refazer, p)
        for e in etapas:
            if p.ok(e).exists():
                log(f'== {e}: pronta (pulando)')
                continue
            faltam = [r for r in REQUER[e] if not p.ok(r).exists()]
            if faltam:
                raise ConfigErro(f'{e} precisa de {faltam} prontas antes')
            log(f'== {e}')
            t0 = time.time()
            res = FUNCS[e](a, p)
            if res is None:
                return 0
            salvar_json({'fim': datetime.now().isoformat(timespec='seconds'),
                         'minutos': round((time.time() - t0) / 60, 1), **res}, p.ok(e))
            log(f'== {e}: pronta em {(time.time() - t0) / 60:.1f} min')
        log('Fase 5: tudo pronto' if len(etapas) == len(ETAPAS) else 'etapas pedidas prontas')
        return 0
    except ConfigErro as ex:
        log(f'ERRO DE CONFIGURAÇÃO: {ex}')
        return EXIT_CONFIG
    except KeyboardInterrupt:
        log('interrompido (Ctrl+C); rode o mesmo comando para continuar de onde parou')
        return EXIT_CTRL_C
    except Exception:
        log('ERRO (o estado salvo está íntegro; rode de novo para continuar):\n' + traceback.format_exc())
        return EXIT_ERRO


if __name__ == '__main__':
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # os._exit: não espera downloads ainda em andamento nas threads (o estado já foi salvo)
    os._exit(code)
