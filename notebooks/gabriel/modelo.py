"""Etapa 6: Temporal Transformer Autoencoder sobre as células de 320 m.

Entrada: sequência Z(t-K), ..., Z(t) de uma célula (K = 8 trimestres = 2 anos), com codificação temporal
(ano e trimestre do ano em seno/cosseno: a sazonalidade faz parte do "normal") e máscara de atenção
para trimestres sem dado.

O que é Z (parâmetro `entrada`; PLANO_JUNTAR_DEEP_LEARNING.md, Fase 4 / caminho B):
- 'tm'      : embedding TerraMind S2 + S1 da célula (384). É o modelo original da Etapa 6.
- 'orig'    : as 20 features da implementação original (features_originais.ORIGINAL_NAMES).
- 'tm+orig' : as duas concatenadas (404). Perda e score = 0,5 · MSE(TerraMind) + 0,5 · MSE(originais), para
              as 20 dimensões originais não sumirem diante das 384 do TerraMind (PROTOCOLO.md, seção 6).
Um passo de uma célula sem dado original (NaN) conta como passo sem dado (máscara).

O modelo nunca vê Z(t) ao produzir Ẑ(t) (PLANO 6.2), em duas formulações com a mesma arquitetura:
- 'forecast': o passo t é sempre substituído pelo token de máscara; prevê Z(t) a partir de t-K..t-1.
- 'masked':   no treino, passos aleatórios (inclusive t) viram máscara e são reconstruídos com o contexto.
Na pontuação, as duas mascaram só o passo t. A escolha entre elas usa a perda na validação de normais
(sem rótulos de anomalia).

Score da célula = erro quadrático médio entre Ẑ(t) e Z(t) (entradas padronizadas pelos normais de
treino). Score do patch = máximo das 49 células. Normalização local (PLANO 6.4): z-score do erro em relação
aos erros das células normais de validação da mesma cobertura do solo.
"""
import math
import os
import time

import numpy as np
import pandas as pd

import config
import embeddings  # noqa: F401  (mantido: notebooks antigos importam modelo.embeddings)
import features_originais as fo

K_SEQ = 8                          # trimestres de contexto
NC = config.N_CELLS
D = config.TERRAMIND_EMBED_DIM     # 384, dimensão do TerraMind
D_ORIG = len(fo.ORIGINAL_NAMES)    # 20
ENTRADAS = {'tm': D, 'orig': D_ORIG, 'tm+orig': D + D_ORIG}


def grupos(entrada):
    """Grupos de dimensões e pesos da perda: [(fatia, peso), ...]. Um grupo só = MSE comum."""
    if entrada == 'tm':
        return [(slice(0, D), 1.0)]
    if entrada == 'orig':
        return [(slice(0, D_ORIG), 1.0)]
    if entrada == 'tm+orig':
        return [(slice(0, D), 0.5), (slice(D, D + D_ORIG), 0.5)]
    raise ValueError(entrada)


# ---------------- dados: sequências por célula ----------------

def time_features(step_idx):
    """(..., L) índices de passo -> (..., L, 4): ano normalizado, sen/cos do trimestre, posição relativa."""
    steps = np.asarray(config.STEPS)
    y = np.vectorize(lambda i: int(steps[i][:4]))(step_idx).astype(np.float32)
    q = np.vectorize(lambda i: int(steps[i][-1]))(step_idx).astype(np.float32)
    rel = (step_idx - step_idx[..., -1:]).astype(np.float32) / K_SEQ
    return np.stack([(y - 2022) / 3, np.sin(2 * math.pi * q / 4), np.cos(2 * math.pi * q / 4), rel], axis=-1)


def _cell_block(cells, sl, entrada, cfg):
    """Lê as células de uma fatia de slots: (b, T, 49, F) float32, NaN onde a feature original falta."""
    parts = []
    if entrada in ('tm', 'tm+orig'):
        parts.append(cells[f'c_{cfg}'][sl].astype(np.float32))
    if entrada in ('orig', 'tm+orig'):
        assert 'orig_cells' in cells, 'sem orig_cells: rode features_originais.orig_from_cube (sub-AOI) ou o fluxo novo'
        parts.append(fo.original_features(cells['spec_cells'][sl].astype(np.float32),
                                          cells['orig_cells'][sl].astype(np.float32)))
    return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=-1)


def sequences(rows, ctx, cfg='s2s1', batch_slots=64, entrada='tm'):
    """Para cada amostra (patch, t) de `rows`: sequências das 49 células, t-K..t.
    Gera (rows do lote, X (n, 49, K+1, F) float32 com NaN onde falta dado, válido (n, 49, K+1) bool,
    tempo (n, K+1, 4)). Válido = passo com S2 suficiente (como antes) e sem NaN naquela célula."""
    cells, valid = ctx['cells'], ctx['valid']
    for b0 in range(0, ctx['n_slots'], batch_slots):
        r = rows[(rows['i'] >= b0) & (rows['i'] < b0 + batch_slots)]
        if r.empty:
            continue
        C = _cell_block(cells, slice(b0, b0 + batch_slots), entrada, cfg)   # (b, T, 49, F)
        li, tt = r['i'].values - b0, r['t'].values
        idx = tt[:, None] + np.arange(-K_SEQ, 1)[None]                    # (n, K+1)
        X = C[li[:, None], idx].transpose(0, 2, 1, 3)                     # (n, 49, K+1, F)
        v = valid[b0:b0 + batch_slots][li[:, None], idx]                  # (n, K+1)
        vc = v[:, None, :] & ~np.isnan(X).any(-1)                         # (n, 49, K+1)
        yield r, X, vc, time_features(idx)


def gather_training(s, ctx, col, n_max, cfg='s2s1', seed=config.SEED, entrada='tm'):
    """Sorteia até n_max sequências de células das amostras com `col` (treino ou validação normal).
    O sorteio é o mesmo para qualquer entrada; sequências cujo passo t não tem dado são descartadas."""
    rows = s[s[col] & (s['t'] >= K_SEQ)]
    p = min(1.0, n_max / max(1, len(rows) * NC))
    rng = np.random.default_rng(seed)
    X, V, TF = [], [], []
    for r, x, v, tf in sequences(rows, ctx, cfg, entrada=entrada):
        pick = rng.random((len(r), NC)) < p
        pick &= v[:, :, -1]
        n_i, c_i = np.nonzero(pick)
        X.append(x[n_i, c_i].astype(np.float16)); V.append(v[n_i, c_i]); TF.append(tf[n_i])
    return np.concatenate(X), np.concatenate(V), np.concatenate(TF)


# ---------------- modelo ----------------

def _atomic_save(obj, path):
    """torch.save num arquivo temporário e troca no fim: um corte no meio não corrompe o checkpoint."""
    import torch
    path = str(path)
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _device(device=None):
    """'cuda', 'cpu' ou None/'auto' (= GPU se houver)."""
    import torch
    if device in (None, 'auto'):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(device)


def _cpu_state(model):
    """state_dict em CPU: checkpoints gravados na GPU abrem em qualquer máquina."""
    return {k: w.detach().cpu().clone() for k, w in model.state_dict().items()}


def build_model(d_model=128, layers=4, heads=4, bottleneck=64, dropout=0.1, d_in=D):
    import torch
    import torch.nn as nn

    class TemporalAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(d_in, d_model)
            self.tenc = nn.Sequential(nn.Linear(4, d_model), nn.GELU(), nn.Linear(d_model, d_model))
            self.mask_token = nn.Parameter(torch.zeros(d_model))
            layer = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout, batch_first=True,
                                               norm_first=True, activation='gelu')
            self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.neck = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, bottleneck), nn.GELU(),
                                      nn.Linear(bottleneck, d_model), nn.GELU())
            self.out = nn.Linear(d_model, d_in)

        def forward(self, x, valid, tf, hide):
            """x (B, L, F) padronizado; valid (B, L); tf (B, L, 4); hide (B, L): passos substituídos pela máscara.
            Retorna a reconstrução (B, L, F). Passos inválidos e escondidos não servem de contexto a ninguém
            (key padding), mas os escondidos ainda recebem uma saída."""
            h = self.inp(x)
            h = torch.where(hide[..., None], self.mask_token.expand_as(h), h) + self.tenc(tf)
            pad = ~valid | hide
            pad = pad & ~pad.all(1, keepdim=True)        # segurança: ao menos um passo visível
            return self.out(self.neck(self.enc(h, src_key_padding_mask=pad)))

    return TemporalAE()


class Trainer:
    """Treina, guarda a curva de perda e pontua. Entradas padronizadas pelos normais de treino.

    Checkpoint de treino (`resume_path`): ao fim de cada época grava modelo, otimizador, scheduler, estados dos
    geradores aleatórios, curva, melhor estado e contador de paciência. Se o arquivo existir com a mesma
    configuração, o treino continua da época seguinte e chega ao mesmo resultado de um treino sem interrupção.
    Configuração diferente: o arquivo é ignorado (aviso) e o treino recomeça. Apagar o arquivo depois de salvar o
    modelo final é responsabilidade de quem chama (ver `09_ae_combinado`)."""

    def __init__(self, Xtr, Vtr, TFtr, Xva, Vva, TFva, mode='forecast', d_model=128, layers=4, epochs=40,
                 patience=6, lr=1e-3, batch=512, seed=config.SEED, threads=8, log=True, entrada='tm',
                 resume_path=None, device=None):
        import torch
        torch.manual_seed(seed); torch.set_num_threads(threads)
        self.device = _device(device)          # fora de cfg: um checkpoint de CPU continua na GPU e vice-versa
        self.mode, self.log, self.entrada = mode, log, entrada
        d_in = Xtr.shape[-1]
        assert d_in == ENTRADAS[entrada], f'entrada {entrada} espera {ENTRADAS[entrada]} dimensões, veio {d_in}'
        self.groups = grupos(entrada)
        flat = Xtr.reshape(-1, d_in).astype(np.float32)
        self.mu = np.nanmean(flat, 0); self.sd = np.nanstd(flat, 0) + 1e-6
        self.model = build_model(d_model, layers, d_in=d_in).to(self.device)
        self.cfg = dict(mode=mode, d_model=d_model, layers=layers, epochs=epochs, patience=patience, lr=lr,
                        batch=batch, seed=seed, n_train=len(Xtr), n_val=len(Xva), K=K_SEQ,
                        entrada=entrada, d_in=d_in)
        tr = self._tensors(Xtr, Vtr, TFtr); va = self._tensors(Xva, Vva, TFva)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        g = torch.Generator().manual_seed(seed)
        self.curve, best, wait, best_state, start, stopped, t_prev = [], np.inf, 0, None, 0, False, 0.0
        if resume_path is not None and os.path.exists(resume_path):
            ck = torch.load(resume_path, weights_only=False, map_location='cpu')
            if ck['cfg'] == self.cfg:
                self.model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched'])
                g.set_state(ck['gen']); torch.set_rng_state(ck['rng'])
                self.curve, best, wait, best_state = ck['curve'], ck['best'], ck['wait'], ck['best_state']
                start, stopped = ck['epoch'], ck['stopped']
                t_prev = self.curve[-1]['min'] if self.curve else 0.0
                if log:
                    print(f'  retomando do checkpoint: {start} épocas feitas'
                          + (' (parada antecipada já atingida)' if stopped else ''), flush=True)
            elif log:
                print('  checkpoint de treino com outra configuração: ignorado, recomeçando', flush=True)
        t0 = time.time() - t_prev * 60
        for ep in range(start, epochs):
            if stopped:
                break
            self.model.train()
            tl, nb = 0.0, 0
            for idx in torch.randperm(len(tr[0]), generator=g).split(batch):
                x, v, tf = (a[idx].to(self.device) for a in tr)
                hide = self._hide_train(v, g)
                rec = self.model(x, v, tf, hide)
                m = hide & v
                loss = (self._step_err(rec, x) * m).sum() / m.sum().clamp(min=1)
                opt.zero_grad(); loss.backward(); opt.step()
                tl += loss.item(); nb += 1
            sched.step()
            vl = self.val_loss(va)
            self.curve.append({'epoca': ep + 1, 'treino': tl / nb, 'val': vl, 'min': (time.time() - t0) / 60})
            if log:
                print(f'  época {ep + 1:>2}: treino {tl / nb:.4f}  val {vl:.4f}  ({(time.time() - t0) / 60:.1f} min)', flush=True)
            if vl < best - 1e-5:
                best, wait = vl, 0
                best_state = _cpu_state(self.model)
            else:
                wait += 1
                stopped = wait >= patience
            if resume_path is not None:
                _atomic_save({'cfg': self.cfg, 'epoch': ep + 1, 'stopped': stopped, 'model': _cpu_state(self.model),
                              'opt': opt.state_dict(), 'sched': sched.state_dict(), 'gen': g.get_state(),
                              'rng': torch.get_rng_state(), 'curve': self.curve, 'best': best, 'wait': wait,
                              'best_state': best_state}, resume_path)
        self.model.load_state_dict(best_state)
        self.best_val = best

    def _step_err(self, rec, x):
        """Erro por passo: MSE de cada grupo de dimensões, somado com os pesos de `grupos(entrada)`."""
        sq = (rec - x) ** 2
        if len(self.groups) == 1:
            return sq.mean(-1)
        return sum(w * sq[..., g].mean(-1) for g, w in self.groups)

    def _tensors(self, X, V, TF):
        import torch
        Xs = np.nan_to_num((X.astype(np.float32) - self.mu) / self.sd)   # NaN = sem dado; o passo já está mascarado
        return (torch.tensor(Xs), torch.tensor(V), torch.tensor(TF, dtype=torch.float32))

    def _hide_train(self, v, g):
        import torch
        hide = torch.zeros_like(v)
        hide[:, -1] = True                                         # o passo t sempre escondido
        if self.mode == 'masked':                                   # + ~25% dos passos de contexto
            hide[:, :-1] = (torch.rand(v[:, :-1].shape, generator=g) < 0.25).to(v.device) & v[:, :-1]
        return hide

    def _last_error(self, x, v, tf, batch=4096):
        """Erro do passo t (escondido), por sequência. Critério comum às duas formulações.
        NaN se o passo t não tem dado naquela célula."""
        import torch
        self.model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(x), batch):
                xb, vb, tb = (a[i:i + batch].to(self.device) for a in (x, v, tf))
                hide = torch.zeros_like(vb); hide[:, -1] = True
                rec = self.model(xb, vb, tb, hide)
                e = self._step_err(rec[:, -1], xb[:, -1])
                out.append(torch.where(vb[:, -1], e, torch.full_like(e, float('nan'))).cpu())
        return torch.cat(out).numpy()

    def val_loss(self, va):
        return float(np.nanmean(self._last_error(*va)))

    def score(self, X, V, TF):
        """X (n, L, F) -> erro do passo t por sequência."""
        return self._last_error(*self._tensors(X, V, TF))

    def save(self, path):
        _atomic_save({'state': _cpu_state(self.model), 'mu': self.mu, 'sd': self.sd, 'cfg': self.cfg,
                      'curve': self.curve}, path)

    @classmethod
    def load(cls, path, threads=8, device=None):
        """Carrega um checkpoint salvo por `save` só para pontuar (sem treinar). Checkpoints anteriores à
        Fase 4 não têm `entrada`: são 'tm'."""
        import torch
        ck = torch.load(path, weights_only=False, map_location='cpu')
        torch.set_num_threads(threads)
        self = cls.__new__(cls)
        self.device = _device(device)
        self.cfg, self.mode, self.log = ck['cfg'], ck['cfg']['mode'], False
        self.entrada = ck['cfg'].get('entrada', 'tm')
        self.groups = grupos(self.entrada)
        self.mu, self.sd, self.curve = ck['mu'], ck['sd'], ck.get('curve', [])
        self.model = build_model(ck['cfg']['d_model'], ck['cfg']['layers'], d_in=len(ck['mu']))
        self.model.load_state_dict(ck['state'])
        self.model.to(self.device).eval()
        self.best_val = min((c['val'] for c in self.curve), default=np.nan)
        return self


# ---------------- pontuação de todas as amostras ----------------

def score_all(trainer, s, ctx, cfg='s2s1', log=True):
    """Erro de cada célula em cada amostra (patch, t) com t >= K. Retorna erros (n_amostras, 49) e o índice de `s`.
    A entrada (TerraMind, originais ou as duas) vem do próprio modelo."""
    rows = s[s['t'] >= K_SEQ]
    entrada = getattr(trainer, 'entrada', 'tm')
    errs, t0, done = [], time.time(), 0
    for r, X, v, tf in sequences(rows, ctx, cfg, entrada=entrada):
        n, F = len(r), X.shape[-1]
        e = trainer.score(X.reshape(n * NC, K_SEQ + 1, F), v.reshape(n * NC, K_SEQ + 1), np.repeat(tf, NC, axis=0))
        errs.append(pd.DataFrame(e.reshape(n, NC), index=r.index))
        done += n
        if log and done % 2000 < n:
            print(f'  {done}/{len(rows)} amostras  {(time.time() - t0) / 60:.1f} min', flush=True)
    return pd.concat(errs).reindex(rows.index)


def local_normalize(err, s, ref=None):
    """PLANO 6.4: z-score do erro de cada célula pelos erros das células normais de validação da mesma
    cobertura do patch (floresta, pastagem, ...). Coberturas sem normais de validação usam todos os normais.
    ref = (err_ref, s_ref): tira as referências de outra AOI (Rondônia não tem `val`: usa as do Tapajós)."""
    rows = s.loc[err.index]
    ref_err, ref_rows = (err, rows) if ref is None else (ref[0], ref[1].loc[ref[0].index])
    vn = ref_rows['val_normal'].values
    ref_v = ref_err.values[vn]
    covers = rows['cobertura'].values
    allmu, allsd = np.nanmean(ref_v), np.nanstd(ref_v) + 1e-9
    out = np.empty_like(err.values)
    for c in np.unique(covers):
        m = covers == c
        rc = ref_err.values[vn & (ref_rows['cobertura'].values == c)]
        ok = np.count_nonzero(~np.isnan(rc)) >= 49 * 20
        mu, sd = (np.nanmean(rc), np.nanstd(rc) + 1e-9) if ok else (allmu, allsd)
        out[m] = (err.values[m] - mu) / sd
    return pd.DataFrame(out, index=err.index)


def patch_scores(err, s, name, ref=None):
    """Agrega as células: máximo (principal) e média, com e sem normalização local (ignora células sem dado)."""
    z = local_normalize(err, s, ref)
    return pd.DataFrame({f'{name}': err.max(axis=1), f'{name} [média]': err.mean(axis=1),
                         f'{name} + norm. local': z.max(axis=1), f'{name} + norm. local [média]': z.mean(axis=1)},
                        index=err.index)
