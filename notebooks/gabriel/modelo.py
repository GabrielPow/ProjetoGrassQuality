"""Etapa 6: Temporal Transformer Autoencoder sobre os embeddings TerraMind das células de 320 m.

Entrada: sequência Z(t-K), ..., Z(t) de uma célula (K = 8 trimestres = 2 anos), com codificação temporal
(ano e trimestre do ano em seno/cosseno: a sazonalidade faz parte do "normal") e máscara de atenção
para trimestres sem dado.

O modelo nunca vê Z(t) ao produzir Ẑ(t) (PLANO 6.2), em duas formulações com a mesma arquitetura:
- 'forecast': o passo t é sempre substituído pelo token de máscara; prevê Z(t) a partir de t-K..t-1.
- 'masked':   no treino, passos aleatórios (inclusive t) viram máscara e são reconstruídos com o contexto.
Na pontuação, as duas mascaram só o passo t. A escolha entre elas usa a perda na validação de normais
(sem rótulos de anomalia).

Score da célula = erro quadrático médio entre Ẑ(t) e Z(t) (embeddings padronizados pelos normais de
treino). Score do patch = máximo das 49 células. Normalização local (PLANO 6.4): z-score do erro em relação
aos erros das células normais de validação da mesma cobertura do solo.
"""
import math
import time

import numpy as np
import pandas as pd

import config
import embeddings

K_SEQ = 8                          # trimestres de contexto
NC = config.N_CELLS
D = config.TERRAMIND_EMBED_DIM


# ---------------- dados: sequências por célula ----------------

def time_features(step_idx):
    """(..., L) índices de passo -> (..., L, 4): ano normalizado, sen/cos do trimestre, posição relativa."""
    steps = np.asarray(config.STEPS)
    y = np.vectorize(lambda i: int(steps[i][:4]))(step_idx).astype(np.float32)
    q = np.vectorize(lambda i: int(steps[i][-1]))(step_idx).astype(np.float32)
    rel = (step_idx - step_idx[..., -1:]).astype(np.float32) / K_SEQ
    return np.stack([(y - 2022) / 3, np.sin(2 * math.pi * q / 4), np.cos(2 * math.pi * q / 4), rel], axis=-1)


def sequences(rows, ctx, cfg='s2s1', batch_slots=64):
    """Para cada amostra (patch, t) de `rows`: sequências das 49 células, t-K..t.
    Gera (rows do lote, X (n, 49, K+1, D) float32, válido (n, K+1) bool, tempo (n, K+1, 4))."""
    cells, valid = ctx['cells'], ctx['valid']
    for b0 in range(0, ctx['n_slots'], batch_slots):
        r = rows[(rows['i'] >= b0) & (rows['i'] < b0 + batch_slots)]
        if r.empty:
            continue
        C = cells[f'c_{cfg}'][b0:b0 + batch_slots]                       # (b, T, 49, D) float16
        li, tt = r['i'].values - b0, r['t'].values
        idx = tt[:, None] + np.arange(-K_SEQ, 1)[None]                    # (n, K+1)
        X = C[li[:, None], idx].transpose(0, 2, 1, 3).astype(np.float32)  # (n, 49, K+1, D)
        v = valid[b0:b0 + batch_slots][li[:, None], idx]
        yield r, X, v, time_features(idx)


def gather_training(s, ctx, col, n_max, cfg='s2s1', seed=config.SEED):
    """Sorteia até n_max sequências de células das amostras com `col` (treino ou validação normal)."""
    rows = s[s[col] & (s['t'] >= K_SEQ)]
    p = min(1.0, n_max / max(1, len(rows) * NC))
    rng = np.random.default_rng(seed)
    X, V, TF = [], [], []
    for r, x, v, tf in sequences(rows, ctx, cfg):
        pick = rng.random((len(r), NC)) < p
        n_i, c_i = np.nonzero(pick)
        X.append(x[n_i, c_i].astype(np.float16)); V.append(v[n_i]); TF.append(tf[n_i])
    return np.concatenate(X), np.concatenate(V), np.concatenate(TF)


# ---------------- modelo ----------------

def build_model(d_model=128, layers=4, heads=4, bottleneck=64, dropout=0.1):
    import torch
    import torch.nn as nn

    class TemporalAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(D, d_model)
            self.tenc = nn.Sequential(nn.Linear(4, d_model), nn.GELU(), nn.Linear(d_model, d_model))
            self.mask_token = nn.Parameter(torch.zeros(d_model))
            layer = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout, batch_first=True,
                                               norm_first=True, activation='gelu')
            self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.neck = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, bottleneck), nn.GELU(),
                                      nn.Linear(bottleneck, d_model), nn.GELU())
            self.out = nn.Linear(d_model, D)

        def forward(self, x, valid, tf, hide):
            """x (B, L, D) padronizado; valid (B, L); tf (B, L, 4); hide (B, L): passos substituídos pela máscara.
            Retorna a reconstrução (B, L, D). Passos inválidos e escondidos não servem de contexto a ninguém
            (key padding), mas os escondidos ainda recebem uma saída."""
            h = self.inp(x)
            h = torch.where(hide[..., None], self.mask_token.expand_as(h), h) + self.tenc(tf)
            pad = ~valid | hide
            pad = pad & ~pad.all(1, keepdim=True)        # segurança: ao menos um passo visível
            return self.out(self.neck(self.enc(h, src_key_padding_mask=pad)))

    return TemporalAE()


class Trainer:
    """Treina, guarda a curva de perda e pontua. Embeddings padronizados pelos normais de treino."""

    def __init__(self, Xtr, Vtr, TFtr, Xva, Vva, TFva, mode='forecast', d_model=128, layers=4, epochs=40,
                 patience=6, lr=1e-3, batch=512, seed=config.SEED, threads=8, log=True):
        import torch
        torch.manual_seed(seed); torch.set_num_threads(threads)
        self.mode, self.log = mode, log
        flat = Xtr.reshape(-1, D).astype(np.float32)
        self.mu = flat.mean(0); self.sd = flat.std(0) + 1e-6
        self.model = build_model(d_model, layers)
        self.cfg = dict(mode=mode, d_model=d_model, layers=layers, epochs=epochs, patience=patience, lr=lr,
                        batch=batch, seed=seed, n_train=len(Xtr), n_val=len(Xva), K=K_SEQ)
        tr = self._tensors(Xtr, Vtr, TFtr); va = self._tensors(Xva, Vva, TFva)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        g = torch.Generator().manual_seed(seed)
        self.curve, best, wait, best_state = [], np.inf, 0, None
        t0 = time.time()
        for ep in range(epochs):
            self.model.train()
            tl, nb = 0.0, 0
            for idx in torch.randperm(len(tr[0]), generator=g).split(batch):
                x, v, tf = (a[idx] for a in tr)
                hide = self._hide_train(v, g)
                rec = self.model(x, v, tf, hide)
                m = hide & v
                loss = (((rec - x) ** 2).mean(-1) * m).sum() / m.sum().clamp(min=1)
                opt.zero_grad(); loss.backward(); opt.step()
                tl += loss.item(); nb += 1
            sched.step()
            vl = self.val_loss(va)
            self.curve.append({'epoca': ep + 1, 'treino': tl / nb, 'val': vl, 'min': (time.time() - t0) / 60})
            if log:
                print(f'  época {ep + 1:>2}: treino {tl / nb:.4f}  val {vl:.4f}  ({(time.time() - t0) / 60:.1f} min)', flush=True)
            if vl < best - 1e-5:
                best, wait = vl, 0
                best_state = {k: w.clone() for k, w in self.model.state_dict().items()}
            else:
                wait += 1
                if wait >= patience:
                    break
        self.model.load_state_dict(best_state)
        self.best_val = best

    def _tensors(self, X, V, TF):
        import torch
        Xs = (X.astype(np.float32) - self.mu) / self.sd
        return (torch.tensor(Xs), torch.tensor(V), torch.tensor(TF, dtype=torch.float32))

    def _hide_train(self, v, g):
        import torch
        hide = torch.zeros_like(v)
        hide[:, -1] = True                                         # o passo t sempre escondido
        if self.mode == 'masked':                                   # + ~25% dos passos de contexto
            hide[:, :-1] = (torch.rand(v[:, :-1].shape, generator=g) < 0.25) & v[:, :-1]
        return hide

    def _last_error(self, x, v, tf, batch=4096):
        """Erro quadrático médio do passo t (escondido), por sequência. Critério comum às duas formulações."""
        import torch
        self.model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(x), batch):
                xb, vb, tb = x[i:i + batch], v[i:i + batch], tf[i:i + batch]
                hide = torch.zeros_like(vb); hide[:, -1] = True
                rec = self.model(xb, vb, tb, hide)
                out.append(((rec[:, -1] - xb[:, -1]) ** 2).mean(-1))
        return torch.cat(out).numpy()

    def val_loss(self, va):
        return float(self._last_error(*va).mean())

    def score(self, X, V, TF):
        """X (n, L, D) -> erro do passo t por sequência."""
        return self._last_error(*self._tensors(X, V, TF))

    def save(self, path):
        import torch
        torch.save({'state': self.model.state_dict(), 'mu': self.mu, 'sd': self.sd, 'cfg': self.cfg,
                    'curve': self.curve}, path)


# ---------------- pontuação de todas as amostras ----------------

def score_all(trainer, s, ctx, cfg='s2s1', log=True):
    """Erro de cada célula em cada amostra (patch, t) com t >= K. Retorna erros (n_amostras, 49) e o índice de `s`."""
    rows = s[s['t'] >= K_SEQ]
    errs, t0, done = [], time.time(), 0
    for r, X, v, tf in sequences(rows, ctx, cfg):
        n = len(r)
        e = trainer.score(X.reshape(n * NC, K_SEQ + 1, D), np.repeat(v, NC, axis=0), np.repeat(tf, NC, axis=0))
        errs.append(pd.DataFrame(e.reshape(n, NC), index=r.index))
        done += n
        if log and done % 2000 < n:
            print(f'  {done}/{len(rows)} amostras  {(time.time() - t0) / 60:.1f} min', flush=True)
    return pd.concat(errs).reindex(rows.index)


def local_normalize(err, s):
    """PLANO 6.4: z-score do erro de cada célula pelos erros das células normais de validação da mesma
    cobertura do patch (floresta, pastagem, ...). Coberturas sem normais de validação usam todos os normais."""
    rows = s.loc[err.index]
    ref = err[rows['val_normal'].values].values
    covers = rows['cobertura'].values
    allmu, allsd = ref.mean(), ref.std() + 1e-9
    out = np.empty_like(err.values)
    for c in np.unique(covers):
        m = covers == c
        rc = err[(rows['val_normal'] & (rows['cobertura'] == c)).values].values
        mu, sd = (rc.mean(), rc.std() + 1e-9) if rc.size >= 49 * 20 else (allmu, allsd)
        out[m] = (err.values[m] - mu) / sd
    return pd.DataFrame(out, index=err.index)


def patch_scores(err, s, name):
    """Agrega as células: máximo (principal) e média, com e sem normalização local."""
    z = local_normalize(err, s)
    return pd.DataFrame({f'{name}': err.max(1), f'{name} [média]': err.mean(1),
                         f'{name} + norm. local': z.max(1), f'{name} + norm. local [média]': z.mean(1)},
                        index=err.index)
