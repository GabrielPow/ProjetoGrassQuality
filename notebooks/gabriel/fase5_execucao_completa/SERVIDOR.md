# Fase 5: execução completa na máquina dedicada (Windows + GPU)

Passo a passo para rodar a Fase 5 (PLANO_JUNTAR_DEEP_LEARNING.md, seção 8) na máquina de deep learning, sem acesso
ao GitHub. Esta pasta (`notebooks/gabriel/fase5_execucao_completa/`) é a **execução completa**, nas AOIs inteiras;
os notebooks de `notebooks/gabriel/` são a sub-AOI. Tudo roda por um único script, `fase5.py`, e é **retomável**: se o processo cair,
basta rodar o mesmo comando de novo (o `rodar_fase5.bat` faz isso sozinho).

## O que roda e quanto tempo leva

| Etapa | O que faz | Checkpoint | Tempo estimado |
|---|---|---|---|
| `stream_tapajos` | baixa S2 + S1 do Earth Engine por (bloco, trimestre), extrai TerraMind e as features originais | cada (bloco, trimestre) pronto fica marcado; salva a cada 3 min | **30–40 h** (limitado pelo download) |
| `stream_norte_ro` | idem, Rondônia | idem | **30–40 h** |
| `ajuste` | ajusta os detectores e treina AE-TM, AE-ORIG e AE-TM+ORIG (só normais do Tapajós) | detectores salvos; um checkpoint por época de cada AE | 1–3 h na GPU |
| `pontuacao` | pontua as duas AOIs com todos os métodos | partes de 40 blocos, cada uma salva ao terminar | 4–10 h (o kNN roda em CPU) |
| `fusao` | junta tudo e faz a fusão J2 (referência: val normal do Tapajós) | um arquivo por AOI | minutos |
| `avaliacao` | AUROC/AUPRC/TPR com IC por blocos; Tapajós em `test`, Rondônia em `test_gen` | um arquivo por AOI | ~1–2 h |

A GPU acelera o TerraMind e o treino, mas **o gargalo é o download do Earth Engine**, que depende da rede e da
cota. O teste rápido (passo 6) mede a velocidade real nesta máquina.

## 1. Preparar o pacote (no notebook, antes de levar)

```
cd notebooks\gabriel\fase5_execucao_completa
..\..\..\venv\Scripts\python empacotar_servidor.py
```

O script gera `ProjetoGrassQuality_servidor_<data>.zip` na raiz do repositório com o código, os rótulos (`rotulos/`,
que o git ignora) e um `VERSAO.txt` com o commit. Commite antes, para o `VERSAO.txt` apontar um commit exato.
**Não regenere os rótulos na máquina:** o INPE edita o PRODES e o DETER depois de publicar, e o protocolo usa
esta versão baixada.

Leve o zip para a máquina (pen drive, rede interna) e descompacte, por exemplo em `D:\ProjetoGrassQuality`.

## 2. Python e dependências (na máquina, uma vez)

Python **3.13** (o mesmo do notebook). No PowerShell, na pasta do projeto:

```
py -3.13 -m venv venv
venv\Scripts\python -m pip install --upgrade pip
```

Instale primeiro o **torch com CUDA**, na mesma versão do `requirements-local.txt` (2.14.0). O comando exato
depende da versão do CUDA do driver (`nvidia-smi` mostra "CUDA Version"); pegue-o no seletor de
https://pytorch.org/get-started/locally/ (Windows, Pip, CUDA). Por exemplo:

```
venv\Scripts\python -m pip install torch==2.14.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cu128
```

Depois, o resto (o pip mantém o torch com CUDA, porque a versão bate):

```
venv\Scripts\python -m pip install -r notebooks\gabriel\requirements-local.txt
venv\Scripts\python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

A última linha precisa mostrar `True` e o nome da GPU.

Os pesos do TerraMind são baixados do Hugging Face na primeira execução. Se a máquina não acessar o
Hugging Face, copie a pasta `%USERPROFILE%\.cache\huggingface` do notebook para o mesmo lugar na máquina.

## 3. Conta de serviço do Earth Engine

1. Copie o JSON da chave para a pasta `chaves\` na raiz do projeto, por exemplo
   `D:\ProjetoGrassQuality\chaves\gee.json`. O código acha sozinho o único `.json` dessa pasta. Ela é ignorada
   pelo git e fica fora do zip do `empacotar_servidor.py`, então **a chave vai para a máquina à parte** (pen drive),
   nunca junto com o pacote.
2. A conta de serviço precisa estar registrada no Earth Engine e ter, no projeto do Cloud, os papéis
   *Earth Engine Resource Viewer* e *Service Usage Consumer*.
3. Se a conta de serviço for de outro projeto do Cloud, defina o projeto (uma vez, vale para novos terminais):

```
setx GEE_PROJECT "projetograssquality"
```

Para usar uma chave em outro lugar, `setx GEE_SERVICE_ACCOUNT_KEY "C:\caminho\gee.json"` tem prioridade sobre
`chaves\`. Teste:

```
cd notebooks\gabriel
..\..\venv\Scripts\python -c "import dados; print(dados.init_ee())"
cd fase5_execucao_completa
```

## 4. (Opcional) Dados em outro disco

Por padrão, rótulos, embeddings, checkpoints e resultados ficam dentro da pasta do projeto. Para usar outro disco
(precisa de **~60 GB livres**), copie `rotulos\` para lá e defina:

```
setx GRASS_DATA_ROOT "E:\grass_dados"
```

## 5. Configurar o Windows para não dormir

A execução leva dias. Em Configurações > Energia, defina suspensão como "Nunca" e, se der, adie as
reinicializações do Windows Update durante a execução. Se o Windows reiniciar mesmo assim, basta rodar de novo
(passo 7): nada se perde além dos últimos minutos.

## 6. Teste rápido (obrigatório antes da execução longa)

```
cd notebooks\gabriel\fase5_execucao_completa
rodar_fase5.bat --teste
```

O teste roda a cadeia inteira em 9 blocos (7 do Tapajós, 2 de Rondônia), com 2 épocas por AE, e grava em
`results\fase5_teste` e `checkpoints\fase5_teste`, separado da execução de verdade. Leva **~30–60 min**, quase
tudo download. Confira:

- o log mostra `GPU: <nome da placa>` e `TerraMind em cuda:0`;
- a linha `N/252 nesta sessão | ... | ~X h restantes` dá a velocidade do download: multiplique para
  ~17.000 (bloco, trimestre) por AOI e terá a estimativa real;
- termina com `Fase 5: tudo pronto` e `results\fase5_teste\resumo.json` existe (os números do teste não valem nada:
  são poucos blocos).

Teste também a retomada: rode `rodar_fase5.bat --teste --refazer ajuste`, interrompa com Ctrl+C durante o treino de
um AE e rode `rodar_fase5.bat --teste` de novo. O log deve mostrar `retomando do checkpoint: N épocas feitas`.

Para medir só a velocidade de download com mais conexões: `..\..\..\venv\Scripts\python fase5.py --teste --workers 16
--max-tasks 60`. Se a velocidade subir sem aparecerem falhas `429` no log, use esse `--workers` na execução longa.

## 7. Execução completa

```
cd notebooks\gabriel\fase5_execucao_completa
rodar_fase5.bat
```

(ou `rodar_fase5.bat --workers 16`, se o teste mostrou ganho). Deixe a janela aberta. Para acompanhar de outro
terminal:

```
..\..\..\venv\Scripts\python fase5.py --status
```

Os logs completos ficam em `results\fase5_execucao_completa\logs\`.

### Se algo der errado

- **Caiu (rede, Earth Engine, falta de energia, memória):** o `.bat` espera 2 min e roda de novo, até 50 vezes. Se a
  própria janela foi fechada ou o Windows reiniciou, rode `rodar_fase5.bat` outra vez. Tudo continua de onde parou.
- **Erro de configuração** (código 3: rótulos faltando, chave não encontrada, GPU pedida e ausente): o `.bat` para e
  mostra a mensagem. Corrija e rode de novo.
- **Downloads falhando:** cada falha aparece no log como `falha no download (...)`; o (bloco, trimestre) fica
  pendente e é refeito na passada seguinte. Mais de 200 falhas numa sessão param o processo (rede ou cota fora
  do ar), e o `.bat` tenta de novo 2 min depois.
- **Refazer uma etapa** (ex.: depois de corrigir código): `rodar_fase5.bat --refazer ajuste` apaga a saída dessa etapa
  e das seguintes. O download nunca é apagado por esse caminho.

### Para ganhar tempo (opcional)

Depois que `stream_tapajos` terminar, dá para treinar enquanto Rondônia baixa. Abra um segundo terminal e rode
`..\..\..\venv\Scripts\python fase5.py --etapas ajuste`, enquanto o primeiro segue no `stream_norte_ro`. Não rode a mesma
etapa em dois terminais ao mesmo tempo.

## 8. O que trazer de volta

| Pasta | Tamanho | Para quê |
|---|---|---|
| `results\fase5_execucao_completa\` | MBs | scores, métricas, comparações pareadas, curvas, logs: **o resultado** |
| `checkpoints\fase5_execucao_completa\ae_*.pt`, `baselines_modelos.pkl` | ~100 MB | modelos treinados (abrem em CPU no notebook) |
| `embeddings\tapajos.npz`, `embeddings\norte_ro.npz` | ~0,5 GB cada | embeddings de patch, para figuras |
| `embeddings\*_cells.zarr` | ~20 GB cada | só se for refazer análises por célula no notebook |

O `resumo.json` tem a AUROC de cada método nas duas AOIs e as diferenças pareadas, já no formato dos slides.
