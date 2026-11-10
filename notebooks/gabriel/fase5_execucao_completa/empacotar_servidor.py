"""Monta o pacote da Fase 5 para a máquina dedicada, que não acessa o GitHub (ver SERVIDOR.md).

    cd notebooks\\gabriel\\fase5_execucao_completa
    ..\\..\\..\\venv\\Scripts\\python empacotar_servidor.py            # -> ProjetoGrassQuality_servidor_<data>.zip na raiz

Leva: todos os arquivos do repositório que o git conhece (versionados e novos não ignorados), mais os rótulos
(rotulos/<aoi>/, ignorados pelo git) e um VERSAO.txt com o commit, para os resultados saberem de que código vieram.
NÃO leva: venv, embeddings, checkpoints, cubos, nem a chave da conta de serviço (chaves/*.json): copie-a à parte.
"""
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # módulos de notebooks/gabriel
import config  # noqa: E402

ROTULOS = ['grade.csv', 'rotulos_anuais.csv.gz', 'rotulos_trimestrais.csv.gz', 'eventos.csv.gz']
PROIBIDO = ('.json',)          # chaves de conta de serviço nunca entram, mesmo que estejam na pasta


def git(*args):
    return subprocess.run(['git', *args], cwd=config.REPO_ROOT, capture_output=True, text=True, check=True).stdout


def main():
    root = config.REPO_ROOT
    arquivos = [f for f in git('ls-files', '--cached', '--others', '--exclude-standard').splitlines()
                if (root / f).is_file() and not (f.startswith('chaves/') and f.endswith('.json'))]
    suspeitos = [f for f in arquivos if f.lower().endswith(PROIBIDO)
                 and any(w in f.lower() for w in ('key', 'chave', 'servico', 'service', 'credential'))]
    if suspeitos:
        sys.exit(f'Parece haver uma chave no repositório, não empacotei: {suspeitos}')
    commit = git('rev-parse', 'HEAD').strip()
    sujo = bool(git('status', '--porcelain').strip())
    versao = (f'commit {commit}{" + alterações não commitadas" if sujo else ""}\n'
              f'empacotado em {datetime.now():%Y-%m-%d %H:%M}\n')
    out = root / f'ProjetoGrassQuality_servidor_{datetime.now():%Y%m%d}.zip'
    n_rot = 0
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in arquivos:
            z.write(root / f, f'ProjetoGrassQuality/{f}')
        for aoi in config.AOIS:
            for f in ROTULOS:
                src = config.LABELS_DIR / aoi / f
                if not src.exists():
                    sys.exit(f'falta {src}: rode o 02_rotulos antes de empacotar')
                z.write(src, f'ProjetoGrassQuality/rotulos/{aoi}/{f}')
                n_rot += 1
        z.writestr('ProjetoGrassQuality/VERSAO.txt', versao)
    print(f'{out}  ({out.stat().st_size / 1e6:.0f} MB; {len(arquivos)} arquivos do repositório + {n_rot} de rótulos)')
    print(versao.strip())
    if sujo:
        print('AVISO: há alterações não commitadas; considere commitar antes, para o VERSAO.txt apontar um commit exato.')


if __name__ == '__main__':
    main()
