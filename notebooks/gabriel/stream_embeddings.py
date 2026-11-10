"""Embeddings TerraMind das AOIs completas, em fluxo (Etapa 4). Para rodar de noite, num terminal:

    cd notebooks/gabriel
    ..\\..\\venv\\Scripts\\python stream_embeddings.py tapajos
    ..\\..\\venv\\Scripts\\python stream_embeddings.py norte_ro

Retomável: interrompa com Ctrl+C a qualquer momento (salva a cada 5 min) e rode de novo para continuar.
Os blocos vão em ordem aleatória fixa, então o que já terminou é uma amostra aleatória da AOI.
Saída: embeddings/<aoi>.npz (formato descrito em embeddings.py).
Para a Fase 5 inteira (download, ajuste, pontuação e avaliação, com checkpoints) use fase5_execucao_completa/fase5.py.
"""
import argparse

import config
import embeddings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('aoi', choices=list(config.AOIS))
    ap.add_argument('--workers', type=int, default=10, help='downloads simultâneos (o cliente do Earth Engine mantém 10 conexões)')
    ap.add_argument('--max-tasks', type=int, default=None, help='limita (bloco, trimestre) nesta sessão (teste)')
    ap.add_argument('--threads', type=int, default=8, help='threads de CPU do PyTorch')
    ap.add_argument('--device', default='auto', help="'cuda', 'cpu' ou 'auto' (GPU se houver)")
    a = ap.parse_args()

    import dados
    print('Earth Engine:', dados.init_ee())
    config.make_dirs()
    out = config.EMBEDDINGS_DIR / f'{a.aoi}.npz'
    embeddings.stream_aoi(a.aoi, out, workers=a.workers, model=embeddings.load_model(a.threads, a.device),
                          max_tasks=a.max_tasks)


if __name__ == '__main__':
    main()
