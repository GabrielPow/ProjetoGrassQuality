# chaves/

Coloque aqui o JSON da conta de serviço do Earth Engine (um arquivo só, por exemplo `gee.json`).
`dados.init_ee()` acha esse arquivo sozinho; a variável `GEE_SERVICE_ACCOUNT_KEY` tem prioridade, se definida.

Tudo nesta pasta, exceto este LEIAME, é ignorado pelo git e fica fora do pacote do servidor
(`notebooks/gabriel/fase5_execucao_completa/empacotar_servidor.py`). Leve a chave para a máquina à parte.
Nunca a envie por e-mail ou chat nem a coloque em outra pasta versionada.
