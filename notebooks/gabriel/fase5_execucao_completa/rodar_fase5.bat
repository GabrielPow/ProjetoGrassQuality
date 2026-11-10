@echo off
REM Roda fase5.py e, se cair (rede, Earth Engine, memoria), espera 2 min e roda de novo.
REM Tudo e retomavel, entao cada nova tentativa continua de onde a anterior parou.
REM Uso (duplo clique ou no terminal):  rodar_fase5.bat            -> Fase 5 inteira
REM                                     rodar_fase5.bat --teste    -> teste rapido
REM Qualquer argumento e repassado ao fase5.py. Para parar: Ctrl+C (e responda S).
REM Codigos do fase5.py: 0 pronto, 1 erro (tenta de novo), 3 configuracao (para), 130 Ctrl+C (para).
setlocal
cd /d "%~dp0"
set "PY=%~dp0..\..\..\venv\Scripts\python.exe"
if not exist "%PY%" (
    echo Nao achei o Python do venv em %PY%. Veja SERVIDOR.md, passo 2.
    pause
    exit /b 3
)
set MAX=50
set N=0

:loop
"%PY%" fase5.py %*
set CODE=%ERRORLEVEL%
if %CODE%==0 goto pronto
if %CODE%==3 goto config
if %CODE%==130 goto parado
set /a N+=1
if %N% GEQ %MAX% goto desiste
echo.
echo [rodar_fase5] fase5.py saiu com codigo %CODE%. Tentativa %N% de %MAX%; recomecando em 2 minutos...
timeout /t 120 /nobreak >nul
goto loop

:pronto
echo [rodar_fase5] Pronto. Resultados em results\fase5_execucao_completa (ou results\fase5_teste).
goto fim
:config
echo [rodar_fase5] Erro de configuracao (veja a mensagem acima). Rodar de novo nao resolve; corrija e rode outra vez.
goto fim
:parado
echo [rodar_fase5] Interrompido pelo usuario. Rode de novo para continuar.
goto fim
:desiste
echo [rodar_fase5] %MAX% falhas seguidas. Veja o log em results\fase5_execucao_completa\logs.
:fim
pause
endlocal
