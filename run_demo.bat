@echo off
chcp 65001 > nul
title SmartRoad AI - Demo Web

echo =====================================================================
echo                🚦 SMARTROAD AI - PLATAFORMA DEMO
echo =====================================================================
echo Iniciando el servidor web de Streamlit para pruebas de modelos...
echo.

set PYTHON_ANACONDA=C:\Users\omarg\anaconda3\python.exe

if exist "%PYTHON_ANACONDA%" (
    echo [OK] Usando entorno Python Anaconda detectado: %PYTHON_ANACONDA%
    "%PYTHON_ANACONDA%" -m streamlit run scripts\app_demo.py --server.port 8501
) else (
    echo [INFO] Buscando python en el sistema...
    python -m streamlit run scripts\app_demo.py --server.port 8501
)

pause
