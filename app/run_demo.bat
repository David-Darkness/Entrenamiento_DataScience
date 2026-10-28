@echo off
setlocal enabledelayedexpansion

REM SmartRoad AI — Lanzador para Windows.
REM Uso desde la raíz del proyecto: .\app\run_demo.bat

REM Cambiar al directorio donde se encuentra este script (app\)
cd /d "%~dp0"

echo ============================================
echo  SmartRoad AI - Comparacion Multimodelo
echo ============================================
echo.

REM Identificar Python disponible en el entorno activo
set "PYTHON_CMD="
for %%P in (python python3 py) do (
    if not defined PYTHON_CMD (
        %%P --version >nul 2>&1
        if !errorlevel! equ 0 (
            set "PYTHON_CMD=%%P"
        )
    )
)

if not defined PYTHON_CMD (
    echo [ERROR] No se encontro Python en el entorno activo.
    echo.
    echo Posibles soluciones:
    echo   1. Activa el entorno Conda del proyecto:
    echo        conda activate entrenamiento-proyecto
    echo   2. Asegurate de que Python este en el PATH.
    echo.
    pause
    exit /b 1
)

echo [OK] Python detectado: %PYTHON_CMD%
%PYTHON_CMD% --version

REM Verificar dependencias criticas
echo.
echo Verificando dependencias criticas...
%PYTHON_CMD% -c "import streamlit, pandas, plotly, PIL, numpy, cv2, torch, ultralytics" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Faltan dependencias de la aplicacion.
    echo.
    echo Instala los paquetes necesarios con:
    echo        %PYTHON_CMD% -m pip install -r app\requirements.txt
    echo.
    echo Si faltan PyTorch o Ultralytics, activa primero el entorno de entrenamiento.
    echo.
    pause
    exit /b 1
)

echo [OK] Dependencias criticas disponibles.
echo.

REM Iniciar Streamlit
echo Iniciando SmartRoad AI en http://localhost:8501 ...
echo.

%PYTHON_CMD% -m streamlit run app.py

if errorlevel 1 (
    echo.
    echo [ERROR] Streamlit no pudo iniciar. Revisa los mensajes de error arriba.
    pause
    exit /b 1
)

endlocal
