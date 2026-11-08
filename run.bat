@echo off
chcp 65001 >nul
rem Индексация датасета и запуск интерфейса.
rem Можно передать свою папку: run.bat путь\к\папке
set "target_path=%~1"
if "%target_path%"=="" set "target_path=dataset"
python index.py "%target_path%"
if errorlevel 1 exit /b 1
streamlit run app.py
