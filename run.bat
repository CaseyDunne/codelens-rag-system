@echo off
chcp 65001 >nul
set /p target_path="Введите путь к папке для индексации: "
python index.py "%target_path%"
if errorlevel 1 exit /b 1
streamlit run app.py
