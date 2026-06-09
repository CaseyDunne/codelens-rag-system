@echo off
set /p target_path="Введите имя файла или директории для индексации: "
python index.py "%target_path%"
streamlit run app.py