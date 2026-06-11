####ИМПОРТ##################################################
import search_engine
import streamlit as st
import pandas as pd
import time
import json

####СТРАНИЦА##################################################

st.set_page_config(
    page_title="CodeLens RAG",
    page_icon="🔍",
    layout="wide"
)

####КЭШИРОВАНИЕ БД##################################################
@st.cache_resource
def init_search():
    return search_engine.init_db()

# custom_css = """
# <style>
#     /* 1. Загружаем крутой шрифт */
#     @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');

#     /* 2. Тотальный контроль: применяем ко ВСЕМ текстовым элементам (включая кнопки, карточки и ввод) */
#     html, body, p, div, span, button, input, textarea, label, h1, h2, h3, h4, h5, h6 {
#         font-family: 'Share Tech Mono', monospace !important;
#     }
    
#     /* 3. УМНАЯ БРОНЯ: Исключения только для иконок Streamlit (чтобы стрелочки и крестики жили) */
#     .material-symbols-rounded, 
#     .stIcon, 
#     svg, 
#     svg *, 
#     [data-testid="collapsedControl"], 
#     [data-testid="collapsedControl"] *, 
#     [data-testid="stIconMaterial"] {
#         font-family: 'Material Symbols Rounded', sans-serif !important;
#     }
    
#     /* 4. Неоновый заголовок */
#     h1 {
#         color: #D5FF40 !important;
#         text-shadow: 0px 0px 10px rgba(213, 255, 64, 0.5);
#     }
# </style>
# """

# st.markdown(custom_css, unsafe_allow_html=True)

# ВЫЗОВ ИНИЦИАЛИЗАЦИИ ПРИ ЗАПУСКЕ ПРИЛОЖЕНИЯ
init_search()

####ФУНКЦИИ И ДАННЫЕ##################################################
def load_metrics():
    try:
        with open("metrics.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        # Если файла вообще нет
        return {"total_accuracy": 0, "ru_accuracy": 0, "en_accuracy": 0}
    except json.JSONDecodeError:
        return {"total_accuracy": 0, "ru_accuracy": 0, "en_accuracy": 0}
    
#results = search_engine.search(user_query, top_k)
results= [
    {
    "path": "laavkz",
    "type": "class",
    "name": "User",
    "percent": 67,
    "code": "class User:"
    },
    {
    "path": "laavkz",
    "type": "class",
    "name": "User",
    "percent": 67,
    "code": "class User:"
    },
    {
    "path": "laavkz",
    "type": "class",
    "name": "User",
    "percent": 67,
    "code": "class User:"
    },
    {
    "path": "laavkz",
    "type": "class",
    "name": "User",
    "percent": 67,
    "code": "class User:"
    },
    {
    "path": "laavkz",
    "type": "class",
    "name": "User",
    "percent": 67,
    "code": "class User:"
    }
    ]

####БОКОВАЯ ПАНЕЛЬ##################################################
with st.sidebar:
    st.header("⚙️ Настройки поиска")
    top_k = st.slider(
        label="Количество результатов:", 
            min_value=1,                   
            max_value=10,                  
            value=5)
    gibrid = st.checkbox('Гибридный поиск')
    LLMchat = st.checkbox('LLM режим')

####ЦЕНТР##################################################

st.title('CodeLens: Умный поиск по коду')
tab_search, tab_metrics = st.tabs(["🔍 Поиск", "📊 Метрики"])

# TAB SEARCH
with tab_search:
    st.write("Введите запрос, чтобы найти нужный фрагмент кода.")
    user_query = st.text_input("Например: как здесь обрабатываются ошибки авторизации?")
    if st.button("Найти"):
        if user_query=="":
            st.warning('Ошибка валидации ввода. Пожалуйста введите запрос.', icon="⚠️")
        else:
            try:
                with st.spinner("Запрос обрабатывается, подождите ...", show_time=True):
                    results = search_engine.search(
                        query=user_query,
                        top_k=top_k,
                        hybrid=gibrid,
                        alpha=0.6
                    )
                st.write(f"Вы искали: {user_query}")
                st.subheader(f"Топ-{top_k} результатов:")
                for result in results[:top_k]:
                    path = result["path"]
                    type = result["type"]
                    name = result["name"]
                    percent = result["percent"]
                    code = result["code"]
                    
                    # Обновленная структура карточки с иконкой пазла и текстом "Точность"
                    card_title = f"📂 {path} | 🧩 {type}: {name} | 🎯 Точность: {percent}%"
                    with st.expander(card_title):
                        st.code(code, language="python")
                        if gibrid:
                            st.caption("Этот фрагмент найден гибридным поиском")
                        else:
                            st.caption("Этот фрагмент найден стандартным поиском")
            except Exception as e:
                st.error(f"Произошла ошибка: {e}", icon="❌")

# TAB METRICS
with tab_metrics:
    st.header("📊 Аналитика и качество поиска")
    metrics_data = load_metrics() 
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Если ключа нет, .get вернет "0"
        total = metrics_data.get('total_accuracy', 0)
        st.metric("Общая точность", f"{total}%")
        
    with col2:
        ru_acc = metrics_data.get('ru_accuracy', 0)
        st.metric("Русский язык", f"{ru_acc}%")
        
    with col3:
        en_acc = metrics_data.get('en_accuracy', 0)
        st.metric("Английский язык", f"{en_acc}%")
        
    st.divider() 
    
    # 3. РИСУЕМ ГРАФИК
    st.subheader("Детализация по языкам")
    
    # Здесь тоже используем наши безопасные переменные
    chart_data = pd.DataFrame({
        "Точность (%)": [ru_acc, en_acc] 
    }, index=["Русский язык", "Английский язык"])
    
    st.bar_chart(chart_data)