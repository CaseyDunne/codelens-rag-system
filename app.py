####ИМПОРТ##################################################

import json
import time

import pandas as pd
import streamlit as st

import search_engine

####СТРАНИЦА##################################################

st.set_page_config(
    page_title="CodeLens RAG",
    page_icon="🔍",
    layout="wide"
)

custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');

    html, body, p, div, span, button, input, textarea, label, h1, h2, h3, h4, h5, h6 {
        font-family: 'Share Tech Mono', monospace !important;
    }

    .material-symbols-rounded,
    .stIcon,
    [data-testid="collapsedControl"],
    [data-testid="collapsedControl"] *,
    [data-testid="stIconMaterial"] {
        font-family: 'Material Symbols Rounded', sans-serif !important;
    }

    h1 {
        color: #D5FF40 !important;
        text-shadow: 0px 0px 10px rgba(213, 255, 64, 0.5);
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

######КЕШ#############################################################

@st.cache_resource
def init_search():
    return search_engine.init_db()


init_search()

####ФУНКЦИИ И ДАННЫЕ##################################################

def load_metrics():
    try:
        with open(search_engine.METRICS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

####БОКОВАЯ ПАНЕЛЬ##################################################

with st.sidebar:
    st.header("⚙️ Настройки поиска")
    top_k = st.slider("Количество результатов:", min_value=1, max_value=10, value=5)
    hybrid = st.checkbox("Гибридный поиск (вектор + TF-IDF)", value=True)
    alpha = st.slider("Вес векторного поиска (alpha)", 0.0, 1.0, 0.8, 0.05,
                      disabled=not hybrid,
                      help="1.0 — только смысловой поиск, 0.0 — только совпадение слов")

    st.divider()
    stats = search_engine.db_stats() if search_engine._collection is not None else None
    if stats and stats["chunks"]:
        st.caption(f"В базе: {stats['files']} файлов, {stats['chunks']} чанков")
    else:
        st.warning("База пуста. Запустите: `python index.py <папка>`")
    if st.button("🔄 Перезагрузить базу", help="После переиндексации без перезапуска приложения"):
        search_engine.reset()
        st.cache_resource.clear()
        st.rerun()

####ЦЕНТР##################################################

st.title("CodeLens: Умный поиск по коду")
tab_search, tab_metrics = st.tabs(["🔍 Поиск", "📊 Метрики"])

# TAB SEARCH
with tab_search:
    st.write("Опишите словами, что ищете, — на русском или английском.")
    with st.form("search_form"):
        user_query = st.text_input(
            label="Что будем искать?",
            placeholder="Например: как обрабатываются ошибки авторизации")
        submitted = st.form_submit_button("Найти")

    if submitted:
        if not user_query.strip():
            st.warning("Пожалуйста, введите запрос.", icon="⚠️")
        else:
            try:
                with st.spinner("Ищем..."):
                    t0 = time.perf_counter()
                    results = search_engine.search(
                        query=user_query, top_k=top_k, hybrid=hybrid, alpha=alpha)
                    elapsed = time.perf_counter() - t0

                if not results:
                    st.info("По вашему запросу ничего не найдено (или база пуста).", icon="🤷‍♀️")
                else:
                    mode = f"гибридный, alpha={alpha}" if hybrid else "векторный"
                    st.subheader(f"Топ-{len(results)} результатов")
                    st.caption(f"Время поиска: {elapsed:.2f} с · режим: {mode}")

                    for rank, r in enumerate(results, start=1):
                        lines = f"строки {r['start_line']}–{r['end_line']}" if r.get("start_line") else ""
                        title = (f"#{rank} · 📂 {r['path']} · 🧩 {r['type']}: {r['name']} "
                                 f"· 🎯 {r['percent']}%")
                        with st.expander(title, expanded=(rank == 1)):
                            st.caption(lines)
                            st.code(r["code"], language="python", line_numbers=True)
                            if hybrid:
                                st.caption(f"Смысловая близость: {r['vector_score']} · "
                                           f"совпадение слов (TF-IDF): {r['tfidf_score']}")
            except Exception as e:
                st.error(f"Произошла ошибка: {e}", icon="❌")

# TAB METRICS
with tab_metrics:
    st.header("📊 Качество поиска")
    st.caption(f"Тестовый набор: `{search_engine.EVAL_FILE}`. "
               "Режим (векторный / гибридный) берётся из настроек слева.")

    if st.button("▶️ Запустить тестирование и обновить метрики", use_container_width=True):
        with st.spinner("Прогоняем тестовые вопросы..."):
            try:
                search_engine.evaluate_precision(hybrid=hybrid, alpha=alpha)
                st.success("Метрики пересчитаны и сохранены!", icon="✅")
            except Exception as e:
                st.error(f"Не удалось посчитать метрики: {e}", icon="❌")

    m = load_metrics()
    if not m or "overall" not in m:
        st.info("Метрик пока нет — нажмите кнопку выше или запустите `python search_engine.py`.")
    else:
        o, k = m["overall"], m["k"]
        mode = "гибридный" + f" (alpha={m['alpha']})" if m["mode"] == "hybrid" else "векторный"
        st.caption(f"Последний прогон: {m['timestamp']} · режим: {mode} · "
                   f"база: {m['db_files']} файлов / {m['db_chunks']} чанков · вопросов: {o['n']}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"Precision@{k}", f"{o['precision_at_k']}%",
                  help="Доля правильных фрагментов среди выданных")
        c2.metric(f"Hit Rate@{k}", f"{o['hit_rate_at_k']}%",
                  help="Доля вопросов, где правильный фрагмент есть в топе")
        c3.metric("MRR", o["mrr"], help="Насколько высоко стоит первый правильный ответ (1 = всегда первый)")
        c4.metric("Время ответа (ср.)", f"{m['latency_s']['mean']} с",
                  help=f"p95: {m['latency_s']['p95']} с, максимум: {m['latency_s']['max']} с")

        if m.get("unresolved_references"):
            st.warning(f"{len(m['unresolved_references'])} эталонных фрагментов не найдено в базе — "
                       "возможно, проиндексирован не тот код: " +
                       ", ".join(m["unresolved_references"][:5]))

        st.divider()
        st.subheader("По языкам")
        lang_names = {"ru": "🇷🇺 Русские вопросы", "en": "🇬🇧 Английские вопросы"}
        for lang, v in m["by_language"].items():
            if not v:
                continue
            st.markdown(f"**{lang_names.get(lang, lang)}** · {v['n']} вопросов")
            l1, l2, l3 = st.columns(3)
            l1.metric(f"Hit Rate@{k}", f"{v['hit_rate_at_k']}%")
            l2.metric(f"Precision@{k}", f"{v['precision_at_k']}%")
            l3.metric("MRR", v["mrr"])

        # Диаграмма: столбцы рядом (stack=False), а не друг на друге
        chart_df = pd.DataFrame({
            lang_names.get(lang, lang).split(" ", 1)[1]: {
                f"Hit Rate@{k}, %": v["hit_rate_at_k"],
                f"Precision@{k}, %": v["precision_at_k"],
            }
            for lang, v in m["by_language"].items() if v
        }).T
        try:
            st.bar_chart(chart_df, stack=False, y_label="%", height=320)
        except TypeError:  # старый Streamlit без параметра stack
            st.dataframe(chart_df, use_container_width=True)

        st.subheader("По вопросам")
        rows = [{
            "Вопрос": r["question"],
            "Язык": r["language"],
            "Найден": "✅" if r["hit"] else "❌",
            "Место первого верного": r["first_rank"] or "—",
            "Время, с": r["latency_s"],
        } for r in m["per_question"]]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
