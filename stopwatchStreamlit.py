
# stopwatch_compatible.py
import time
from datetime import timedelta
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Cronómetro", page_icon="⏱️", layout="centered")

# -----------------------------
# Estado de la app
# -----------------------------
if "running" not in st.session_state:
    st.session_state.running = False
if "t0" not in st.session_state:
    st.session_state.t0 = time.monotonic()
if "elapsed_before" not in st.session_state:
    st.session_state.elapsed_before = 0.0  # acumulado antes del último start
if "laps" not in st.session_state:
    st.session_state.laps = []  # (nro, total_seg, lap_seg)

# -----------------------------
# Utilidades
# -----------------------------
def now_elapsed() -> float:
    base = st.session_state.elapsed_before
    if st.session_state.running:
        base += time.monotonic() - st.session_state.t0
    return base

def fmt(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    whole, ms = divmod(seconds, 1.0)
    return f"{str(td).split('.')[0]}.{int(ms*1000):03d}"

def start():
    if not st.session_state.running:
        st.session_state.t0 = time.monotonic()
        st.session_state.running = True

def pause():
    if st.session_state.running:
        st.session_state.elapsed_before += time.monotonic() - st.session_state.t0
        st.session_state.running = False

def reset():
    st.session_state.running = False
    st.session_state.t0 = time.monotonic()
    st.session_state.elapsed_before = 0.0
    st.session_state.laps = []

def lap():
    total = now_elapsed()
    last_total = st.session_state.laps[-1][1] if st.session_state.laps else 0.0
    this_lap = total - last_total
    st.session_state.laps.append((len(st.session_state.laps) + 1, total, this_lap))

# -----------------------------
# Sidebar (opciones)
# -----------------------------
st.sidebar.header("⚙️ Opciones")
refresh_ms = st.sidebar.slider("Frecuencia de actualización (ms)", 50, 1000, 100, step=50)
auto_start_opt = st.sidebar.checkbox("Iniciar automáticamente", value=False)

if auto_start_opt and not st.session_state.running and st.session_state.elapsed_before == 0:
    start()

# -----------------------------
# Título y display principal
# -----------------------------
st.title("⏱️ Cronómetro")
placeholder = st.empty()

# === Actualización sin st.autorefresh ===
if st.session_state.running:
    # pintar tiempo y forzar un rerun suave cada refresh_ms
    placeholder.markdown(
        f"<h1 style='text-align:center;font-size:4rem;'>{fmt(now_elapsed())}</h1>",
        unsafe_allow_html=True,
    )
    time.sleep(refresh_ms / 1000)
    st.rerun()
else:
    # pintar en estado pausado/ detenido
    placeholder.markdown(
        f"<h1 style='text-align:center;font-size:4rem;'>{fmt(now_elapsed())}</h1>",
        unsafe_allow_html=True,
    )

# -----------------------------
# Controles
# -----------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.session_state.running:
        if st.button("⏸️ Pausar", use_container_width=True):
            pause()
    else:
        if st.button("▶️ Iniciar", use_container_width=True):
            start()
with col2:
    if st.button("🔁 Reiniciar", use_container_width=True):
        reset()
with col3:
    if st.button("🏁 Vuelta", use_container_width=True, disabled=not (st.session_state.running or now_elapsed() > 0)):
        lap()
with col4:
    if st.session_state.laps:
        df = pd.DataFrame(
            [
                {
                    "Vuelta": n,
                    "Tiempo total (s)": round(total, 3),
                    "Tiempo de vuelta (s)": round(lap_s, 3),
                    "Tiempo total": fmt(total),
                    "Tiempo de vuelta": fmt(lap_s),
                }
                for (n, total, lap_s) in st.session_state.laps
            ]
        )
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("💾 Exportar CSV", data=csv, file_name="laps.csv", mime="text/csv", use_container_width=True)
    else:
        st.button("💾 Exportar CSV", disabled=True, use_container_width=True)

# -----------------------------
# Tabla de vueltas
# -----------------------------
if st.session_state.laps:
    st.subheader("Vueltas")
    df_show = pd.DataFrame(
        [
            {"Vuelta #": n, "Total": fmt(total), "Tiempo de vuelta": fmt(lap_s)}
            for (n, total, lap_s) in st.session_state.laps
        ]
    )
    st.dataframe(df_show, use_container_width=True, hide_index=True)
else:
    st.caption("No hay vueltas registradas todavía.")