
# import time
# from datetime import timedelta
# import streamlit as st

# # -----------------------------
# # Configuración de la página
# # -----------------------------
# st.set_page_config(page_title="Simulador de Semáforos", page_icon="🚦", layout="wide")

# # -----------------------------
# # Estado de la app
# # -----------------------------
# if "running" not in st.session_state:
#     st.session_state.running = False
# if "t0" not in st.session_state:
#     st.session_state.t0 = time.monotonic()
# if "offsets" not in st.session_state:
#     st.session_state.offsets = []

# # -----------------------------
# # Sidebar: parámetros
# # -----------------------------
# st.sidebar.header("⚙️ Parámetros de simulación")
# n_lights = st.sidebar.slider("Número de semáforos", 1, 6, 4)

# # Tiempos de fase (segundos)
# with st.sidebar.expander("⏱️ Duraciones de fase (por ciclo)", expanded=True):
#     t_green = st.number_input("Verde (s)", min_value=1, max_value=300, value=15, step=1)
#     t_yellow = st.number_input("Amarillo (s)", min_value=1, max_value=60, value=3, step=1)
#     t_red = st.number_input("Rojo (s)", min_value=1, max_value=300, value=15, step=1)
# cycle = t_green + t_yellow + t_red

# # Offset por semáforo para simular coordinación (en segundos)
# with st.sidebar.expander("🧭 Offsets por semáforo (coordinación)", expanded=False):
#     new_offsets = []
#     for i in range(n_lights):
#         # Distribución por defecto: escalonado a lo largo del ciclo
#         default_offset = int((i * cycle) / max(n_lights, 1)) % cycle
#         val = st.number_input(f"Offset S{i+1} (s)", min_value=0, max_value=9999, value=default_offset, step=1, key=f"off_{i}")
#         new_offsets.append(val)
#     st.session_state.offsets = new_offsets

# tick_ms = st.sidebar.slider("Frecuencia de actualización (ms)", 100, 2000, 300, step=50)
# st.sidebar.caption("Sugerencia: 200–500 ms da una animación fluida sin consumir demasiados recursos.")

# # Controles
# c1, c2, c3 = st.sidebar.columns(3)
# if c1.button("▶️ Iniciar"):
#     st.session_state.running = True
#     st.session_state.t0 = time.monotonic()
# if c2.button("⏸️ Pausar"):
#     st.session_state.running = False
#     # Guardar elapsed al momento de pausar
#     st.session_state.elapsed_paused = time.monotonic() - st.session_state.t0
# if c3.button("🔁 Reiniciar"):
#     st.session_state.running = False
#     st.session_state.t0 = time.monotonic()
#     st.session_state.elapsed_paused = 0.0

# # -----------------------------
# # Funciones auxiliares
# # -----------------------------
# def phase_at(t):
#     """
#     Dado t en [0, cycle), devuelve (nombre_fase, tiempo_restante_en_fase).
#     Fases en orden: VERDE -> AMARILLO -> ROJO
#     """
#     if t < t_green:
#         return "VERDE", t_green - t
#     elif t < t_green + t_yellow:
#         return "AMARILLO", (t_green + t_yellow) - t
#     else:
#         return "ROJO", cycle - t

# def color_css(phase):
#     if phase == "VERDE":
#         return "#22c55e"  # verde
#     if phase == "AMARILLO":
#         return "#f59e0b"  # ámbar
#     return "#ef4444"      # rojo

# def render_light(phase, remaining, idx):
#     # Indicador circular con HTML/CSS
#     color = color_css(phase)
#     # Barra de progreso de la fase actual (cuánto ya pasó)
#     if phase == "VERDE":
#         t_phase = t_green
#         elapsed = t_phase - remaining
#     elif phase == "AMARILLO":
#         t_phase = t_yellow
#         elapsed = t_phase - remaining
#     else:
#         t_phase = t_red
#         elapsed = t_phase - remaining

#     st.markdown(f"### S{idx+1}")
#     st.markdown(
#         f'''
#         <div style="display:flex;align-items:center;gap:16px;">
#           <div style="width:64px;height:64px;border-radius:50%;background:{color};
#                       box-shadow: 0 0 12px rgba(0,0,0,0.25), inset 0 0 12px rgba(255,255,255,0.25);
#                       border: 4px solid rgba(0,0,0,0.15);"></div>
#           <div>
#             <div style="font-size:1.25rem;font-weight:600;">{phase}</div>
#             <div style="opacity:0.8;">Restante en fase: <strong>{int(remaining)} s</strong></div>
#           </div>
#         </div>
#         ''',
#         unsafe_allow_html=True
#     )
#     st.progress(min(max(elapsed / max(t_phase, 0.0001), 0.0), 1.0))

# # -----------------------------
# # Cálculo del tiempo actual
# # -----------------------------
# now = time.monotonic()
# if st.session_state.running:
#     elapsed = now - st.session_state.t0
# else:
#     # Si está pausado, "congelamos" elapsed al momento de pausar
#     elapsed = st.session_state.get("elapsed_paused", 0.0)

# # -----------------------------
# # Encabezado
# # -----------------------------
# st.title("🚦 Simulador de Semáforos (3 fases)")
# st.caption("Fases: **VERDE → AMARILLO → ROJO**. Usa offsets para coordinar semáforos.")

# # Información de ciclo
# met1, met2, met3, met4 = st.columns(4)
# met1.metric("Ciclo total", f"{cycle} s")
# met2.metric("Verde", f"{t_green} s")
# met3.metric("Amarillo", f"{t_yellow} s")
# met4.metric("Rojo", f"{t_red} s")

# # -----------------------------
# # Render de semáforos
# # -----------------------------
# cols = st.columns(n_lights)
# for i in range(n_lights):
#     with cols[i]:
#         off = st.session_state.offsets[i] if i < len(st.session_state.offsets) else 0
#         t_local = (elapsed + off) % cycle
#         phase, remaining = phase_at(t_local)
#         render_light(phase, remaining, i)

# # -----------------------------
# # Próximos cambios (tabla simple)
# # -----------------------------
# st.subheader("⏭️ Próximos cambios de fase (próximo evento)")
# rows = []
# for i in range(n_lights):
#     off = st.session_state.offsets[i] if i < len(st.session_state.offsets) else 0
#     t_local = (elapsed + off) % cycle
#     phase, remaining = phase_at(t_local)
#     rows.append((f"S{i+1}", phase, int(remaining)))

# # Ordenar por el que cambia antes
# rows.sort(key=lambda x: x[2])
# st.table({"Semáforo": [r[0] for r in rows],
#           "Fase actual": [r[1] for r in rows],
#           "Cambio en (s)": [r[2] for r in rows]})

# # -----------------------------
# # Auto-actualización
# # -----------------------------
# if st.session_state.running:
#     st.query_params["r"] = str(int(now * 1000) % 1000000)
#   # evita cachear estados de URL
#     st.markdown(f"<script>setTimeout(() => window.location.reload(), {tick_ms});</script>", unsafe_allow_html=True)

import time
import streamlit as st

# -----------------------------
# Configuración de la página
# -----------------------------
st.set_page_config(page_title="Simulador de Semáforos", page_icon="🚦", layout="wide")

# -----------------------------
# Estado inicial
# -----------------------------
if "running" not in st.session_state:
    st.session_state.running = True              # 🔹 ahora inicia automáticamente
if "t0" not in st.session_state:
    st.session_state.t0 = time.monotonic()
if "offsets" not in st.session_state:
    st.session_state.offsets = []

# -----------------------------
# Sidebar: parámetros
# -----------------------------
st.sidebar.header("⚙️ Parámetros de simulación")
n_lights = st.sidebar.slider("Número de semáforos", 1, 6, 4)

# Tiempos de fase (segundos)
with st.sidebar.expander("⏱️ Duraciones de fase (por ciclo)", expanded=True):
    t_green = st.number_input("Verde (s)", min_value=1, max_value=300, value=15, step=1)
    t_yellow = st.number_input("Amarillo (s)", min_value=1, max_value=60, value=3, step=1)
    t_red = st.number_input("Rojo (s)", min_value=1, max_value=300, value=15, step=1)
cycle = t_green + t_yellow + t_red

# Offset por semáforo
with st.sidebar.expander("🧭 Offsets por semáforo (coordinación)", expanded=False):
    new_offsets = []
    for i in range(n_lights):
        default_offset = int((i * cycle) / max(n_lights, 1)) % cycle
        val = st.number_input(f"Offset S{i+1} (s)", min_value=0, max_value=9999,
                              value=default_offset, step=1, key=f"off_{i}")
        new_offsets.append(val)
    st.session_state.offsets = new_offsets

tick_ms = st.sidebar.slider("Frecuencia de actualización (ms)", 100, 2000, 300, step=50)

# Controles manuales (por si deseas pausar o reiniciar)
c1, c2 = st.sidebar.columns(2)
if c1.button("⏸️ Pausar"):
    st.session_state.running = False
    st.session_state.elapsed_paused = time.monotonic() - st.session_state.t0
if c2.button("🔁 Reiniciar"):
    st.session_state.running = True
    st.session_state.t0 = time.monotonic()
    st.session_state.elapsed_paused = 0.0

# -----------------------------
# Funciones auxiliares
# -----------------------------
def phase_at(t):
    if t < t_green:
        return "VERDE", t_green - t
    elif t < t_green + t_yellow:
        return "AMARILLO", (t_green + t_yellow) - t
    else:
        return "ROJO", cycle - t

def color_css(phase):
    return {"VERDE": "#22c55e", "AMARILLO": "#f59e0b", "ROJO": "#ef4444"}[phase]

def render_light(phase, remaining, idx):
    color = color_css(phase)
    if phase == "VERDE":
        t_phase = t_green
    elif phase == "AMARILLO":
        t_phase = t_yellow
    else:
        t_phase = t_red
    elapsed = t_phase - remaining

    st.markdown(f"### S{idx+1}")
    st.markdown(
        f'''
        <div style="display:flex;align-items:center;gap:16px;">
          <div style="width:64px;height:64px;border-radius:50%;background:{color};
                      box-shadow: 0 0 12px rgba(0,0,0,0.25), inset 0 0 12px rgba(255,255,255,0.25);
                      border: 4px solid rgba(0,0,0,0.15);"></div>
          <div>
            <div style="font-size:1.25rem;font-weight:600;">{phase}</div>
            <div style="opacity:0.8;">Restante en fase: <strong>{int(remaining)} s</strong></div>
          </div>
        </div>
        ''',
        unsafe_allow_html=True
    )
    st.progress(min(max(elapsed / max(t_phase, 0.001), 0.0), 1.0))

# -----------------------------
# Cálculo del tiempo actual
# -----------------------------
now = time.monotonic()
if st.session_state.running:
    elapsed = now - st.session_state.t0
else:
    elapsed = st.session_state.get("elapsed_paused", 0.0)

# -----------------------------
# Encabezado
# -----------------------------
st.title("🚦 Simulador de Semáforos (3 fases)")
st.caption("Fases: **VERDE → AMARILLO → ROJO**. El conteo inicia automáticamente.")

met1, met2, met3, met4 = st.columns(4)
met1.metric("Ciclo total", f"{cycle} s")
met2.metric("Verde", f"{t_green} s")
met3.metric("Amarillo", f"{t_yellow} s")
met4.metric("Rojo", f"{t_red} s")

# -----------------------------
# Render de semáforos
# -----------------------------
cols = st.columns(n_lights)
for i in range(n_lights):
    with cols[i]:
        off = st.session_state.offsets[i] if i < len(st.session_state.offsets) else 0
        t_local = (elapsed + off) % cycle
        phase, remaining = phase_at(t_local)
        render_light(phase, remaining, i)

# -----------------------------
# Próximos cambios
# -----------------------------
st.subheader("⏭️ Próximos cambios de fase (próximo evento)")
rows = []
for i in range(n_lights):
    off = st.session_state.offsets[i] if i < len(st.session_state.offsets) else 0
    t_local = (elapsed + off) % cycle
    phase, remaining = phase_at(t_local)
    rows.append((f"S{i+1}", phase, int(remaining)))

rows.sort(key=lambda x: x[2])
st.table({"Semáforo": [r[0] for r in rows],
          "Fase actual": [r[1] for r in rows],
          "Cambio en (s)": [r[2] for r in rows]})

# -----------------------------
# Auto-actualización automática
# -----------------------------
if st.session_state.running:
    st.query_params["r"] = str(int(now * 1000) % 1000000)
    st.markdown(f"<script>setTimeout(() => window.location.reload(), {tick_ms});</script>", unsafe_allow_html=True)
