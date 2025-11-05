import streamlit as st
import pandas as pd
import math

st.set_page_config(page_title="Distribución de Tiempos de Semáforos", page_icon="🚦", layout="wide")

st.title("🚦 Distribución de Tiempos de Luz Verde (Método Proporcional)")
st.caption("Soporta reparto proporcional por **volumen** o por **volumen × espaciamiento** y mínimo peatonal opcional.")

# -----------------------------
# Utilidades
# -----------------------------
def allocate_green_times(
    cycle_s: float,
    lost_per_phase_s: float,
    volumes,
    spacings=None,
    ped_start_s=None,
    ped_cross_s=None,
    yellow_s=None,
    round_to: float = 0.1
):
    volumes = [float(v) for v in volumes]
    n = len(volumes)
    spacings = [float(e) if e not in (None, "") else None for e in spacings] if spacings is not None else None

    total_lost = lost_per_phase_s * n
    green_available = cycle_s - total_lost
    if green_available <= 0:
        raise ValueError("El verde disponible es no positivo. Reduce pérdidas o incrementa el ciclo.")

    # Pesos
    if spacings is None or any(e is None for e in spacings):
        weights = volumes[:]
        mode = "volumes_only"
    else:
        weights = [v * e for v, e in zip(volumes, spacings)]
        mode = "volume_times_spacing"

    weight_sum = sum(weights)
    if weight_sum <= 0:
        raise ValueError("La suma de pesos debe ser positiva.")

    greens = [green_available * w / weight_sum for w in weights]

    # Mínimo peatonal
    ped_min = None
    if ped_start_s is not None and ped_cross_s is not None and yellow_s is not None:
        ped_min = max(0.0, ped_start_s + ped_cross_s - yellow_s)

    if ped_min is not None:
        locked = [max(g, ped_min) for g in greens]
        locked_sum = sum(locked)
        if locked_sum > green_available:
            # No factible: escalar todos al total disponible
            scale = green_available / (ped_min * n) if ped_min > 0 else 0.0
            greens = [ped_min * scale for _ in range(n)]
        else:
            remaining = green_available - sum(max(ped_min, g) for g in greens)
            above = [i for i, g in enumerate(greens) if g > ped_min]
            if above:
                total_above = sum(greens[i] - ped_min for i in above)
                new_greens = []
                for i, g in enumerate(greens):
                    if g > ped_min and total_above > 0:
                        extra = (g - ped_min) / total_above * remaining
                        new_greens.append(ped_min + extra)
                    else:
                        new_greens.append(ped_min)
                greens = new_greens
            else:
                greens = [ped_min for _ in greens]

    # Redondeo y rebalanceo
    if round_to and round_to > 0:
        greens = [round(g / round_to) * round_to for g in greens]
        diff = green_available - sum(greens)
        step = round_to if diff >= 0 else -round_to
        i = 0
        while abs(diff) >= round_to - 1e-9 and i < 10000:
            greens[i % n] += step
            diff -= step
            i += 1

    return mode, green_available, ped_min, greens

# -----------------------------
# Sidebar parámetros globales
# -----------------------------
with st.sidebar:
    st.header("⚙️ Parámetros")
    cycle_s = st.number_input("Ciclo total C (s)", min_value=10.0, max_value=300.0, value=90.0, step=1.0)
    lost_per_phase_s = st.number_input("Pérdidas por fase L (s)", min_value=0.0, max_value=20.0, value=5.0, step=0.5)
    round_to = st.number_input("Redondeo (s)", min_value=0.0, max_value=5.0, value=0.1, step=0.1)

    st.divider()
    use_spacings = st.checkbox("Usar espaciamiento por fase (s/veh)", value=True)

    st.divider()
    apply_ped = st.checkbox("Aplicar mínimo peatonal", value=True)
    ped_start = st.number_input("Inicio peatón (s)", min_value=0.0, max_value=15.0, value=5.0, step=0.5, disabled=not apply_ped)
    ped_cross = st.number_input("Cruce peatón (s)", min_value=0.0, max_value=60.0, value=14.0, step=1.0, disabled=not apply_ped)
    yellow = st.number_input("Amarillo (s)", min_value=0.0, max_value=10.0, value=3.0, step=0.5, disabled=not apply_ped)

# -----------------------------
# Tabla editable de fases
# -----------------------------
st.subheader("Fases y Demanda")
st.caption("Edita nombres, volúmenes (veh/h) y opcionalmente espaciamientos (s/veh). Deja vacío el espaciamiento para reparto solo por volumen.")

default_data = pd.DataFrame({
    "Fase": ["A", "B", "C"],
    #"Volumen (veh/h)": [500, 350, 250],
    "Volumen (veh/h)": [100, 250, 450],
    "Espaciamiento (s/veh)": [3.0, 4.0, 6.0] if use_spacings else [None, None, None]
})

edited = st.data_editor(
    default_data,
    num_rows="dynamic",
    use_container_width=True
)

# -----------------------------
# Cálculo
# -----------------------------
if st.button("Calcular distribución", type="primary"):
    try:
        names = edited["Fase"].astype(str).tolist()
        volumes = edited["Volumen (veh/h)"].astype(float).tolist()
        spacings = edited["Espaciamiento (s/veh)"].tolist() if use_spacings else None

        mode, green_available, ped_min, greens = allocate_green_times(
            cycle_s=cycle_s,
            lost_per_phase_s=lost_per_phase_s,
            volumes=volumes,
            spacings=spacings,
            ped_start_s=ped_start if apply_ped else None,
            ped_cross_s=ped_cross if apply_ped else None,
            yellow_s=yellow if apply_ped else None,
            round_to=round_to
        )

        result_df = pd.DataFrame({
            "Fase": names,
            "Volumen (veh/h)": volumes,
            "Espaciamiento (s/veh)": spacings if spacings is not None else [None]*len(volumes),
            "Verde asignado (s)": greens
        })

        col1, col2 = st.columns([1.2, 1])
        with col1:
            st.markdown("### Resultados")
            st.write(f"**Verde disponible total:** {green_available:.1f} s")
            if ped_min is not None:
                st.write(f"**Mínimo peatonal aplicado:** {ped_min:.1f} s")
            st.write(f"**Modo de reparto:** {'Volumen × Espaciamiento' if mode=='volume_times_spacing' else 'Solo Volumen'}")
            st.dataframe(result_df, use_container_width=True, height=260)

        with col2:
            st.markdown("### Gráfico de verdes (s)")
            chart_df = result_df.set_index("Fase")[["Verde asignado (s)"]]
            st.bar_chart(chart_df)

        st.download_button(
            "Descargar resultados (CSV)",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name="verde_asignado.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Error en el cálculo: {e}")

with st.expander("🔎 Notas metodológicas"):
    st.markdown("- Si hay **espaciamientos** en todas las fases, el reparto usa **Volumen × Espaciamiento**." "- Si hay **algún** espaciamiento vacío, se toma el reparto **solo por Volumen**." "- El **mínimo peatonal** se calcula como `inicio + cruce - amarillo` y se fuerza por fase."
        "- Se redondea y se **rebalancea** para que la suma de verdes coincida con el disponible."
    )
