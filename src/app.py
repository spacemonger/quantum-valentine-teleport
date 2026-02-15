import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pennylane as qml
import plotly.graph_objects as go

# ----------------------------------------
# Quantum helpers
# ----------------------------------------

# Analytic statevector device (no shots)
dev = qml.device("default.qubit", wires=3)

WIRE_LABELS = ["S", "A", "B"]  # wire 0,1,2


def prep_psi_single(theta, phi):
    """|ψ⟩ = cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩"""
    return np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)], dtype=complex)


def reduced_density_matrix_from_state(state, keep, n=3):
    """Reduced density matrix of one qubit from full pure statevector."""
    psi = np.asarray(state, dtype=complex).reshape([2] * n)  # psi[iS, iA, iB]
    trace_axes = [i for i in range(n) if i != keep]
    rho = np.tensordot(psi, np.conj(psi), axes=(trace_axes, trace_axes))
    return rho  # (2,2)


def bloch_vector_from_rho(rho):
    """Bloch vector v: v_i = Tr(ρ σ_i)."""
    rho = np.asarray(rho, dtype=complex)
    x = 2 * np.real(rho[0, 1])
    y = -2 * np.imag(rho[0, 1])  # sign convention
    z = np.real(rho[0, 0] - rho[1, 1])
    return np.array([float(x), float(y), float(z)])


def get_qubit_probabilities(full_state, qubit_idx, n=3):
    rho = reduced_density_matrix_from_state(full_state, qubit_idx, n)
    p0 = float(np.real(rho[0, 0]))
    p1 = float(np.real(rho[1, 1]))
    s = p0 + p1
    if s > 0:
        p0 /= s
        p1 /= s
    return p0, p1


def principal_pure_state_from_rho(rho):
    """
    For display: if rho is mixed, pick eigenvector with largest eigenvalue.
    (Bloch + probabilities should come from rho, not this.)
    """
    vals, vecs = np.linalg.eigh(np.asarray(rho, dtype=complex))
    psi = vecs[:, int(np.argmax(vals))]
    if abs(psi[0]) > 1e-12:
        psi = psi * np.exp(-1j * np.angle(psi[0]))
    return psi / np.linalg.norm(psi)


def state_fidelity_density(rho, psi):
    """F(ρ, |ψ⟩) = ⟨ψ| ρ |ψ⟩ for pure psi."""
    psi = np.asarray(psi, dtype=complex)
    rho = np.asarray(rho, dtype=complex)
    return float(np.real(np.vdot(psi, rho @ psi)))


# ---------- Plotly Bloch sphere (Qiskit-like) ----------

def _sphere_mesh(res=55):
    u = np.linspace(0, 2 * np.pi, res)
    v = np.linspace(0, np.pi, res)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def _circle_lat(theta, res=200):
    """Latitude circle at polar angle theta (0=north pole)."""
    t = np.linspace(0, 2 * np.pi, res)
    x = np.cos(t) * np.sin(theta)
    y = np.sin(t) * np.sin(theta)
    z = np.full_like(t, np.cos(theta))
    return x, y, z


def _circle_lon(phi, res=200):
    """Longitude great circle at azimuth phi."""
    t = np.linspace(0, 2 * np.pi, res)
    x = np.cos(phi) * np.sin(t)
    y = np.sin(phi) * np.sin(t)
    z = np.cos(t)
    return x, y, z


def plot_bloch_vector_plotly(vec, title="", res=45, show_grid=True):
    """
    Plotly Bloch sphere + vector with Qiskit-ish styling:
    - transparent sphere
    - latitude/longitude grid
    - ±X, ±Y, ±Z labels + |0>, |1> at poles
    - vector arrow (line + cone)
    """
    vec = np.asarray(vec, dtype=float)
    n = np.linalg.norm(vec)
    if n > 1.0 + 1e-9:
        vec = vec / n

    fig = go.Figure()

    # Sphere surface
    xs, ys, zs = _sphere_mesh(res=res)
    fig.add_trace(
        go.Surface(
            x=xs, y=ys, z=zs,
            opacity=0.12,
            showscale=False,
            hoverinfo="skip"
        )
    )

    # Grid lines (lat/lon)
    if show_grid:
        # Latitudes (skip poles)
        for th in [np.pi / 6, np.pi / 3, np.pi / 2, 2 * np.pi / 3, 5 * np.pi / 6]:
            xg, yg, zg = _circle_lat(th)
            fig.add_trace(
                go.Scatter3d(
                    x=xg, y=yg, z=zg,
                    mode="lines",
                    line=dict(width=2, color="rgba(120,120,120,0.25)"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        # Longitudes
        for ph in [0, np.pi / 6, np.pi / 3, np.pi / 2, 2 * np.pi / 3, 5 * np.pi / 6]:
            xg, yg, zg = _circle_lon(ph)
            fig.add_trace(
                go.Scatter3d(
                    x=xg, y=yg, z=zg,
                    mode="lines",
                    line=dict(width=2, color="rgba(120,120,120,0.25)"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    # Axes
    axis_len = 1.12
    axes = [
        ([-axis_len, axis_len], [0, 0], [0, 0]),  # X
        ([0, 0], [-axis_len, axis_len], [0, 0]),  # Y
        ([0, 0], [0, 0], [-axis_len, axis_len]),  # Z
    ]
    for xs_, ys_, zs_ in axes:
        fig.add_trace(
            go.Scatter3d(
                x=xs_, y=ys_, z=zs_,
                mode="lines",
                line=dict(width=6, color="rgba(90,90,90,0.65)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Axis labels and pole labels
    lp = 1.22
    fig.add_trace(
        go.Scatter3d(
            x=[lp, -lp, 0, 0, 0, 0],
            y=[0, 0, lp, -lp, 0, 0],
            z=[0, 0, 0, 0, lp, -lp],
            mode="text",
            text=["+X", "-X", "+Y", "-Y", "+Z", "-Z"],
            textposition="middle center",
            hoverinfo="skip",
            showlegend=False,
            textfont=dict(size=12),
        )
    )

    # |0>, |1> at poles
    fig.add_trace(
        go.Scatter3d(
            x=[0, 0],
            y=[0, 0],
            z=[1.28, -1.28],
            mode="text",
            text=["|0⟩", "|1⟩"],
            textposition="middle center",
            hoverinfo="skip",
            showlegend=False,
            textfont=dict(size=13),
        )
    )

    # Vector line + marker
    fig.add_trace(
        go.Scatter3d(
            x=[0, vec[0]],
            y=[0, vec[1]],
            z=[0, vec[2]],
            mode="lines+markers",
            line=dict(width=10, color="rgba(220, 50, 50, 0.95)"),
            marker=dict(size=4, color="rgba(220, 50, 50, 0.95)"),
            hovertemplate="Bloch v = (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>",
            showlegend=False,
        )
    )

    # Arrowhead cone at the tip
    if np.linalg.norm(vec) > 1e-6:
        fig.add_trace(
            go.Cone(
                x=[vec[0]], y=[vec[1]], z=[vec[2]],
                u=[vec[0]], v=[vec[1]], w=[vec[2]],
                anchor="tip",
                sizemode="absolute",
                sizeref=0.18,
                colorscale=[[0, "rgba(220, 50, 50, 0.95)"], [1, "rgba(220, 50, 50, 0.95)"]],
                showscale=False,
                hoverinfo="skip",
            )
        )

    # Fixed camera to be consistent across S/A/B
    camera = dict(
        eye=dict(x=1.4, y=1.2, z=1.0),
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
    )

    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, t=28 if title else 0, b=0),
        height=360,
        scene=dict(
            xaxis=dict(range=[-1.25, 1.25], showgrid=False, zeroline=False, showticklabels=False, title=""),
            yaxis=dict(range=[-1.25, 1.25], showgrid=False, zeroline=False, showticklabels=False, title=""),
            zaxis=dict(range=[-1.25, 1.25], showgrid=False, zeroline=False, showticklabels=False, title=""),
            aspectmode="cube",
            camera=camera,
        ),
        showlegend=False,
    )
    return fig


def format_complex(z, threshold=1e-10):
    """Short complex formatting for display."""
    z = complex(z)
    if abs(z) < threshold:
        return "0"

    r = np.real(z)
    im = np.imag(z)

    r_str = "" if abs(r) < threshold else f"{r:.3f}"
    im_str = ""
    if abs(im) >= threshold:
        if im >= 0 and r_str:
            im_str = f"+{im:.3f}i"
        else:
            im_str = f"{im:.3f}i"

    out = (r_str + im_str).strip()
    return out if out else "0"


def state_to_dirac_terms(state_vector, wire_labels=WIRE_LABELS, threshold=1e-10):
    """List of (amp_str, basis_bits, prob) for |state⟩."""
    state_vector = np.asarray(state_vector, dtype=complex)
    n_qubits = int(np.log2(len(state_vector)))
    terms = []
    for i, amp in enumerate(state_vector):
        p = float(np.abs(amp) ** 2)
        if p > threshold:
            bits = format(i, f"0{n_qubits}b")
            terms.append((format_complex(amp), bits, p))
    return terms


def plot_measurement_histogram(counts_dict):
    """Histogram for counts on wires [S,A] => bitstring 'SA'."""
    fig, ax = plt.subplots(figsize=(6, 3))
    labels = ["00", "01", "10", "11"]  # S A
    values = [counts_dict.get(lbl, 0) for lbl in labels]
    ax.bar(labels, values)
    ax.set_xlabel("Measurement outcome (S A) = (m0 m1)", fontsize=11)
    ax.set_ylabel("Counts", fontsize=11)
    ax.set_title("Alice's Measurement Results")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def normalize_counts_keys(counts):
    """
    PennyLane counts keys are usually strings like '01', but make it robust.
    Returns dict with '00','01','10','11' keys.
    """
    out = {}
    for k, v in counts.items():
        if isinstance(k, str):
            bits = k
        else:
            bits = "".join(str(int(b)) for b in k)
        if len(bits) == 2:
            out[bits] = int(v)
    return out


def collapse_on_SA(state_after_bell, m0, m1):
    """
    Collapse the 3-qubit state |ψ⟩ (S,A,B) onto measured S=m0, A=m1.
    Returns collapsed full 8-vector and Bob's normalized 2-vector.
    """
    psi = np.asarray(state_after_bell, dtype=complex).reshape(2, 2, 2)  # (S,A,B)
    psi_B = psi[m0, m1, :].copy()
    nrm = np.linalg.norm(psi_B)
    if nrm < 1e-12:
        psi_B = np.array([1.0, 0.0], dtype=complex)
    else:
        psi_B /= nrm

    full = np.zeros(8, dtype=complex)
    base = (m0 << 2) + (m1 << 1)  # B=0 at base, B=1 at base+1
    full[base + 0] = psi_B[0]
    full[base + 1] = psi_B[1]
    return full, psi_B


def apply_bob_corrections(psi_B, m0, m1):
    """Teleportation corrections: X if m1=1 (A), Z if m0=1 (S)."""
    out = np.array(psi_B, dtype=complex)
    if m1 == 1:  # X
        out = np.array([out[1], out[0]], dtype=complex)
    if m0 == 1:  # Z
        out = np.array([out[0], -out[1]], dtype=complex)
    return out


def compose_product_state_SA_B(m0, m1, psi_B):
    """Build |m0 m1⟩_{SA} ⊗ |psi_B⟩ as 8-vector."""
    full = np.zeros(8, dtype=complex)
    base = (m0 << 2) + (m1 << 1)
    full[base + 0] = psi_B[0]
    full[base + 1] = psi_B[1]
    return full


# ----------------------------------------
# Circuit builders
# ----------------------------------------

@qml.qnode(dev)
def state_after_steps(actions, theta, phi):
    """Unitary part only (encode/entangle/bell_basis). No decode here."""
    if actions.get("encode", False):
        qml.RY(theta, wires=0)
        qml.RZ(phi, wires=0)

    if actions.get("entangle", False):
        qml.Hadamard(wires=1)
        qml.CNOT(wires=[1, 2])

    if actions.get("bell_basis", False):
        qml.CNOT(wires=[0, 1])
        qml.Hadamard(wires=0)

    return qml.state()


def run_measurement_counts(actions, theta, phi, shots):
    """Sample measurement outcomes on wires [S,A] after bell-basis setup."""
    dev_shots = qml.device("default.qubit", wires=3, shots=shots)

    @qml.qnode(dev_shots)
    def meas():
        if actions.get("encode", False):
            qml.RY(theta, wires=0)
            qml.RZ(phi, wires=0)

        if actions.get("entangle", False):
            qml.Hadamard(wires=1)
            qml.CNOT(wires=[1, 2])

        qml.CNOT(wires=[0, 1])
        qml.Hadamard(wires=0)

        return qml.counts(wires=[0, 1])  # bitstrings in order (S then A)

    raw_counts = meas()
    return normalize_counts_keys(raw_counts)


# ----------------------------------------
# Streamlit UI
# ----------------------------------------

st.set_page_config(page_title="Quantum Valentine's Day", layout="wide")
st.title("💝 Interactive Quantum Valentine Teleportation")

# Session state init
if "actions" not in st.session_state:
    st.session_state.actions = {}
if "measured" not in st.session_state:
    st.session_state.measured = False
if "m0" not in st.session_state:
    st.session_state.m0 = 0  # S
if "m1" not in st.session_state:
    st.session_state.m1 = 0  # A
if "counts" not in st.session_state:
    st.session_state.counts = {}
if "theta_phi_cache" not in st.session_state:
    st.session_state.theta_phi_cache = (None, None)
if "actions_cache" not in st.session_state:
    st.session_state.actions_cache = None

# Header area
col_img_left, col_img_right = st.columns([0.7, 1.3])

with col_img_left:
    try:
        from PIL import Image
        img = Image.open("alice_bob.png")
        st.image(img, use_container_width=True, caption="Alice and Bob in quantum uncertainty")
    except Exception:
        st.info("💝 Alice & Bob\n\n(place alice_bob.png here)")

with col_img_right:
    st.markdown("""
    Alice wants to send Bob a Valentine's Day message using quantum teleportation! 
    She'll prepare a quantum state where |0⟩ means **"Happy Valentine's Day!"** 💝 and |1⟩ means **"Better luck next time"** 💔.
    Adjust θ and φ to control the probability of each message, and try the protocol yourself! Click the action buttons below in any order and see what happens.

    **The correct protocol:**
    1. 🔗 **Entangle** A-B (create Bell pair)
    2. 📝 **Encode** Alice's message |ψ⟩ on S (using θ and φ)
        - Higher P(|0⟩) → More likely "Happy Valentine's Day!" 💝
        - Higher P(|1⟩) → More likely "Better luck next time" 💔
    3. 🔀 **Bell Basis** measurement setup: CNOT(S→A) then H(S)
    4. 📏 **Measure** Alice's qubits S and A → classical bits (m0, m1)
    5. 🔓 **Decode** Bob applies corrections: X if m1=1, Z if m0=1 → receives message!

    **Try breaking it!** What happens if you skip entanglement? Or measure before encoding?
    """)

st.divider()

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Message Parameters")

    st.subheader("Alice's Message")
    st.latex(r"|\psi\rangle = \cos(\theta/2)|0\rangle + e^{i\phi}\sin(\theta/2)|1\rangle")

    theta = st.slider("θ (angle)", 0.0, float(np.pi), float(np.pi / 2), 0.01)
    phi = st.slider("φ (phase)", 0.0, float(2 * np.pi), 0.0, 0.01)

    p0 = float(np.cos(theta / 2) ** 2)
    st.write(f"💝 P(Happy) = {p0:.1%}")
    st.write(f"💔 P(Unlucky) = {1 - p0:.1%}")

    st.divider()
    st.subheader("Measurement shots")
    shots = st.slider("Number of shots", 100, 5000, 1000, 100)

    st.divider()
    if st.button("🔄 Reset All", use_container_width=True):
        st.session_state.actions = {}
        st.session_state.measured = False
        st.session_state.m0 = 0
        st.session_state.m1 = 0
        st.session_state.counts = {}
        st.session_state.theta_phi_cache = (None, None)
        st.session_state.actions_cache = None
        st.rerun()

# Invalidate measurement if unitary actions or theta/phi change AFTER measuring (decode is excluded)
current_actions_fingerprint = tuple(
    sorted((k for k, v in st.session_state.actions.items() if v is True and k != "decode"))
)
if st.session_state.measured:
    if (theta, phi) != st.session_state.theta_phi_cache or current_actions_fingerprint != st.session_state.actions_cache:
        st.session_state.measured = False
        st.session_state.counts = {}
        st.session_state.m0 = 0
        st.session_state.m1 = 0

st.session_state.theta_phi_cache = (theta, phi)
st.session_state.actions_cache = current_actions_fingerprint

# Control panel
st.subheader("🎮 Control Panel")

c1, c2, c3, c4, c5 = st.columns(5)


def toggle_action(key):
    st.session_state.actions[key] = not st.session_state.actions.get(key, False)


with c1:
    if st.button(
        "🔗 Entangle\nA-B",
        use_container_width=True,
        type="primary" if st.session_state.actions.get("entangle", False) else "secondary",
    ):
        toggle_action("entangle")
        st.rerun()
    st.caption("✅ Done" if st.session_state.actions.get("entangle", False) else "⬜ Pending")

with c2:
    if st.button(
        "📝 Encode\nMessage",
        use_container_width=True,
        type="primary" if st.session_state.actions.get("encode", False) else "secondary",
    ):
        toggle_action("encode")
        st.rerun()
    st.caption("✅ Done" if st.session_state.actions.get("encode", False) else "⬜ Pending")

with c3:
    if st.button(
        "🔀 Bell Basis\nSetup",
        use_container_width=True,
        type="primary" if st.session_state.actions.get("bell_basis", False) else "secondary",
    ):
        toggle_action("bell_basis")
        st.rerun()
    st.caption("✅ Done" if st.session_state.actions.get("bell_basis", False) else "⬜ Pending")

with c4:
    measure_disabled = not st.session_state.actions.get("bell_basis", False)
    if st.button(
        "📏 Measure\nAlice",
        use_container_width=True,
        disabled=measure_disabled,
        type="primary" if st.session_state.measured else "secondary",
    ):
        if not st.session_state.measured:
            counts = run_measurement_counts(st.session_state.actions, theta, phi, shots)

            outcomes = list(counts.keys())
            probs = np.array([counts[o] for o in outcomes], dtype=float)
            probs /= probs.sum()

            sampled = np.random.choice(outcomes, p=probs)  # "SA"

            st.session_state.m0 = int(sampled[0])  # S
            st.session_state.m1 = int(sampled[1])  # A
            st.session_state.counts = counts
            st.session_state.measured = True
        else:
            st.session_state.measured = False
            st.session_state.m0 = 0
            st.session_state.m1 = 0
            st.session_state.counts = {}
        st.rerun()

    if st.session_state.measured:
        st.caption("✅ Done")
    else:
        st.caption("⬜ Need Bell Basis" if measure_disabled else "⬜ Pending")

with c5:
    decode_disabled = not st.session_state.measured
    if st.button(
        "🔓 Decode\nBob",
        use_container_width=True,
        disabled=decode_disabled,
        type="primary" if st.session_state.actions.get("decode", False) else "secondary",
    ):
        toggle_action("decode")
        st.rerun()
    st.caption(
        "✅ Done"
        if st.session_state.actions.get("decode", False)
        else ("⬜ Need Measure" if decode_disabled else "⬜ Pending")
    )

# Protocol check (selection-based)
st.markdown("---")
need = ["entangle", "encode", "bell_basis"]
done = [k for k in need if st.session_state.actions.get(k, False)]

if len(done) == 3 and st.session_state.measured and st.session_state.actions.get("decode", False):
    st.success("✅ Perfect! You completed all protocol steps.")
elif len(done) > 0:
    st.info(f"🔄 Steps enabled: {', '.join(done)}")
else:
    st.info("👆 Toggle steps above to build the protocol")

st.divider()

# ----------------------------------------
# Compute state consistently (collapse + decode)
# ----------------------------------------

state_unitary = state_after_steps(st.session_state.actions, theta, phi)
state_display = state_unitary

if st.session_state.measured and st.session_state.actions.get("bell_basis", False):
    m0 = int(st.session_state.m0)
    m1 = int(st.session_state.m1)

    collapsed_full, psi_B = collapse_on_SA(state_unitary, m0, m1)

    if st.session_state.actions.get("decode", False):
        psi_B_corr = apply_bob_corrections(psi_B, m0, m1)
        state_display = compose_product_state_SA_B(m0, m1, psi_B_corr)
    else:
        state_display = collapsed_full

# ----------------------------------------
# Bloch spheres (Plotly)
# ----------------------------------------

st.subheader("📊 Quantum States on Bloch Spheres")
bc1, bc2, bc3 = st.columns(3)

for idx, col, label in [(0, bc1, "S"), (1, bc2, "A"), (2, bc3, "B")]:
    with col:
        rho = reduced_density_matrix_from_state(state_display, idx, 3)
        vec = bloch_vector_from_rho(rho)

        fig = plot_bloch_vector_plotly(vec, title=label, res=45, show_grid=True)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        p0_i, p1_i = get_qubit_probabilities(state_display, idx, 3)
        st.caption(f"P(|0⟩)={p0_i:.2f}, P(|1⟩)={p1_i:.2f}")

st.divider()

# ----------------------------------------
# Circuit visualization
# ----------------------------------------

st.subheader("🔧 Current Quantum Circuit")

try:
    @qml.qnode(dev)
    def circ():
        if st.session_state.actions.get("encode", False):
            qml.RY(theta, wires=0)
            qml.RZ(phi, wires=0)
        if st.session_state.actions.get("entangle", False):
            qml.Hadamard(wires=1)
            qml.CNOT(wires=[1, 2])
        if st.session_state.actions.get("bell_basis", False):
            qml.CNOT(wires=[0, 1])
            qml.Hadamard(wires=0)

        if st.session_state.actions.get("decode", False) and st.session_state.measured:
            m0 = int(st.session_state.m0)  # S
            m1 = int(st.session_state.m1)  # A
            if m1 == 1:
                qml.PauliX(wires=2)
            if m0 == 1:
                qml.PauliZ(wires=2)

        return qml.state()

    fig_c, _ = qml.draw_mpl(circ, wire_order=[0, 1, 2])()
    st.pyplot(fig_c, clear_figure=True)
    st.caption("Wire 0: S | Wire 1: A | Wire 2: B")
except Exception:
    st.info("Toggle steps above to build the circuit.")

st.divider()

# ----------------------------------------
# Wavefunction display
# ----------------------------------------

st.subheader("🌊 Quantum State (Wavefunction)")

terms = state_to_dirac_terms(state_display)

st.markdown("**Individual Qubit States (best pure approx):**")
ws, wa, wb = st.columns(3)

for idx, col, name in [(0, ws, "S"), (1, wa, "A"), (2, wb, "B")]:
    with col:
        rho = reduced_density_matrix_from_state(state_display, idx, 3)
        psi = principal_pure_state_from_rho(rho)
        a0 = format_complex(psi[0])
        a1 = format_complex(psi[1])
        st.markdown(f"**{name}:**")
        st.latex(rf"|\psi_{name}\rangle = {a0}|0\rangle + {a1}|1\rangle")
        p0_i, p1_i = get_qubit_probabilities(state_display, idx, 3)
        st.caption(f"P(|0⟩)={p0_i:.3f}, P(|1⟩)={p1_i:.3f}")

st.markdown("---")
st.markdown("**Full 3-qubit composite state (nonzero terms):**")

latex_terms = []
for amp_str, bits, _p in terms:
    ket = rf"|{bits[0]}\rangle_S |{bits[1]}\rangle_A |{bits[2]}\rangle_B"
    if amp_str == "1.000":
        latex_terms.append(ket)
    else:
        coeff = amp_str
        if "+" in coeff or (coeff.count("-") >= 1 and not coeff.startswith("-")):
            coeff = f"({coeff})"
        latex_terms.append(rf"{coeff}\,{ket}")

wave_latex = " + ".join(latex_terms) if latex_terms else "0"
st.latex(rf"|\psi\rangle = {wave_latex}")

st.markdown("**State probabilities:**")
if terms:
    import pandas as pd
    dfp = pd.DataFrame(
        [{"Basis": f"|{bits}⟩", "Probability": f"{p:.4f}"} for (_a, bits, p) in sorted(terms, key=lambda t: -t[2])]
    )
    st.dataframe(dfp, hide_index=True, use_container_width=True)

st.divider()

# ----------------------------------------
# Measurement results panel
# ----------------------------------------

st.subheader("📡 Measurement Results")

if st.session_state.measured:
    m0 = int(st.session_state.m0)
    m1 = int(st.session_state.m1)
    outcome = f"{m0}{m1}"

    col_hist, col_info = st.columns([1.2, 1])

    with col_hist:
        if st.session_state.counts:
            st.pyplot(plot_measurement_histogram(st.session_state.counts), clear_figure=True)

    with col_info:
        outcome_map = {
            "00": "I (Identity - no correction needed)",
            "01": "X (bit flip)",
            "10": "Z (phase flip)",
            "11": "XZ (bit + phase flip)",
        }
        st.info(f"**Outcome: {outcome}**\n\nBob must apply:\n**{outcome_map[outcome]}**")

    with st.expander("🔍 What do these outcomes mean?"):
        st.markdown(r"""
The measurement outcome determines which correction Bob needs:

| Outcome (m0 m1) | Correction | Effect on Bob |
|---|---|---|
| **00** | **I** | do nothing |
| **01** | **X** | swaps \|0⟩ and \|1⟩ (bit flip) |
| **10** | **Z** | adds a minus sign to \|1⟩ (phase flip) |
| **11** | **XZ** | both flips |

**If Bob does NOT apply the correct operation**, his qubit becomes a *Pauli-rotated* version of Alice’s state.
That generally changes the Bloch vector (flips it) and drops the fidelity vs the intended \(|\psi\rangle\).
        """)
else:
    st.info("Click **Measure Alice** to generate results (requires Bell Basis setup).")

st.divider()

# ----------------------------------------
# Bob's message + fidelity
# ----------------------------------------

st.subheader("💌 Bob's Valentine's Message")

target_state = prep_psi_single(theta, phi)
rho_B = reduced_density_matrix_from_state(state_display, 2, 3)
fidelity = state_fidelity_density(rho_B, target_state)
prob_happy, prob_unlucky = get_qubit_probabilities(state_display, 2, 3)

mid = st.columns([1, 2, 1])[1]
with mid:
    if prob_happy >= 0.5:
        st.success(f"## 💝 Happy Valentine's Day!\n{prob_happy:.1%} probability")
    else:
        st.error(f"## 💔 Better luck next time\n{prob_unlucky:.1%} probability")

    st.metric("Fidelity", f"{fidelity:.4f}")

    if not st.session_state.actions.get("entangle", False):
        st.error("❌ No entanglement → teleportation can't work.")
    elif not st.session_state.measured:
        st.info("⏳ Measure (and optionally decode) to complete the story.")
    else:
        if st.session_state.actions.get("decode", False) and fidelity > 0.99:
            st.success("✅ Perfect teleportation (after measurement + correction).")
        elif st.session_state.actions.get("decode", False):
            st.warning("⚠️ Decode is on, but fidelity isn't perfect — try enabling all steps.")
        else:
            st.info("🔎 You measured, but decode is off — Bob hasn't corrected yet.")

    st.write(f"Alice intended: **{np.cos(theta/2)**2:.1%} Happy**")
    st.write(f"Bob has: **{prob_happy:.1%} Happy**")

st.divider()

# ----------------------------------------
# Teleportation analysis
# ----------------------------------------

st.subheader("🎯 Teleportation Analysis")
L, R = st.columns(2)

with L:
    st.metric("Fidelity F(|ψ⟩, Bob)", f"{fidelity:.6f}")
    st.markdown("**What is fidelity?**")
    st.write("- **F = 1.0** → Bob's qubit matches Alice's original |ψ⟩")
    st.write("- Lower values → missing steps / wrong setup / no entanglement")

with R:
    st.markdown("**Probability comparison:**")
    target_p0 = float(np.abs(target_state[0]) ** 2)
    target_p1 = float(np.abs(target_state[1]) ** 2)
    bob_p0, bob_p1 = prob_happy, prob_unlucky

    a1, b1 = st.columns(2)
    with a1:
        st.markdown("**Alice sent:**")
        st.write(f"💝 {target_p0:.1%}")
        st.write(f"💔 {target_p1:.1%}")
    with b1:
        st.markdown("**Bob has:**")
        st.write(f"💝 {bob_p0:.1%}")
        st.write(f"💔 {bob_p1:.1%}")

    if st.session_state.actions.get("decode", False) and st.session_state.measured:
        diff = abs(bob_p0 - target_p0)
        if diff < 0.01:
            st.success(f"✨ Match! Δ={diff:.4f}")
        else:
            st.warning(f"⚠️ Mismatch Δ={diff:.4f}")

st.divider()

with st.expander("ℹ️ What Does 'Teleportation Success' Mean?"):
    st.markdown("""
### Understanding Quantum Teleportation 🔮

**Important: Bob ALWAYS gets a message when he measures!** The question is: does it match what Alice sent?

**The Goal:** Alice wants to send Bob her quantum Valentine's message:
- |ψ⟩ = α|0⟩ + β|1⟩
- Where |α|² = probability of "Happy Valentine's Day!" 💝
- And |β|² = probability of "Better luck next time" 💔

**What Gets Teleported:**
- NOT a definite outcome (Bob's measurement is still probabilistic!)
- The **exact probability distribution** that Alice prepared
- The quantum state including amplitudes and phases

**Fidelity Tells Us If It Worked:**
- **F = 1.0**: 🎉 Perfect! Bob's probabilities exactly match Alice's
- **F = 0.9**: 😐 Close, but some error - probabilities slightly off
- **F = 0.5**: 😕 Random - Bob's message has nothing to do with Alice's
""")
