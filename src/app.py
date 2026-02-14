import math
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace, state_fidelity
from qiskit.visualization import plot_bloch_vector, plot_histogram

# Aer import (varies by qiskit version)
try:
    from qiskit_aer import Aer
except Exception:
    from qiskit.providers.aer import Aer

# ----------------------------
# Constants / helpers
# ----------------------------
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)

# Qubit indices (wires): [S, A, B]
S, A, B = 0, 1, 2

STEP_LABELS = {
    0: "0) Start |000⟩",
    1: "1) Create entanglement (A–B)",
    2: "2) Prepare |ψ⟩ on S (θ, φ)",
    3: "3) Bell-measurement setup on (S,A)",
    4: "4) Alice measures (S,A) → (m0,m1)",
    5: "5) Send classical bits to Bob",
    6: "6) Bob applies corrections → teleported |ψ⟩",
}

def prep_psi_single(theta: float, phi: float) -> Statevector:
    """|psi> = cos(theta/2)|0> + e^{i phi} sin(theta/2)|1> (up to global phase)."""
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)
    qc.rz(phi, 0)
    return Statevector.from_instruction(qc)

def build_circuit_upto(step: int, theta: float, phi: float, entangle_on: bool) -> QuantumCircuit:
    """
    Build the teleportation circuit up to a given step.
    Uses a 'coherent correction' implementation at step 6:
      - CX(A->B) and CZ(S->B)
    This is equivalent to the classical feedforward corrections (X if m1, Z if m0),
    but avoids collapse bookkeeping and keeps the demo stable.
    """
    qc = QuantumCircuit(3)

    # Step 1: Entangle A and B (Bell pair), if enabled
    if step >= 1 and entangle_on:
        qc.h(A)
        qc.cx(A, B)

    # Step 2: Prepare |psi> on S
    if step >= 2:
        qc.ry(theta, S)
        qc.rz(phi, S)

    # Step 3: Bell-measurement setup on S and A
    if step >= 3:
        qc.cx(S, A)
        qc.h(S)

    # Step 6: Bob correction (implemented coherently)
    if step >= 6:
        # Equivalent to: X on B if m1 (A measurement) and Z on B if m0 (S measurement)
        qc.cx(A, B)   # X correction controlled by A
        qc.cz(S, B)   # Z correction controlled by S

    return qc

def bloch_vector_of_qubit(state_sv: Statevector, qubit: int) -> np.ndarray:
    """Return Bloch vector (x,y,z) of a single qubit reduced state. Length<1 means mixed state."""
    rho = DensityMatrix(state_sv)
    trace_out = [q for q in [S, A, B] if q != qubit]
    rho_red = partial_trace(rho, trace_out)  # 2x2 DensityMatrix
    r = rho_red.data
    x = float(np.real(np.trace(r @ PAULI_X)))
    y = float(np.real(np.trace(r @ PAULI_Y)))
    z = float(np.real(np.trace(r @ PAULI_Z)))
    return np.array([x, y, z], dtype=float)

def reduced_density_of_qubit(state_sv: Statevector, qubit: int) -> DensityMatrix:
    rho = DensityMatrix(state_sv)
    trace_out = [q for q in [S, A, B] if q != qubit]
    return partial_trace(rho, trace_out)

def measurement_histogram_for_alice(theta: float, phi: float, entangle_on: bool, shots: int, seed: int = 7):
    """
    Simulate Alice measuring S and A after Bell setup (step 3).
    Returns counts for the 2-bit string.
    """
    qc = build_circuit_upto(step=3, theta=theta, phi=phi, entangle_on=entangle_on)

    # Measure S->c0, A->c1
    qc_meas = QuantumCircuit(3, 2)
    qc_meas.compose(qc, inplace=True)
    qc_meas.measure(S, 0)
    qc_meas.measure(A, 1)

    backend = Aer.get_backend("qasm_simulator")
    job = backend.run(qc_meas, shots=shots, seed_simulator=seed)
    counts = job.result().get_counts()

    # Note: Qiskit typically reports bitstrings as "c1c0"
    return counts

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Quantum Teleportation Demo", layout="wide")
st.title("Quantum Teleportation (Streamlit + Qiskit)")

with st.sidebar:
    st.header("Controls")

    entangle_on = st.toggle("Entanglement ON (A–B Bell pair)", value=True)

    st.subheader("Pick the state |ψ⟩ on S")
    theta = st.slider("θ (0 → π)", min_value=0.0, max_value=float(math.pi), value=float(math.pi/2), step=0.01)
    phi = st.slider("φ (0 → 2π)", min_value=0.0, max_value=float(2*math.pi), value=0.0, step=0.01)

    st.subheader("Protocol step")
    step = st.slider("Step", min_value=0, max_value=6, value=6, step=1, format="%d")
    st.write(f"**{STEP_LABELS[step]}**")

    st.subheader("Measurement shots")
    shots = st.slider("Shots (for Alice's measurement histogram)", 10, 2000, 400, 10)

# Build state at current step (no measurements in the statevector circuit)
qc_step = build_circuit_upto(step=step if step != 4 and step != 5 else 3, theta=theta, phi=phi, entangle_on=entangle_on)
sv_step = Statevector.from_instruction(qc_step)

# Top row: three Bloch spheres
col1, col2, col3 = st.columns(3)

def show_bloch(col, label, qubit_index):
    with col:
        st.subheader(label)
        v = bloch_vector_of_qubit(sv_step, qubit_index)
        fig = plot_bloch_vector(v)
        st.pyplot(fig, clear_figure=True)
        st.caption(f"Bloch vector = [{v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f}]  (length {np.linalg.norm(v):.3f})")

show_bloch(col1, "S (state qubit)", S)
show_bloch(col2, "A (Alice entanglement qubit)", A)
show_bloch(col3, "B (Bob qubit)", B)

st.divider()

# Middle row: protocol explanation + histogram around measurement
left, right = st.columns([1.1, 1.0])

with left:
    st.subheader("What the user is doing at this step")
    st.markdown(
        """
- **Step 1:** Entangle **A–B** (this is the resource).
- **Step 2:** Prepare the unknown state **|ψ⟩** on **S** using θ and φ.
- **Step 3:** Alice performs the **Bell-measurement setup** on (S,A): CNOT(S→A), then H(S).
- **Step 4–5:** Alice measures (S,A) giving two random classical bits **(m0,m1)** and sends them to Bob.
- **Step 6:** Bob applies corrections (**X if m1**, **Z if m0**) and his qubit becomes **|ψ⟩**.
        """
    )

with right:
    st.subheader("Alice’s two classical bits (measurement results)")
    counts = measurement_histogram_for_alice(theta, phi, entangle_on, shots=shots)
    fig_h = plot_histogram(counts)
    st.pyplot(fig_h, clear_figure=True)

    st.caption("Qiskit typically shows bitstrings as **c1c0** where c0=measure(S), c1=measure(A).")
    # Show most common string just to give a concrete sample
    sample = max(counts, key=counts.get)
    if len(sample) == 2:
        m1 = int(sample[0])  # c1 = A
        m0 = int(sample[1])  # c0 = S
        st.write(f"Example outcome (most frequent in this run): **m0={m0}, m1={m1}**")
        st.write("So Bob would apply:", ("X " if m1 else "") + ("Z" if m0 else "I (no correction)") )

st.divider()

# Bottom: circuit + fidelity check
c1, c2 = st.columns([1.3, 0.9])

with c1:
    st.subheader("Circuit (up to the selected step)")
    fig_circ = qc_step.draw(output="mpl")
    st.pyplot(fig_circ, clear_figure=True)

with c2:
    st.subheader("Did Bob receive |ψ⟩?")
    psi_target = prep_psi_single(theta, phi)
    rho_B = reduced_density_of_qubit(sv_step, B)

    F = state_fidelity(rho_B, psi_target)
    st.metric("Fidelity F( Bob , |ψ⟩ )", f"{F:.6f}")

    # Also show Bob's measurement probabilities in Z-basis as a cute "Happy vs Valentine" mapping
    p0 = float(np.real(rho_B.data[0, 0]))
    p1 = float(np.real(rho_B.data[1, 1]))
    st.write("Bob measurement probabilities (Z-basis):")
    st.write(f"- P(|0⟩) = {p0:.3f}  → label this **Happy**")
    st.write(f"- P(|1⟩) = {p1:.3f}  → label this **Valentine’s Day**")

    if step < 6:
        st.info("Go to **Step 6** to complete teleportation and see fidelity jump (when entanglement is ON).")
    else:
        if entangle_on:
            st.success("With entanglement ON, fidelity should be ~1 (ideal simulation).")
        else:
            st.warning("With entanglement OFF, teleportation should fail (fidelity drops).")
