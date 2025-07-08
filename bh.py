import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium", app_title="Bosonic Hubbard Model")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Bose-Hubbard Model Analysis

    This notebook studies the ground state properties of the 1D Bose-Hubbard model. We analyze the ground state energy per site and the energy gap as functions of the system size.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from quspin.operators import hamiltonian
    from quspin.basis import boson_basis_1d
    from quspin.operators import quantum_LinearOperator

    plt.style.use('default')
    return boson_basis_1d, hamiltonian, mo, np, plt, quantum_LinearOperator


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Hamiltonian Construction

    We define a function to construct the Bose-Hubbard Hamiltonian for a given number of sites $L$ and particles $N$ using open boundary conditions. The hopping parameter is $J$ and the interaction strength is $U$.
    """
    )
    return


@app.cell(hide_code=True)
def _(boson_basis_1d, hamiltonian, np):
    def construct_bh_hamiltonian(L, Np, sps, J, U):
        basis = boson_basis_1d(L=int(L), Nb=[int(Np)], sps=sps)
        hop = [[-J, i, i + 1] for i in range(L - 1)]
        int_list = [[0.5 * U, i] for i in range(L)]
        static = [["+-", hop], ["-+", hop], ["n", int_list]]
        H = hamiltonian(static, [], basis=basis, dtype=np.float64, check_herm=False, check_pcon=False, check_symm=False)
        return H, basis
    return (construct_bh_hamiltonian,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Ground State Energy per Site vs. System Size

    We calculate and plot the ground state energy per site for several system sizes at unit filling.
    """
    )
    return


@app.cell(hide_code=True)
def _(construct_bh_hamiltonian):
    def compute_energy_vs_L(L_list, sps, J, U):
        energies = []
        for L in L_list:
            Np = L
            H, basis = construct_bh_hamiltonian(L, Np, sps, J, U)
            E0 = H.eigsh(k=1, which="SA", return_eigenvectors=False)[0]
            energies.append(E0 / L)
        return energies
    return (compute_energy_vs_L,)


@app.cell(hide_code=True)
def _(mo):
    # Create a form with multiple elements
    gs_input = (
        mo.md("""
                **Ground state sweep inputs** \n
                {L_range} \n
                {sps} \n
                {J} \n
                {U} \n
            """)
        .batch(
            L_range=mo.ui.range_slider(
                start=2, stop=30, step=2, value=[4, 14], label="L range: "
            ),
            sps=mo.ui.number(start=1, stop=30, step=1, value=4, label="Spins per site: "),
            J=mo.ui.number(start=0, stop=30, step=0.2, value=10, label="Value of J: "),
            U=mo.ui.number(start=0, stop=30, step=0.2, value=2, label="Value of U: "),
        )
        .form(show_clear_button=True, bordered=False)
    )

    gs_input
    return (gs_input,)


@app.cell(hide_code=True)
def _(gs_input, np):
    L_vals = np.arange(gs_input.value["L_range"][0], gs_input.value["L_range"][1], 2)
    sps = gs_input.value["sps"]
    J = gs_input.value["J"]
    U = gs_input.value["U"]
    return J, L_vals, U, sps


@app.cell(hide_code=True)
def _(J, L_vals, U, compute_energy_vs_L, mo, plt, sps):
    energies = compute_energy_vs_L(L_vals, sps, J, U)
    plt.plot(L_vals, energies, "o-", label="Energy per site")
    plt.xlabel('System size L')
    plt.ylabel('Ground state energy per site')
    plt.title("Ground state energy per site vs. system size")
    plt.grid(True)
    mo.center(plt.gca())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Energy Gap vs. System Size

    We compute the energy difference between the lowest states with $N$ and $N+1$ particles to estimate the energy gap.
    """
    )
    return


@app.cell(hide_code=True)
def _(construct_bh_hamiltonian):
    def compute_charge_gap_vs_L(L_list, sps, J, U):
        gaps = []
        for L in L_list:
            Np = L  # unit filling
            H0, _ = construct_bh_hamiltonian(L, Np, sps, J, U)
            Hm, _ = construct_bh_hamiltonian(L, Np - 1, sps, J, U)
            Hp, _ = construct_bh_hamiltonian(L, Np + 1, sps, J, U)

            E0 = H0.eigsh(k=1, which="SA", return_eigenvectors=False)[0]
            Em = Hm.eigsh(k=1, which="SA", return_eigenvectors=False)[0]
            Ep = Hp.eigsh(k=1, which="SA", return_eigenvectors=False)[0]

            gap = Ep + Em - 2 * E0
            gaps.append(gap)

        return gaps
    return (compute_charge_gap_vs_L,)


@app.cell(hide_code=True)
def _(J, L_vals, U, compute_charge_gap_vs_L, mo, plt, sps):
    charge_gaps = compute_charge_gap_vs_L(L_vals, sps, J, U)
    plt.plot(1 / L_vals, charge_gaps, 'o-', label='Charge gap Δ')
    plt.xlabel('1 / L')
    plt.ylabel('Charge gap Δ')
    plt.title(r'Charge gap vs 1/L (thermodynamic limit)')
    plt.grid(True)
    plt.legend()
    mo.center(plt.gca())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Expectation Value of Occupation Number

    We compute the expectation value of the occupation number at each site for the ground state.
    """
    )
    return


@app.cell(hide_code=True)
def _(
    J,
    U,
    construct_bh_hamiltonian,
    mo,
    np,
    plt,
    quantum_LinearOperator,
    sps,
):
    L = 8
    Np = L
    H, basis = construct_bh_hamiltonian(L, Np, sps, J, U)
    E, psi = H.eigsh(k=1, which='SA')
    n_op_list = [quantum_LinearOperator(basis=basis, static_list=[['n', [[1.0, i]]]], check_pcon=False, check_symm=False, check_herm=False) for i in range(L)]
    n_expect = np.array([n.expt_value(psi[:, 0]) for n in n_op_list])
    plt.figure()
    plt.bar(range(L), n_expect)
    plt.xlabel('Site index')
    plt.ylabel('<n>')
    plt.title('Expectation value of occupation number per site')
    mo.center(plt.gca())
    return


if __name__ == "__main__":
    app.run()
