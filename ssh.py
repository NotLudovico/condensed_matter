import marimo

__generated_with = "0.14.10"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo
    from quspin.operators import hamiltonian, exp_op
    from quspin.basis import spinless_fermion_basis_1d
    from quspin.operators import hamiltonian
    from quspin.basis import boson_basis_1d
    from quspin.tools.block_tools import block_diag_hamiltonian
    import numpy as np
    import matplotlib.pyplot as plt

    plt.style.use("default")
    return (
        block_diag_hamiltonian,
        boson_basis_1d,
        hamiltonian,
        mo,
        np,
        plt,
        spinless_fermion_basis_1d,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$H = \sum_{j=0}^{L-1} -(J+(-1)^j\delta J)\left(c_jc^\dagger_{j+1} - c^\dagger_{j}c_{j+1}\right) + \Delta(-1)^jn_j$$

    where:

    - $L$ is the system size
    - $J$ is the hopping parameter
    - $\delta J$ controls the dimerization
    - $\Delta$ is the staggered potential
    - $\beta$ is the inverse temperature
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    input = (
        mo.md("""
        **Parameters** \n
            {L} \n
            {J} \n
            {deltaJ} \n
            {Delta} \n
            {beta}
        """)
        .batch(
            L=mo.ui.number(value=100, label="L: "),
            J=mo.ui.number(value=1, label="J: "),
            deltaJ=mo.ui.number(value=0.1, label="$\delta J$:"),
            Delta=mo.ui.number(value=0.5, label="$\Delta$:"),
            beta=mo.ui.number(value=100, label=r"$\beta$:"),
        )
        .form(show_clear_button=True, bordered=False)
    )
    input
    return (input,)


@app.cell(hide_code=True)
def _(
    block_diag_hamiltonian,
    hamiltonian,
    input,
    mo,
    np,
    plt,
    spinless_fermion_basis_1d,
):
    L = input.value["L"]
    J = input.value["J"]
    deltaJ = input.value["deltaJ"]
    Delta = input.value["Delta"]
    beta = input.value["beta"]

    ##### construct single-particle Hamiltonian #####
    hop_pm = [[-J - deltaJ * (-1) ** i, i, (i + 1) % L] for i in range(L)]  # PBC
    hop_mp = [[+J + deltaJ * (-1) ** i, i, (i + 1) % L] for i in range(L)]  # PBC
    stagg_pot = [[Delta * (-1) ** i, i] for i in range(L)]
    # define static and dynamic lists
    static = [["+-", hop_pm], ["-+", hop_mp], ["n", stagg_pot]]
    dynamic = []
    # define basis
    basis = spinless_fermion_basis_1d(L, Nf=1)
    # build real-space Hamiltonian
    H = hamiltonian(
        static,
        dynamic,
        basis=basis,
        dtype=np.float64,
        check_herm=False,
        check_pcon=False,
        check_symm=False,
    )


    E, V = H.eigh()
    blocks = [dict(Nf=1, kblock=i, a=2) for i in range(L // 2)]
    basis_args = (L,)
    FT, Hblock = block_diag_hamiltonian(
        blocks,
        static,
        dynamic,
        spinless_fermion_basis_1d,
        basis_args,
        np.complex128,
        get_proj_kwargs=dict(pcon=True),
        check_herm=False,
        check_pcon=False,
        check_symm=False,
    )
    Eblock, Vblock = Hblock.eigh()

    # Plotting
    plt.plot(np.arange(H.Ns), E / L, marker="o", color="b", label="real space")
    plt.plot(
        np.arange(Hblock.Ns),
        Eblock / L,
        marker="x",
        color="r",
        markersize=2,
        label="momentum space",
    )
    plt.xlabel("state number", fontsize=16)
    plt.ylabel("energy", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.grid()
    mo.center(plt.gca())
    return J, deltaJ


@app.cell(hide_code=True)
def _(boson_basis_1d, hamiltonian, np):
    def n_expectation_per_site(N, Np, J, U, V, delta, sps):
        basis = boson_basis_1d(N, Nb=Np, sps=sps)

        # Liste di termini per l'Hamiltoniana
        hopping = [[-J + (-1) ** i * delta, i, (i + 1) % N] for i in range(N - 1)]
        int_bb = [[0.5 * U, j, j] for j in range(N)]
        int_b = [[-0.5 * U, j] for j in range(N)]

        static = [
            ["+-", hopping],
            ["-+", hopping],
            ["n", int_b],
            ["nn", int_bb],
        ]

        H = hamiltonian(
            static,
            [],
            basis=basis,
            dtype=np.float64,
            check_herm=False,
            check_pcon=False,
            check_symm=False,
        )

        # Calcolo del ground state
        E0, psi0 = H.eigsh(k=1, which="SA")
        psi0 = psi0[:, 0]

        # Calcolo di ⟨n_i⟩ su ogni sito
        n_vals = np.array(
            [
                np.vdot(
                    psi0,
                    hamiltonian(
                        [["n", [[1.0, i]]]],
                        [],
                        basis=basis,
                        check_herm=False,
                        check_pcon=False,
                        check_symm=False,
                    ).dot(psi0),
                ).real
                for i in range(N)
            ]
        )

        return n_vals
    return (n_expectation_per_site,)


@app.cell(hide_code=True)
def _(mo):
    N = mo.ui.number(value=10, label="N:")
    Np = mo.ui.number(value=6, label="Np:")
    U = mo.ui.number(value=100, label="U:")
    V_s = mo.ui.number(value=0, label="V:")
    sps = mo.ui.number(value=5, label="Sps:")

    mo.vstack([N, Np, U, V_s, sps])
    return N, Np, U, V_s, sps


@app.cell(hide_code=True)
def _(J, N, Np, U, V_s, deltaJ, mo, n_expectation_per_site, plt, sps):
    n_vals = n_expectation_per_site(
        N=N.value,
        Np=Np.value,
        J=J,
        U=U.value,
        V=V_s.value,
        delta=deltaJ,
        sps=sps.value,
    )
    n_vals_del = n_expectation_per_site(
        N=10, Np=6, J=1, U=100, V=0, delta=0.3, sps=5
    )

    plt.plot(range(len(n_vals)), n_vals, marker="o")
    plt.plot(range(len(n_vals_del)), n_vals, marker="o")
    plt.xlabel("Sito i")
    plt.ylabel(r"$\langle n_i \rangle$")
    plt.title("Occupazione media per sito")
    plt.grid(True, linestyle=':', alpha=0.6)
    mo.center(plt.gca())
    return


if __name__ == "__main__":
    app.run()
