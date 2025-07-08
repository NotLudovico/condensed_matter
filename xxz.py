import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from quspin.operators import hamiltonian
    from quspin.basis import spin_basis_1d
    plt.style.use('default')
    return hamiltonian, mo, np, plt, spin_basis_1d


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Analysis of the 1D XXZ Heisenberg Spin Chain

    This notebook explores the ground state properties of the 1D spin-$\frac{1}{2}$ XXZ Heisenberg model using exact diagonalization techniques. The model captures rich quantum behavior, including quantum phase transitions, and serves as a benchmark system in the study of strongly correlated quantum systems.

    We'll proceed step by step, constructing the Hamiltonian, computing observables, and studying the scaling behavior of physical quantities like the energy gap and local magnetization.

    ## 1. The XXZ Model

    We consider the Hamiltonian of the spin-1/2 XXZ chain with open boundary conditions:

    $$
    H = \sum_{i=0}^{L-2} \left[ J_{xy}(S^+_i S^-_{i+1} + S^-_i S^+_{i+1}) + J_{zz} S^z_i S^z_{i+1} \right] + h_z \sum_{i=0}^{L-1} S^z_i
    $$

    - $J_{xy}$ controls the flip-flop (XY) interaction.
    - $J_{zz}$ represents the Ising-type interaction along the $z$ direction.
    - $h_z$ is a uniform magnetic field applied along the $z$ axis.
    - The spin operators $S^z_i$, $S^+_i$, $S^-_i$ act on site $i$.
    - $L$ is the length of the spin chain.

    The model conserves total $S^z_{	ext{tot}}$, which allows us to work within fixed magnetization sectors.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""️ 2. Building the XXZ Hamiltonian""")
    return


@app.cell(hide_code=True)
def _(hamiltonian, np, spin_basis_1d):
    def build_xxz_hamiltonian(L, Jzz=1.0, Jxy=1.0, hz=0.0, m=0):
        Nup = L // 2 + m
        basis = spin_basis_1d(L, Nup=Nup)
        J_zz = [[Jzz, i, i + 1] for i in range(L - 1)]
        J_xy = [[Jxy / 2.0, i, i + 1] for i in range(L - 1)]
        h_z = [[hz, i] for i in range(L)]
        static = [['+-', J_xy], ['-+', J_xy], ['zz', J_zz], ['z', h_z]]
        H = hamiltonian(static, [], basis=basis, dtype=np.float64, check_herm=False, check_symm=False, check_pcon=False)
        return (H, basis)
    return (build_xxz_hamiltonian,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Let's test the Hamiltonian construction for L=4, Jzz=1, hz=0.1, m=0. The output of 6 for space dimension makes sense because it's the result of ${4 \choose 2} = 6$, since by providing  $m=0$ we have 2 spins and 4 lattice sites""")
    return


@app.cell
def _(build_xxz_hamiltonian):
    H, basis = build_xxz_hamiltonian(L=4, Jzz=1.0, hz=0.1, m=0)
    print('Hilbert space dimension:', basis.Ns)
    print('Hamiltonian shape:', H.shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 3. Ground State Energy and the Spin Gap

    To analyze quantum phase transitions, we examine the energy difference between the lowest states in two magnetization sectors:

    $$
    \Delta = E_{m=1} - E_{m=0}
    $$

    This is known as the **spin gap**. It serves as a key quantity:

    - If $\Delta_o \rightarrow 0$ as $L_o \rightarrow \infty$, the system is in a **gapless phase**.
    - If $\Delta > 0$ in the thermodynamic limit, the system is in a **gapped phase**.

    The 1D XXZ chain exhibits a well-known transition:
    - For $J_{zz} < 1$, the system is gapless (critical XY phase).
    - For $J_{zz} > 1$, the system enters a gapped antiferromagnetic phase.
    - The critical point is at $J_{zz} = 1$.

    We compute this gap by evaluating the ground state energy in the $m=0$ and $m=1$ sectors as a function of chain length $L$.
    """
    )
    return


@app.cell(hide_code=True)
def _(build_xxz_hamiltonian, np):
    def energy_sweep(Jzz=1.0, Jxy=1.0, hz=0.0, m=0, maxsize=16):
        energies = []
        for L in range(4, maxsize + 1, 4):
            H, _ = build_xxz_hamiltonian(L, Jzz=Jzz, Jxy=Jxy, hz=hz, m=m)
            E0 = H.eigsh(k=1, which='SA', return_eigenvectors=False)[0]
            energies.append(E0)
        return np.array(energies)
    return (energy_sweep,)


@app.cell
def _(mo):
    gap_input = (
        mo.md("""
                **Ground state sweep inputs** \n
                {L_max} \n
                {Jzz} \n
                {Jxy} \n
                {hz} \n
            """)
        .batch(
            L_max=mo.ui.slider(
                start=4, stop=30, step=4, value=12, label="Max $L$ Value: "
            ),
            Jzz=mo.ui.number(start=0, stop=30, step=0.2, value=1, label="Value of $J_{zz}$:"),
            Jxy=mo.ui.number(start=0, stop=30, step=0.2, value=1, label="Value of $J_{xy}$:"),
            hz=mo.ui.number(start=0, stop=1000, step=0.1, value=0.1, label="Value of $h_z$:"),
        )
        .form(show_clear_button=True, bordered=False)
    )
    gap_input
    return (gap_input,)


@app.cell(hide_code=True)
def _(energy_sweep, gap_input, mo, np, plt):
    gap = energy_sweep(
        m=1,
        maxsize=gap_input.value["L_max"],
        hz=gap_input.value["hz"],
        Jzz=gap_input.value["Jzz"],
        Jxy=gap_input.value["Jxy"],
    ) - energy_sweep(
        m=0,
        maxsize=gap_input.value["L_max"],
        hz=gap_input.value["hz"],
        Jzz=gap_input.value["Jzz"],
        Jxy=gap_input.value["Jxy"],
    )
    Ls = np.arange(4, gap_input.value["L_max"] + 1, 4)

    plt.plot(Ls, gap, "o-")
    plt.xlabel("System size $L$")
    plt.ylabel("$\Delta E$")
    plt.title("Gap between $m=0$ and $m=1$ regimes")
    plt.grid(True)
    mo.center(plt.gca())
    return Ls, gap


@app.cell(hide_code=True)
def _(np):
    def fit_gap_to_thermodynamicLimit(gap, Ls, show_plot=True):
        x = 1.0 / Ls
        coeffs = np.polyfit(x, gap, 2)
        p = np.poly1d(coeffs)
        return coeffs[2], x, p
    return (fit_gap_to_thermodynamicLimit,)


@app.cell(hide_code=True)
def _(Ls, fit_gap_to_thermodynamicLimit, gap):
    tl_gap, x, p = fit_gap_to_thermodynamicLimit(gap, Ls)
    return p, tl_gap, x


@app.cell(hide_code=True)
def _(gap, mo, np, p, plt, x):
    xs = np.linspace(0, x.max(), 200)
    plt.scatter(x, gap, s=60, edgecolor="k", label="Gap data")
    plt.plot(xs, p(xs), "-", linewidth=2, label="Quadratic fit")
    plt.xlabel(r"$1/L$")
    plt.ylabel(r"$\Delta E$")
    plt.title("Gap Extrapolation to Thermodynamic Limit")
    plt.grid(True)
    plt.legend()
    mo.center(plt.gca())
    return


@app.cell(hide_code=True)
def _(mo, tl_gap):
    mo.md(rf"""Estimated gap in thermodynamic limit: **{tl_gap:.2}**""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 5. Local Magnetization Profile $\langle S^z_i \rangle$

    Once we obtain the ground state of the system, we can compute the expectation value of the local spin operator $S^z_i$ at each site.

    This observable provides insight into:
    - Edge effects or boundary-induced polarization
    - Broken symmetry in the antiferromagnetic phase
    - Uniform magnetization profiles in external fields

    The expectation value is computed as:

    $$
    \langle S^z_i \rangle = \langle \psi_0 | S^z_i | \psi_0 \rangle
    $$

    where $|\psi_0\rangle$ is the ground state of the Hamiltonian. To ensure a unique and stable ground state (especially for $m=0$), we include a small uniform field $h_z$.
    """
    )
    return


@app.cell(hide_code=True)
def _(hamiltonian, np, spin_basis_1d):
    def sz_expectation_per_site(L=12, Jzz=1.0, Jxy=1.0, hz=0.0, m=0):
        Nup = L // 2 + m
        basis = spin_basis_1d(L, Nup=Nup)
        J_zz = [[Jzz, i, i + 1] for i in range(L - 1)]
        J_xy = [[Jxy / 2.0, i, i + 1] for i in range(L - 1)]
        h_z = [[hz, i] for i in range(L)]
        static = [["+-", J_xy], ["-+", J_xy], ["zz", J_zz], ["z", h_z]]
        H = hamiltonian(
            static,
            [],
            basis=basis,
            dtype=np.float64,
            check_symm=False,
            check_herm=False,
            check_pcon=False,
        )
        _, psi0 = H.eigsh(k=1, which="SA")
        psi0 = psi0[:, 0]
        sz_vals = np.array(
            [
                np.vdot(
                    psi0,
                    hamiltonian(
                        [["z", [[1.0, i]]]],
                        [],
                        basis=basis,
                        check_symm=False,
                        check_herm=False,
                        check_pcon=False,
                    ).dot(psi0),
                ).real
                for i in range(L)
            ]
        )
        return sz_vals
    return (sz_expectation_per_site,)


@app.cell(hide_code=True)
def _(gap_input, mo, plt, sz_expectation_per_site):
    L = gap_input.value["L_max"]
    plt.figure(figsize=(8, 4))
    plt.plot(
        range(gap_input.value["L_max"]),
        sz_expectation_per_site(
            L, Jzz=gap_input.value["Jzz"], hz=gap_input.value["hz"], m=1
        ),
        "o-",
        label="$m=1$",
    )
    plt.xlabel("Site $i$")
    plt.ylabel("$\\langle S^z_i \\rangle$")
    plt.title(rf"Local Magnetization Profile for $L = {L}$, $h_z = {gap_input.value['hz']}$")

    plt.plot(
        range(gap_input.value["L_max"]),
        sz_expectation_per_site(
            L, Jzz=gap_input.value["Jzz"], hz=gap_input.value["hz"], m=0
        ),
        "o-",
        label="$m=0$",
    )
    plt.legend()

    plt.grid(True)
    plt.legend()
    mo.center(plt.gca())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Conclusion

    In this notebook, we systematically analyzed the 1D XXZ Heisenberg spin chain.

    Key steps included:
    - Defining and constructing the Hamiltonian using symmetry sectors
    - Calculating ground state energies in $m=0$ and $m=1$ magnetization sectors
    - Extracting the spin gap and extrapolating it to the thermodynamic limit
    - Computing site-resolved local magnetization $\langle S^z_i \rangle$

    These techniques provide a foundation for studying quantum phase transitions and low-energy physics in one-dimensional systems.
    """
    )
    return


if __name__ == "__main__":
    app.run()
