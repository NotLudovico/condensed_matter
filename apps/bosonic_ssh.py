import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Bosonic SSH model""")
    return


@app.cell(hide_code=True)
def _():
    from quspin.basis import boson_basis_1d
    from quspin.operators import hamiltonian
    import numpy as np
    import marimo as mo
    import matplotlib.pyplot as plt

    plt.style.use("default")
    return boson_basis_1d, hamiltonian, mo, np, plt


@app.cell(hide_code=True)
def _(boson_basis_1d, hamiltonian, np):
    def bosonic_ssh_hamiltonian(*, L, J, deltaJ, U, Nbosons, sps, BC):
        match BC:
            case "Periodic":
                hop_list = [
                    [-(J + (-1) ** i * deltaJ), i, (i + 1) % L] for i in range(L)
                ]
            case "Open":
                hop_list = [
                    [-(J + (-1) ** i * deltaJ), i, i + 1] for i in range(L - 1)
                ]

        nn_list = [[U / 2, i, i] for i in range(L)]
        n_list = [[-U / 2, i] for i in range(L)]

        static = [
            ["+-", hop_list],
            ["-+", hop_list],
            ["nn", nn_list],
            ["n", n_list],
        ]

        basis = boson_basis_1d(L=L, Nb=Nbosons, sps=sps)
        H = hamiltonian(
            static,
            [],
            basis=basis,
            dtype=np.float64,
            check_herm=False,
            check_pcon=False,
            check_symm=False,
        )

        return H, basis
    return (bosonic_ssh_hamiltonian,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Ground-state symmetry""")
    return


@app.cell(hide_code=True)
def _(mo):
    BC = mo.ui.dropdown(value="Open", options=["Open", "Periodic"])
    L = mo.ui.number(value=6)
    U = mo.ui.number(value=2)
    J = mo.ui.number(value=1)
    sps = mo.ui.number(value=4)
    mo.hstack(
        [
            mo.vstack(
                [
                    mo.md("Boundary Conditions: "),
                    mo.md("System Size (L): "),
                    mo.md("U:"),
                    mo.md("J:"),
                    mo.md("Number of spins per site:"),
                ]
            ),
            mo.vstack([BC, L, U, J, sps]),
        ]
    )
    return BC, J, L, U, sps


@app.cell(hide_code=True)
def _(BC, J, L, U, bosonic_ssh_hamiltonian, np, sps):
    N = L.value // 2
    deltaJs = np.linspace(-1, 1, 10)
    E_gs = {}

    for deltaJ in deltaJs:
        H, _ = bosonic_ssh_hamiltonian(
            L=L.value,
            J=J.value,
            deltaJ=deltaJ,
            U=U.value,
            Nbosons=N + 1,
            BC=BC.value,
            sps=sps.value,
        )
        E, _ = H.eigh()
        E_gs[deltaJ] = E[0]
    return E_gs, N, deltaJs


@app.cell(hide_code=True)
def _(BC, E_gs, J, L, N, U, deltaJs, mo, plt, sps):
    plt.plot(
        [-dj for dj in deltaJs],
        [E_gs[dj] for dj in deltaJs],
        "o-",
        label=r"$E_{gs}(-\delta J)$",
    )

    plt.plot(
        deltaJs,
        [E_gs[dj] for dj in deltaJs],
        "o-",
        label=r"$E_{gs}(\delta J)$",
    )

    plt.xlabel(r"$\delta J$", fontsize=12)
    plt.ylabel(r"$E_{gs}$", fontsize=12)
    plt.title(
        rf"$L = {L.value}$, $N={N}$, $J={J.value}$, $U={U.value}$, $SPS={sps.value}$, {BC.value} Boundary Conditions"
    )
    plt.legend(frameon=False, fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    mo.center(plt.gca())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Charge Gap""")
    return


@app.cell(hide_code=True)
def _(mo):
    deltaJ_cg = mo.ui.number(value=0.2, step=0.1)
    J_cg = mo.ui.number(value=1, step=0.1)
    U_cg = mo.ui.array(
        [
            mo.ui.number(1, 20, value=2),
            mo.ui.number(1, 20, value=4),
            mo.ui.number(1, 20, value=10),
        ]
    )

    mo.hstack(
        [
            mo.vstack([mo.md("$\delta J$"), mo.md("$J$"), mo.md("$U$ values")]),
            mo.vstack([deltaJ_cg, J_cg, U_cg]),
        ]
    )
    return J_cg, U_cg, deltaJ_cg


@app.cell(hide_code=True)
def _(bosonic_ssh_hamiltonian):
    def gap(_L, _J,_deltaJ, _U):
        _N = _L // 2
        H_plus, _ = bosonic_ssh_hamiltonian(
            L=_L,
            J=_J,
            deltaJ=_deltaJ,
            U=_U,
            Nbosons=_N + 1,
            sps=_N + 2,
            BC="Periodic",
        )
        H_minus, _ = bosonic_ssh_hamiltonian(
            L=_L,
            J=_J,
            deltaJ=_deltaJ,
            U=_U,
            Nbosons=_N - 1,
            sps=_N,
            BC="Periodic",
        )
        _H, _ = bosonic_ssh_hamiltonian(
            L=_L,
            J=_J,
            deltaJ=_deltaJ,
            U=_U,
            Nbosons=_N,
            sps=_N + 1,
            BC="Periodic",
        )

        E_plus, _ = H_plus.eigh()
        E_minus, _ = H_minus.eigh()
        _E, _ = _H.eigh()

        E_gs_plus = E_plus[0]
        E_gs_minus = E_minus[0]
        _E_gs = _E[0]

        return E_gs_plus + E_gs_minus - 2 * _E_gs
    return (gap,)


@app.cell(hide_code=True)
def _(J_cg, U_cg, deltaJ_cg, gap):
    # varying params
    L_list = [4, 6, 8]
    U_list = U_cg.value

    gaps = {}
    for _U in U_list:
        gaps[_U] = {}
        for _L in L_list:
            gaps[_U][_L] = gap(_L, J_cg.value, deltaJ_cg.value, _U)
    return L_list, U_list, gaps


@app.cell(hide_code=True)
def _(L_list, U_list, gaps, mo, np, plt):
    # # Compute 1/L
    inv_L = np.array([1 / L for L in L_list])
    _inv_L_fit = np.linspace(0, max(inv_L), 300)

    for i, _U in enumerate(U_list):
        y = np.array([gaps[_U][L] for L in L_list])

        # Quadratic Fit
        coeffs = np.polyfit(inv_L, y, 1)
        fit = np.poly1d(coeffs)(_inv_L_fit)

        # Need to do this (line,) business to have the same color
        # from the theme for the dots and the fit lines
        (line,) = plt.plot(inv_L, y, "o", markersize=5)  # Calculated points
        c = line.get_color()
        plt.plot(_inv_L_fit, fit, "--", color=c, label=f"U = {_U}")  # Fit result


    plt.xlabel(r"$1/L$")
    plt.ylabel(r"$\Delta_c$")
    plt.xlim(0, 0.26)
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    mo.center(plt.gca())
    return (inv_L,)


@app.cell(hide_code=True)
def _(J_cg, L_list, deltaJ_cg, gap, inv_L, mo, np, plt):
    gap_at_infty = []
    _gaps = {}
    _U_list = np.linspace(0, 20, 20)

    for _U in _U_list:
        _gaps[_U] = {}
        for _L in L_list:
             _gaps[_U][_L] = gap(_L, J_cg.value, deltaJ_cg.value, _U)
        
        _y = np.array([_gaps[_U][_L] for _L in L_list])
        _coeffs = np.polyfit(inv_L, _y, 1)
        gap_L_inf = _coeffs[-1]  # termine costante
        gap_at_infty.append(gap_L_inf)

    plt.plot(_U_list, gap_at_infty, 'o-')
    plt.title(f"Gap in the thermodynamical limit $\delta J = {deltaJ_cg.value}$, $J={J_cg.value}$")
    plt.xlabel('$U$')
    plt.ylabel(r'$\Delta_c(L \to \infty)$')
    plt.grid(True, linestyle=':', alpha=0.6)
    mo.center(plt.gca())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Local Density Profile""")
    return


@app.cell(hide_code=True)
def _(mo):
    L_dp = mo.ui.number(value=6, label="Size ($L$): ")
    deltaJ_dp = mo.ui.number(value=0.9, step=0.1, label="$\delta J$: ")
    J_dp = mo.ui.number(value=1, label="$J$: ")
    U_dp = mo.ui.number(value=2, label="$U$: ")
    BC_dp = mo.ui.dropdown(
        value="Open", options=["Periodic", "Open"], label="Boundary Conditions:"
    )
    L_dp, deltaJ_dp, J_dp, U_dp, BC_dp
    return BC_dp, J_dp, L_dp, U_dp, deltaJ_dp


@app.cell(hide_code=True)
def _(
    BC_dp,
    J_dp,
    L_dp,
    U_dp,
    bosonic_ssh_hamiltonian,
    deltaJ_dp,
    hamiltonian,
    mo,
    np,
    plt,
):
    _N = (L_dp.value // 2) + 1

    H_obc, basis_obc = bosonic_ssh_hamiltonian(
        L=L_dp.value,
        J=J_dp.value,
        deltaJ=deltaJ_dp.value,
        U=U_dp.value,
        Nbosons=_N,
        sps=_N + 1,
        BC=BC_dp.value,
    )
    E_obc, V_obc = H_obc.eigh()
    gs_obc = V_obc[:, 0]

    n_avg_obc = np.real(
        np.array(
            [
                hamiltonian(
                    [["n", [[1.0, j]]]],
                    [],
                    basis=basis_obc,
                    check_herm=False,
                    check_pcon=False,
                    check_symm=False,
                ).expt_value(gs_obc)
                for j in range(L_dp.value)
            ]
        )
        / _N
    )

    x = np.arange(L_dp.value)
    plt.bar(x + 1, n_avg_obc)
    plt.ylim(0, 0.3)
    plt.title(f"Local Density Profile for $J = {J_dp.value}$,  $\delta J = {deltaJ_dp.value}$, $U = {U_dp.value}$")
    plt.ylabel(r"$\langle n_i \rangle$")
    plt.xlabel("Site")
    plt.xticks(x + 1)
    plt.grid(True, linestyle=':', alpha=0.6)
    mo.center(plt.gca())
    return


if __name__ == "__main__":
    app.run()
