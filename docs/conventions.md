# Conventions

## SYK model

$$H = \sum_{i<j<k<l} J_{ijkl}\, \chi_i \chi_j \chi_k \chi_l, \qquad
\langle J_{ijkl}^2 \rangle = \frac{3!\, J^2}{N^3}.$$

The Majoranas are Jordan–Wigner strings, $\chi_{2j} = Z_0\cdots Z_{j-1} X_j$ and
$\chi_{2j+1} = Z_0\cdots Z_{j-1} Y_j$, so $\chi^2 = 1$. Maldacena & Stanford
(PRD 94, 106002, 2016) and Cotler et al. (JHEP 05 (2017) 118) use
$\{\chi_a, \chi_b\} = \delta_{ab}$; with the same couplings their Hamiltonian is
$H/4$ (`SYKHamiltonian(..., chi_squared=0.5)`).

**Symmetry classes.** The fermion parity $\Gamma = \prod_j Z_j$ splits the
spectrum into two sectors. The antiunitary particle-hole operator
$P = (\chi_0\chi_2\cdots\chi_{N-2})K$ satisfies $[P,H]=0$,
$P^2 = (-1)^{n(n-1)/2}$ and $P\Gamma P^{-1} = (-1)^n\Gamma$ with $n=N/2$, so per
sector the class is GOE for $N \bmod 8 = 0$, GSE for $N \bmod 8 = 4$ (Kramers
pairs inside each sector) and GUE for $N \bmod 8 = 2, 6$ (the two sectors are
degenerate with each other).

## Spacing ratio

$\tilde r_n = \min(s_n, s_{n-1}) / \max(s_n, s_{n-1})$ with $s_n = E_{n+1}-E_n$,
computed inside one symmetry sector. Sectors are never merged; exact
degeneracies are kept (and reported), never filtered out. Reference values:
Atas et al., PRL 110, 084101 (2013): Poisson $2\ln 2 - 1 = 0.3863$, GOE 0.5307,
GUE 0.5996, GSE 0.6744.

## Hubbard ring

$$H = -t\sum_{j\sigma}(c^\dagger_{j+1\sigma}c_{j\sigma} + \text{h.c.})
      -t'\sum_{j\sigma}(c^\dagger_{j+2\sigma}c_{j\sigma} + \text{h.c.})
      + U\sum_j n_{j\uparrow}n_{j\downarrow}$$

with periodic boundary conditions for the fermion operators. Resolved
symmetries: $N_\uparrow$, $N_\downarrow$, momentum, reflection at $K=0,\pi$,
total spin $S$, and — for $t'=0$ and even $L$ — the η-spin. For $t'\neq 0$ the
expected class is GOE in every momentum sector (time reversal × reflection).

## Noise model

T1/T2 relaxation for the physical duration of each step, with pure dephasing
$1/T_\varphi = 1/T_2 - 1/(2T_1)$ (requires $T_2 \le 2T_1$), followed by one
depolarizing error $\rho \to (1-p)\rho + p\,I/2$ per gate layer (average gate
fidelity $1-p/2$; PennyLane's `DepolarizingChannel` parameter is $3p/4$).
