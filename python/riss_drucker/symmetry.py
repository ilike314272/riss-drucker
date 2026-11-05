"""Finite molecular symmetry groups: point groups, permutation-inversion (PI)
groups with large-amplitude coordinates, and double groups.

A group is generated from a few generators. Each element carries
* R      equivalent rotation (proper 3x3), acting on molecule-fixed J and S,
* u      its SU(2) lift (part of the element identity only for double groups),
* parity -1 for E* type elements (improper point operation / inversion),
* perm   permutation of identical nuclei (for spin statistics),
* coords affine action gamma -> s*gamma + c on each LAM coordinate.

Operator action convention (README conventions section 18):
    g.J_a = sum_b R_ba J_b,   g.a_c = sum_c' rho(g)_c'c a_c',   g.f(gamma) = f(s gamma + c)
so that R(gh) = R(g) R(h), rho(gh) = rho(g) rho(h), and h_gh = h_h o h_g for the
coordinate maps.

Irreducible representations are found numerically by decomposing the regular
representation (random element of the commutant), split over C where needed,
and named from a character table on the generators. Character-based results
(counts, multiplicities, Frobenius-Schur, spin weights) are integers recovered
by rounding; non-integrality is an error, never rounded away silently.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

SIGMA = (np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]], complex))
TWO_PI = 2 * math.pi


class SymmetryError(ValueError):
    pass


# ---- rotations and SU(2) ------------------------------------------------------------

def rotation_from_su2(u: np.ndarray) -> np.ndarray:
    """R_kj = 1/2 tr(sigma_k u sigma_j u^dagger)  (u sigma_j u^dagger = sum_k R_kj sigma_k)."""
    return np.array([[0.5 * np.trace(SIGMA[k] @ u @ SIGMA[j] @ u.conj().T).real for j in range(3)]
                     for k in range(3)])


def su2_from_axis_angle(n, theta: float) -> np.ndarray:
    n = np.asarray(n, float) / np.linalg.norm(n)
    return math.cos(theta / 2) * np.eye(2) - 1j * math.sin(theta / 2) * sum(n[k] * SIGMA[k] for k in range(3))


def su2_lift(R: np.ndarray) -> np.ndarray:
    """A lift of the proper rotation R (rotation angle in [0, pi])."""
    R = np.asarray(R, float)
    if abs(np.linalg.det(R) - 1) > 1e-8 or np.max(np.abs(R @ R.T - np.eye(3))) > 1e-8:
        raise SymmetryError("equivalent rotation must be a proper orthogonal matrix")
    c = (np.trace(R) - 1) / 2
    av = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2
    theta = math.atan2(float(np.linalg.norm(av)), float(c))     # well conditioned at 0 and pi
    if theta < 1e-12:
        return np.eye(2, dtype=complex)
    if abs(theta - math.pi) < 1e-3:
        w, V = np.linalg.eigh((R + R.T) / 2)        # axis: eigenvector with eigenvalue +1
        n = V[:, int(np.argmax(w))]
    else:
        n = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    for sgn in (1, -1):
        u = su2_from_axis_angle(sgn * n, theta)
        if np.max(np.abs(rotation_from_su2(u) - R)) < 1e-9:
            return u
    raise SymmetryError("could not lift rotation to SU(2)")


def su2_angle_axis(u: np.ndarray):
    """u = cos(t/2) - i sin(t/2) n.sigma with t in [0, 2pi]; returns (t, n)."""
    c = float(np.clip(np.trace(u).real / 2, -1, 1))
    t = 2 * math.acos(c)
    s = math.sin(t / 2)
    if abs(s) < 1e-12:
        return t, np.array([0.0, 0.0, 1.0])
    n = np.array([(1j * np.trace(SIGMA[k] @ u)).real / (2 * s) for k in range(3)])
    return t, n / np.linalg.norm(n)


def spin_character(two_j: int, u: np.ndarray) -> float:
    """chi^J(u) = sin((2J+1) t/2) / sin(t/2), t the SU(2) angle in [0, 2pi]."""
    t, _ = su2_angle_axis(u)
    s = math.sin(t / 2)
    if abs(s) < 1e-10:
        return float((two_j + 1) * (1 if math.cos(t / 2) > 0 else (-1) ** two_j))
    return math.sin((two_j + 1) * t / 2) / s


def angular_momentum_matrices(two_j: int):
    """Standard (space-fixed convention) Jx, Jy, Jz in the basis k = J, J-1, ..., -J? No:
    ascending k = -J..J (index i <-> k = -J + i)."""
    J = two_j / 2
    ks = np.array([-J + i for i in range(two_j + 1)])
    jp = np.zeros((two_j + 1, two_j + 1))
    for i in range(two_j):
        k = ks[i]
        jp[i + 1, i] = math.sqrt(J * (J + 1) - k * (k + 1))
    jx = (jp + jp.T) / 2
    jy = (jp - jp.T) / (2j)
    return jx.astype(complex), jy, np.diag(ks).astype(complex)


def wigner_D(two_j: int, u: np.ndarray) -> np.ndarray:
    """D^J(u) = exp(-i t n.J) (standard matrices), a genuine rep of SU(2)."""
    from scipy.linalg import expm
    t, n = su2_angle_axis(u)
    jx, jy, jz = angular_momentum_matrices(two_j)
    return expm(-1j * t * (n[0] * jx + n[1] * jy + n[2] * jz))


# ---- elements and groups ------------------------------------------------------------

@dataclass
class Element:
    R: np.ndarray
    u: np.ndarray
    parity: int = 1
    perm: tuple = ()
    coords: tuple = ()          # ((name, s, c), ...) sorted by name
    gauge: bool = False         # generator flag; physical states are invariant under gauge elements

    def coord(self, name: str):
        for n, s, c in self.coords:
            if n == name:
                return s, c
        return 1, 0.0


def make_element(R=None, *, O=None, parity=None, perm=(), coords=None, gauge=False) -> Element:
    """From an equivalent rotation R (with parity) or a point operation O (R = det(O) O)."""
    if O is not None:
        O = np.asarray(O, float)
        d = round(float(np.linalg.det(O)))
        R = d * O
        parity = d if parity is None else parity
    R = np.asarray(R, float)
    coords = tuple(sorted((k, int(v[0]), float(v[1])) for k, v in (coords or {}).items()))
    return Element(R, su2_lift(R), int(parity or 1), tuple(perm), coords, gauge)


class Group:
    """Closure of generators; irreps from the regular representation.

    names: {irrep name: (dim, [character on each generator])} used to name the
           irreps found (others are named G<dim>_<k>, spinor irreps S<dim>_<k>).
    preferred: {irrep name: [matrix for each generator]} to fix a basis
           (e.g. F2 = the Cartesian vector representation); verified.
    """

    def __init__(self, name: str, generators: list[Element], *, double: bool = False,
                 periods: dict | None = None, names: dict | None = None, preferred: dict | None = None,
                 max_order: int = 2000):
        self.name = name
        self.double = bool(double)
        self.periods = dict(periods or {})
        self.generators = list(generators)
        ident = Element(np.eye(3), np.eye(2, dtype=complex), 1, tuple(range(len(generators[0].perm))) if generators
                        and generators[0].perm else (), tuple((n, 1, 0.0) for n, _, _ in
                                                              (generators[0].coords if generators else ())))
        self.elements = [ident]
        self.word = [(-1, -1)]
        index = {self._key(ident): 0}
        frontier = [0]
        while frontier:
            nxt = []
            for i in frontier:
                for gi, gen in enumerate(self.generators):
                    h = self.mul(self.elements[i], gen)
                    k = self._key(h)
                    if k not in index:
                        index[k] = len(self.elements)
                        self.elements.append(h)
                        self.word.append((i, gi))
                        nxt.append(index[k])
                        if len(self.elements) > max_order:
                            raise SymmetryError(f"group {name} exceeds order {max_order}")
            frontier = nxt
        self._index = index
        n = len(self.elements)
        self.order = n
        self.table = np.array([[index[self._key(self.mul(a, b))] for b in self.elements] for a in self.elements])
        self.identity = 0
        self.inverse = np.array([int(np.where(self.table[i] == 0)[0][0]) for i in range(n)])
        self.gauge_elements = self._gauge_subgroup()
        self._classes()
        self._find_irreps(names or {}, preferred or {})

    # -- element algebra -------------------------------------------------------------
    def mul(self, g: Element, h: Element) -> Element:
        perm = tuple(g.perm[i] for i in h.perm) if g.perm else h.perm
        cg, ch = dict((n, (s, c)) for n, s, c in g.coords), dict((n, (s, c)) for n, s, c in h.coords)
        coords = []
        for name in sorted(set(cg) | set(ch)):
            sg, c_g = cg.get(name, (1, 0.0))
            sh, c_h = ch.get(name, (1, 0.0))
            per = self.periods.get(name, TWO_PI)
            coords.append((name, sh * sg, float((sh * c_g + c_h) % per)))
        return Element(g.R @ h.R, g.u @ h.u, g.parity * h.parity, perm, tuple(coords))

    def _key(self, g: Element):
        def r(x):
            return tuple(np.round(np.asarray(x).ravel() + 0.0, 6).tolist())
        coords = []
        for n, s, c in g.coords:
            per = self.periods.get(n, TWO_PI)
            cc = round(c % per, 6)
            coords.append((n, s, 0.0 if abs(cc - per) < 1e-6 else cc))
        k = (r(g.R), g.parity, g.perm, tuple(coords))
        if self.double:
            k = k + (r(g.u.real), r(g.u.imag))
        return k

    def find(self, g: Element) -> int:
        return self._index[self._key(g)]

    def rep_from_generators(self, mats: list[np.ndarray]) -> list[np.ndarray]:
        out = [None] * self.order
        d = np.asarray(mats[0]).shape[0] if mats else 1
        out[0] = np.eye(d, dtype=complex)
        for i in range(1, self.order):
            p, gi = self.word[i]
            out[i] = out[p] @ np.asarray(mats[gi], complex)
        return out

    def is_homomorphism(self, rep: list[np.ndarray], tol: float = 1e-8) -> bool:
        for a in range(self.order):
            for b in range(self.order):
                if np.max(np.abs(rep[a] @ rep[b] - rep[self.table[a, b]])) > tol:
                    return False
        return True

    def _gauge_subgroup(self) -> list[int]:
        gens = [i for i, g in enumerate(self.generators) if g.gauge]
        if not gens:
            return [0]
        sub = {0}
        frontier = [0]
        gidx = [self.find(self.generators[i]) for i in gens]
        while frontier:
            nxt = []
            for a in frontier:
                for b in gidx:
                    c = int(self.table[a, b])
                    if c not in sub:
                        sub.add(c)
                        nxt.append(c)
            frontier = nxt
        return sorted(sub)

    def _classes(self):
        seen, classes = set(), []
        for g in range(self.order):
            if g in seen:
                continue
            cl = sorted({int(self.table[self.table[h, g], self.inverse[h]]) for h in range(self.order)})
            seen.update(cl)
            classes.append(cl)
        self.classes = classes

    # -- irreps ----------------------------------------------------------------------
    def _regular(self):
        n = self.order
        mats = []
        for g in range(n):
            M = np.zeros((n, n))
            M[self.table[g], np.arange(n)] = 1.0
            mats.append(M)
        return mats

    @staticmethod
    def _eigenspaces(A, tol):
        w, V = np.linalg.eigh(A)
        spaces, i = [], 0
        while i < len(w):
            j = i
            while j + 1 < len(w) and abs(w[j + 1] - w[i]) < tol:
                j += 1
            spaces.append(V[:, i:j + 1])
            i = j + 1
        return spaces

    def _find_irreps(self, names, preferred):
        rng = np.random.default_rng(12345)
        reg = self._regular()
        n = self.order
        X = rng.standard_normal((n, n))
        X = X + X.T
        A = sum(M @ X @ M.T for M in reg)
        found = []

        def char_of(rep):
            return np.array([np.trace(m) for m in rep])

        for V in self._eigenspaces(A, 1e-7 * max(1.0, np.max(np.abs(A)))):
            rep = [V.T @ M @ V for M in reg]
            chi = char_of(rep)
            norm = round(float(np.real(np.vdot(chi, chi))) / n)
            if norm == 1:
                pieces = [[m.astype(complex) for m in rep]]
            elif norm in (2, 4):
                d = V.shape[1]
                Y = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
                Y = Y + Y.conj().T
                B = sum(m @ Y @ m.T for m in rep)
                pieces = [[W.conj().T @ m @ W for m in rep]
                          for W in self._eigenspaces(B, 1e-7 * max(1.0, np.max(np.abs(B))))]
            else:
                raise SymmetryError(f"regular-representation splitting failed (norm {norm})")
            for p in pieces:
                c = char_of(p)
                if not any(np.max(np.abs(c - f[1])) < 1e-6 for f in found):
                    found.append((p, c))
        total = sum(round(f[1][0].real) ** 2 for f in found)
        if total != n:
            raise SymmetryError(f"irreps found have sum of squared dimensions {total} != |G| = {n}")
        # naming
        gen_idx = [self.find(g) for g in self.generators]
        self.irreps, self.chars, used = {}, {}, set()
        counters = {}
        for rep, chi in sorted(found, key=lambda f: (round(f[1][0].real), -np.real(np.sum(f[1])))):
            d = round(chi[0].real)
            name = None
            for nm, (nd, gchars) in names.items():
                if nd == d and nm not in used and all(abs(chi[gi] - complex(gc)) < 1e-6
                                                      for gi, gc in zip(gen_idx, gchars)):
                    name = nm
                    break
            if name is None:
                spinor = self.double and self._is_spinor(chi)
                key = ("S" if spinor else "G") + str(d)
                counters[key] = counters.get(key, 0) + 1
                name = f"{key}_{counters[key]}"
            used.add(name)
            if name in preferred:
                pr = self.rep_from_generators(preferred[name])
                if not self.is_homomorphism(pr) or np.max(np.abs(char_of(pr) - chi)) > 1e-6:
                    raise SymmetryError(f"preferred matrices for {name} are not that irrep")
                rep = pr
            elif np.max(np.abs(np.imag(np.array(rep)))) < 1e-10:
                rep = [m.real.astype(complex) for m in rep]
            self.irreps[name] = rep
            self.chars[name] = chi
        self.trivial = next(nm for nm, c in self.chars.items() if np.max(np.abs(c - 1)) < 1e-9)

    def _is_spinor(self, chi) -> bool:
        for i, g in enumerate(self.elements):
            if np.max(np.abs(g.R - np.eye(3))) < 1e-6 and np.max(np.abs(g.u + np.eye(2))) < 1e-6 \
                    and g.parity == 1 and not any(c for _, _, c in g.coords) and \
                    (not g.perm or g.perm == tuple(range(len(g.perm)))):
                return abs(chi[i] + chi[0]) < 1e-6
        return False

    def subgroup(self, keep, name: str, names: dict | None = None) -> "Group":
        """Subgroup of the elements with keep(element) true (e.g. the symmetry of an isotopologue).
        The parent's irrep matrices restricted to it are kept as `restricted[parent_irrep]`
        (usable by spaces declared with the parent's irrep names)."""
        els = [g for g in self.elements if keep(g)]
        sub = Group(name, els[1:] if len(els) > 1 else [], double=self.double, periods=self.periods,
                    names=names or {})
        sub.restricted = {nm: [rep[self.find(g)] for g in sub.elements] for nm, rep in self.irreps.items()}
        return sub

    def rep(self, name: str):
        """Irrep matrices by name, falling back to restricted parent irreps."""
        if name in self.irreps:
            return self.irreps[name]
        return getattr(self, "restricted", {})[name]

    # -- character algebra ---------------------------------------------------------------
    def physical(self, name: str) -> bool:
        """Invariant under the gauge subgroup (character = dim on every gauge element)."""
        chi = self.chars[name]
        return all(abs(chi[g] - chi[0]) < 1e-9 for g in self.gauge_elements)

    def conjugate(self, name: str) -> str:
        c = np.conj(self.chars[name])
        return next(nm for nm, x in self.chars.items() if np.max(np.abs(x - c)) < 1e-6)

    def frobenius_schur(self, name: str) -> int:
        chi = self.chars[name]
        v = sum(chi[self.table[g, g]] for g in range(self.order)) / self.order
        r = round(v.real)
        if abs(v - r) > 1e-6:
            raise SymmetryError("non-integer Frobenius-Schur indicator")
        return int(r)

    def decompose(self, chi, tol: float = 1e-6) -> dict[str, int]:
        chi = np.asarray(chi, complex)
        out = {}
        for nm, c in self.chars.items():
            m = np.vdot(c, chi) / self.order
            r = round(m.real)
            if abs(m - r) > tol:
                raise SymmetryError(f"non-integer multiplicity {m} of {nm}: character is not a representation")
            if r:
                out[nm] = int(r)
        return out

    def rotation_character(self, two_j: int) -> np.ndarray:
        """Character of D^J restricted to the group through the equivalent rotations."""
        if two_j % 2 and not self.double:
            raise SymmetryError("half-integer J needs a double group (double=True)")
        return np.array([spin_character(two_j, g.u) for g in self.elements])

    def level_counts(self, two_j: int, other=None) -> dict[str, int]:
        chi = self.rotation_character(two_j)
        if other is not None:
            chi = chi * np.asarray(other)
        return self.decompose(chi)

    def symmetric_power_character(self, name: str, n: int) -> np.ndarray:
        """chi of Sym^n(rho) from the eigenvalues of rho(g): complete homogeneous polynomial h_n."""
        out = []
        for m in self.irreps[name]:
            lam = np.linalg.eigvals(m)
            out.append(complete_homogeneous(lam, n))
        return np.array(out)

    def intertwiners(self, a: str, b: str, c: str) -> np.ndarray:
        """Orthonormal basis of maps C: V_c -> V_a (x) V_b with (rho_a x rho_b) C = C rho_c.
        Their number equals the multiplicity of c in a x b (certificate); the basis
        within the multiplicity space is a gauge (here: eigenvectors of the projector)."""
        ra, rb, rc = self.irreps[a], self.irreps[b], self.irreps[c]
        da, db, dc = ra[0].shape[0], rb[0].shape[0], rc[0].shape[0]
        N = da * db * dc
        P = np.zeros((N, N), complex)
        for g in range(self.order):
            L = np.kron(ra[g], rb[g])
            # vec(L C R^dagger) = (conj(R) x L) vec(C)  (column-major vec)
            P += np.kron(rc[g].conj(), L)
        P /= self.order
        w, V = np.linalg.eigh((P + P.conj().T) / 2)
        sel = V[:, w > 0.5]
        m = sel.shape[1]
        expected = self.decompose(self.chars[a] * self.chars[b]).get(c, 0)
        if m != expected:
            raise SymmetryError(f"intertwiner count {m} != character multiplicity {expected}")
        return np.array([sel[:, k].reshape((da * db, dc), order="F") for k in range(m)])

    def jahn_teller_active(self, el: str) -> dict[str, int]:
        """Vibrational irreps in the symmetric square [Gamma_el^2] other than the totally
        symmetric one (linear JT-active modes; spin-free, real electronic states)."""
        chi = self.chars[el]
        sq = np.array([chi[self.table[g, g]] for g in range(self.order)])
        sym2 = (chi**2 + sq) / 2
        d = self.decompose(sym2)
        d.pop(self.trivial, None)
        return d

    def spin_statistics(self, spins: list[float], tol: float = 1e-6) -> dict[str, int]:
        """Nuclear-spin statistical weight of each rovibronic irrep.

        spins[i] = nuclear spin of nucleus i (perm acts on these indices). The total
        wavefunction must change sign under odd permutations of identical fermions;
        under E* elements both parities are allowed.
        """
        if not self.generators or not self.elements[1:] or not self.elements[0].perm:
            raise SymmetryError("group elements carry no nuclear permutations")
        two_i = [round(2 * s) for s in spins]
        chi_ns, sign = [], []
        for g in self.elements:
            p = g.perm
            seen, prod, sgn = set(), 1, 1
            for i in range(len(p)):
                if i in seen:
                    continue
                cyc, j = [], i
                while j not in seen:
                    seen.add(j)
                    cyc.append(j)
                    j = p[j]
                if len({two_i[k] for k in cyc}) != 1:
                    raise SymmetryError("a permutation mixes nuclei of different spin")
                prod *= two_i[cyc[0]] + 1
                if two_i[cyc[0]] % 2 and len(cyc) % 2 == 0:
                    sgn = -sgn
            chi_ns.append(prod)
            sign.append(sgn)
        chi_ns, sign = np.array(chi_ns, float), np.array(sign, float)
        allowed = []
        for nm, c in self.chars.items():
            if round(c[0].real) != 1:
                continue
            ok = True
            for gi, g in enumerate(self.elements):
                if g.parity == 1 and abs(c[gi] - sign[gi]) > tol:
                    ok = False
                if g.parity == -1 and abs(abs(c[gi]) - 1) > tol:
                    ok = False
            if ok:
                allowed.append(nm)
        out = {}
        for nm, c in self.chars.items():
            w = sum(self.decompose(c * chi_ns).get(t, 0) for t in allowed)
            out[nm] = int(w)
        return out

    def character_table(self) -> dict:
        reps = [cl[0] for cl in self.classes]
        return {"group": self.name, "order": self.order,
                "classes": [{"size": len(cl), "representative": int(cl[0])} for cl in self.classes],
                "characters": {nm: [complex(c[r]) if abs(np.imag(c[r])) > 1e-9 else float(np.real(c[r]))
                                    for r in reps] for nm, c in self.chars.items()},
                "frobenius_schur": {nm: self.frobenius_schur(nm) for nm in self.chars}}


def complete_homogeneous(lam, n: int) -> complex:
    """h_n of the values lam (coefficient of t^n in prod 1/(1 - t lam))."""
    if n == 0:
        return 1.0 + 0j
    p = [np.sum(np.asarray(lam) ** k) for k in range(n + 1)]
    h = [1.0 + 0j]
    for k in range(1, n + 1):
        h.append(sum(p[i] * h[k - i] for i in range(1, k + 1)) / k)
    return h[n]


# ---- library -----------------------------------------------------------------------------

def _rz(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


C2X, C2Y, C2Z = np.diag([1.0, -1, -1]), np.diag([-1.0, 1, -1]), np.diag([-1.0, -1, 1])


def _point_perm(O, positions):
    P = np.asarray(positions, float)
    out = []
    for p in P:
        q = np.asarray(O) @ p
        j = int(np.argmin(np.linalg.norm(P - q, axis=1)))
        if np.linalg.norm(P[j] - q) > 1e-6:
            raise SymmetryError("point operation does not map the nuclear framework to itself")
        out.append(j)
    return tuple(out)


def get_group(name: str, *, double: bool = False) -> Group:
    """Library: C1, Cs, C2, C2v, C3v, D2, D3h, Td, and G8(H2O2) (G4 x gauge, half-angle torsion)."""
    if name == "C1":
        return Group("C1", [], double=double, names={"A": (1, [])})
    if name == "Cs":
        return Group("Cs", [make_element(O=np.diag([1.0, 1, -1]))], double=double,
                     names={"A'": (1, [1]), "A''": (1, [-1])})
    if name == "C2":
        return Group("C2", [make_element(O=C2Z)], double=double, names={"A": (1, [1]), "B": (1, [-1])})
    if name == "C2v":
        return Group("C2v", [make_element(O=C2Z), make_element(O=np.diag([1.0, -1, 1]))], double=double,
                     names={"A1": (1, [1, 1]), "A2": (1, [1, -1]), "B1": (1, [-1, 1]), "B2": (1, [-1, -1])})
    if name == "D2":
        return Group("D2", [make_element(C2Z), make_element(C2X)], double=double,
                     names={"A": (1, [1, 1]), "B1": (1, [1, -1]), "B3": (1, [-1, 1]), "B2": (1, [-1, -1]),
                            "E1/2": (2, [0, 0])})
    if name == "C3v":
        c3, sv = _rz(TWO_PI / 3), np.diag([1.0, -1, 1])
        return Group("C3v", [make_element(O=c3), make_element(O=sv)], double=double,
                     names={"A1": (1, [1, 1]), "A2": (1, [1, -1]), "E": (2, [-1, 0])},
                     preferred={"E": [c3[:2, :2], sv[:2, :2]]})
    if name == "D3h":
        c3, c2, sh = _rz(TWO_PI / 3), C2X, np.diag([1.0, 1, -1])
        return Group("D3h", [make_element(O=c3), make_element(O=c2), make_element(O=sh)], double=double,
                     names={"A1'": (1, [1, 1, 1]), "A2'": (1, [1, -1, 1]), "E'": (2, [-1, 0, 2]),
                            "A1''": (1, [1, 1, -1]), "A2''": (1, [1, -1, -1]), "E''": (2, [-1, 0, -2])},
                     preferred={"E'": [c3[:2, :2], c2[:2, :2], sh[:2, :2]]})
    if name == "Td":
        s4 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, -1.0]])
        c3 = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0.0]])
        H = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
        gens = [make_element(O=s4, perm=_point_perm(s4, H)), make_element(O=c3, perm=_point_perm(c3, H))]
        return Group("Td", gens, double=double,
                     names={"A1": (1, [1, 1]), "A2": (1, [-1, 1]), "E": (2, [0, -1]), "F1": (3, [1, 0]),
                            "F2": (3, [-1, 0]), "E1/2": (2, [math.sqrt(2), 1]), "E5/2": (2, [-math.sqrt(2), 1]),
                            "G3/2": (4, [0, -1])},
                     preferred={"F2": [s4, c3], "F1": [-s4, c3]})
    if name in ("G8(H2O2)", "G4(H2O2)"):
        # nuclei (O1, O2, H1, H2); frame z = O1->O2, x = bisector of the OH projections,
        # torsion t = HOOH dihedral in [0, 4pi): the frame turns by pi about z when t -> t + 2pi.
        swap = make_element(C2X, parity=1, perm=(1, 0, 3, 2), coords={"t": (1, 0.0)})
        estar = make_element(C2Y, parity=-1, perm=(0, 1, 2, 3), coords={"t": (-1, 0.0)})
        gauge = make_element(C2Z, parity=1, perm=(0, 1, 2, 3), coords={"t": (1, TWO_PI)}, gauge=True)
        return Group("G8(H2O2)", [swap, estar, gauge], double=double, periods={"t": 2 * TWO_PI},
                     names={"A1": (1, [1, 1, 1]), "A2": (1, [1, -1, 1]), "B1": (1, [-1, 1, 1]),
                            "B2": (1, [-1, -1, 1]), "A1g": (1, [1, 1, -1]), "A2g": (1, [1, -1, -1]),
                            "B1g": (1, [-1, 1, -1]), "B2g": (1, [-1, -1, -1])})
    raise SymmetryError(f"unknown group {name!r}")
