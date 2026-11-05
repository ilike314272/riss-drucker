"""Any molecule -> a certified effective-Hamiltonian model (README 7).

    mol = rd.molecule.Molecule.from_xyz(text, charge=0, multiplicity=1)
    info = mol.analyse()                  # point group, rotor type, Gamma_vib, spin weights, ...
    plan = mol.plan(frequencies=...)      # BuildSpec + initial parameters (+ symbolic LM prior)
    result = rd.molecule.run(mol, data)   # autobuild with the symbolic language model

* Geometry: centre of mass, principal axes (z = unique axis of a symmetric top, z = a for an
  asymmetric top), rotational constants (cm^-1), rotor type.
* Point group: symmetry operations found by testing candidate axes and planes on the
  nuclear framework, closed under multiplication; Schoenflies name from the classical
  decision tree; the group itself (with nuclear permutations) is built from the detected
  operations, so its frame is the molecule's. Irreps get Mulliken-style names (A/B/E/F/G/H,
  1/2 from the secondary operation, g/u, '/'').
* Gamma_vib = Gamma_3N - Gamma_trans - Gamma_rot from characters (no Hessian needed);
  harmonic frequencies optionally from PySCF (`harmonic_frequencies`), assigned to irreps by
  projecting the normal modes.
* Spin from the multiplicity; a degenerate electronic state (given as an irrep) adds an
  electronic space (Jahn-Teller terms then come from the dictionary).
* Linear molecules are recognised; their rotation needs l-type spaces not built here, so
  `plan` builds a vibrational model for them and says so.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from .symmetry import Group, make_element

MASS = {"H": 1.00782503207, "D": 2.01410177812, "He": 4.00260325413, "Li": 7.016004, "B": 11.0093054,
        "C": 12.0, "N": 14.0030740048, "O": 15.99491461956, "F": 18.99840322, "Ne": 19.9924401754,
        "Na": 22.9897692809, "Si": 27.9769265325, "P": 30.97376163, "S": 31.972071, "Cl": 34.96885268,
        "Br": 78.9183371, "I": 126.904473}
NUCLEAR_SPIN = {"H": 0.5, "D": 1.0, "He": 0.0, "Li": 1.5, "B": 1.5, "C": 0.0, "N": 1.0, "O": 0.0, "F": 0.5,
                "Ne": 0.0, "Na": 1.5, "Si": 0.0, "P": 0.5, "S": 0.0, "Cl": 1.5, "Br": 1.5, "I": 2.5}
HBAR2_2 = 16.857629206   # hbar^2 / (2 amu A^2) in cm^-1


@dataclass
class Molecule:
    symbols: list
    coords: np.ndarray          # Angstrom
    charge: int = 0
    multiplicity: int = 1
    name: str = ""
    electronic_irrep: str | None = None    # degenerate ground state, e.g. "F2" for CH4+
    _cache: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_xyz(cls, text: str, **kw) -> "Molecule":
        lines = [l.split() for l in text.strip().splitlines() if l.strip()]
        if lines and len(lines[0]) == 1 and lines[0][0].isdigit():
            lines = lines[2:] if len(lines) > 1 and len(lines[1]) != 4 else lines[1:]
        sym = [l[0].capitalize() for l in lines]
        xyz = np.array([[float(x) for x in l[1:4]] for l in lines])
        return cls(sym, xyz, **kw)

    # ---- geometry --------------------------------------------------------------------------
    @property
    def masses(self) -> np.ndarray:
        return np.array([MASS[s] for s in self.symbols])

    def principal_frame(self):
        """Coordinates in the principal-axis frame and the moments (amu A^2), ordered so that
        z is the unique axis of a symmetric top / the a axis of an asymmetric top."""
        if "frame" in self._cache:
            return self._cache["frame"]
        m = self.masses
        X = self.coords - (m[:, None] * self.coords).sum(0) / m.sum()
        I = sum(mi * ((x @ x) * np.eye(3) - np.outer(x, x)) for mi, x in zip(m, X))
        w, V = np.linalg.eigh(I)                  # Ia <= Ib <= Ic
        tol = 1e-3 * max(w[-1], 1e-9)
        if abs(w[1] - w[2]) < tol and abs(w[0] - w[1]) >= tol:      # prolate: unique = a
            order = [1, 2, 0]
        elif abs(w[0] - w[1]) < tol and abs(w[1] - w[2]) >= tol:    # oblate: unique = c
            order = [0, 1, 2]
        else:                                                       # asymmetric / spherical: z = a
            order = [1, 2, 0]
        V = V[:, order]
        if np.linalg.det(V) < 0:
            V[:, 0] *= -1
        out = (X @ V, w[order])
        self._cache["frame"] = out
        return out

    def rotational_constants(self) -> dict:
        _, I = self.principal_frame()
        B = [float(HBAR2_2 / x) if x > 1e-8 else float("inf") for x in I]
        return {"x": B[0], "y": B[1], "z": B[2]}

    def rotor_type(self) -> str:
        _, I = self.principal_frame()
        Is = np.sort(I)
        tol = 1e-3 * Is[-1]
        if Is[0] < 1e-6 * Is[-1]:
            return "linear"
        if Is[2] - Is[0] < tol:
            return "spherical"
        if Is[1] - Is[0] < tol:
            return "oblate"
        if Is[2] - Is[1] < tol:
            return "prolate"
        return "asymmetric"

    # ---- point group ---------------------------------------------------------------------------
    def _maps(self, O, X, tol):
        perm = []
        for i, x in enumerate(X):
            y = O @ x
            d = np.linalg.norm(X - y, axis=1)
            j = int(np.argmin(d))
            if d[j] > tol or self.symbols[j] != self.symbols[i]:
                return None
            perm.append(j)
        return tuple(perm) if len(set(perm)) == len(perm) else None

    def symmetry_operations(self, tol: float = 2e-2):
        """All point operations (3x3, principal frame) mapping the framework onto itself."""
        if "ops" in self._cache:
            return self._cache["ops"]
        X, _ = self.principal_frame()
        axes = [np.eye(3)[k] for k in range(3)]
        for x in X:
            if np.linalg.norm(x) > 1e-3:
                axes.append(x)
        for i, j in itertools.combinations(range(len(X)), 2):
            if self.symbols[i] == self.symbols[j]:
                for v in (X[i] + X[j], X[i] - X[j], np.cross(X[i], X[j])):
                    if np.linalg.norm(v) > 1e-3:
                        axes.append(v)
        uniq = []
        for a in axes:
            a = a / np.linalg.norm(a)
            if not any(abs(abs(a @ b) - 1) < 1e-4 for b in uniq):
                uniq.append(a)
        found = {}                                   # (perm, det) -> refined operation

        def refine(perm, d):
            """Orthogonal Procrustes: the O with det d mapping X onto X[perm] best."""
            M = X[list(perm)].T @ X
            U, sv, Vt = np.linalg.svd(M)
            D = np.diag([1.0, 1.0, d * np.linalg.det(U @ Vt)])
            return U @ D @ Vt

        def add(O):
            perm = self._maps(O, X, tol)
            if perm is None:
                return
            d = round(float(np.linalg.det(O)))
            if (perm, d) not in found:
                Oref = refine(perm, d)
                if self._maps(Oref, X, tol) == perm:
                    found[(perm, d)] = Oref
        add(np.eye(3))
        add(-np.eye(3))
        for a in uniq:
            K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
            for n in range(2, 9):
                t = 2 * math.pi / n
                R = np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K
                add(R)
                add(R @ (np.eye(3) - 2 * np.outer(a, a)))     # S_n
            add(np.eye(3) - 2 * np.outer(a, a))               # sigma
        # exact closure on (permutation, determinant)
        changed = True
        while changed:
            changed = False
            for (p1, d1), (p2, d2) in itertools.product(list(found), repeat=2):
                k = (tuple(p1[i] for i in p2), d1 * d2)
                if k not in found:
                    found[k] = refine(*k)
                    changed = True
            if len(found) > 240:
                raise ValueError("too many symmetry operations (linear molecule?)")
        keys = sorted(found, key=lambda k: (k[0] != tuple(range(len(X))) or k[1] != 1, k))
        # symmetrize the framework with the detected group (Reynolds average) and refit: the
        # operations become exact to round-off, as the group algebra requires
        for _ in range(4):
            Xs = sum(found[k].T @ X[list(k[0])].T for k in keys).T / len(keys)
            X = Xs
            for k in keys:
                found[k] = refine(*k)
        self._cache["frame"] = (X, self._cache["frame"][1])
        self._cache["symmetrized_shift_A"] = float(np.max(np.abs(X - self.principal_frame()[0])))
        ops = [found[k] for k in keys]
        self._cache["ops"] = ops
        return ops

    def schoenflies(self) -> str:
        if self.rotor_type() == "linear":
            X, _ = self.principal_frame()
            return "D∞h" if self._maps(-np.eye(3), X, 2e-2) is not None else "C∞v"
        ops = self.symmetry_operations()
        info = [_classify(O) for O in ops]
        inv = any(k == "i" for k, _, _ in info)
        proper = [(n, ax) for k, n, ax in info if k == "C"]
        nmax = max([n for n, _ in proper] + [1])
        high = [ax for n, ax in proper if n >= 3]
        dirs = []
        for ax in high:
            if not any(abs(abs(ax @ b) - 1) < 1e-4 for b in dirs):
                dirs.append(ax)
        if len(dirs) >= 2:
            has5 = any(n == 5 for n, _ in proper)
            has4 = any(n == 4 for n, _ in proper)
            nsig = sum(1 for k, _, _ in info if k == "sigma")
            if has5:
                return "Ih" if inv else "I"
            if has4:
                return "Oh" if inv else "O"
            return "Th" if inv else ("Td" if nsig else "T")
        principal = next((ax for n, ax in proper if n == nmax), np.array([0, 0, 1.0]))
        c2perp = [ax for n, ax in proper if n == 2 and abs(ax @ principal) < 1e-4]
        sig = [ax for k, _, ax in info if k == "sigma"]
        sh = any(abs(abs(ax @ principal) - 1) < 1e-4 for ax in sig)
        sv = [ax for ax in sig if abs(ax @ principal) < 1e-4]
        if nmax == 1:
            return "Cs" if sig else ("Ci" if inv else "C1")
        if c2perp:
            return f"D{nmax}h" if sh else (f"D{nmax}d" if sv else f"D{nmax}")
        if sh:
            return f"C{nmax}h"
        if sv:
            return f"C{nmax}v"
        if any(k == "S" and n == 2 * nmax for k, n, _ in info):
            return f"S{2 * nmax}"
        return f"C{nmax}"

    def group(self, *, double: bool = False) -> Group:
        key = ("group", double)
        if key in self._cache:
            return self._cache[key]
        X, _ = self.principal_frame()
        ops = self.symmetry_operations()
        gens = _minimal_generators(ops)
        els = [make_element(O=O, perm=self._maps(O, X, 2e-2)) for O in gens]
        g = Group(self.schoenflies() if self.rotor_type() != "linear" else "C1", els, double=double)
        _mulliken_names(g, ops)
        self._cache[key] = g
        return g

    # ---- representations -------------------------------------------------------------------------
    def gamma_vib(self) -> dict:
        g = self.group()
        X, _ = self.principal_frame()
        chi3n, chit, chir = [], [], []
        for el in g.elements:
            O = el.R * el.parity
            fixed = sum(1 for i, j in enumerate(el.perm) if i == j)
            chi3n.append(fixed * np.trace(O))
            chit.append(np.trace(O))
            chir.append(np.trace(el.R))
        chi = np.array(chi3n) - np.array(chit) - (0 if self.rotor_type() == "linear" else np.array(chir))
        return g.decompose(chi)

    def spin_weights(self) -> dict:
        return self.group().spin_statistics([NUCLEAR_SPIN.get(s, 0.0) for s in self.symbols])

    def analyse(self) -> dict:
        rot = self.rotational_constants()
        out = {"formula": _formula(self.symbols), "point_group": self.schoenflies(), "rotor": self.rotor_type(),
               "rotational_constants_cm": rot, "charge": self.charge, "multiplicity": self.multiplicity}
        if self.rotor_type() != "linear":
            g = self.group()
            out.update({"group_order": g.order, "irreps": list(g.chars), "gamma_vib": self.gamma_vib(),
                        "spin_weights": self.spin_weights(), "character_table": g.character_table()})
            if self.electronic_irrep:
                out["jahn_teller_active"] = g.jahn_teller_active(self.electronic_irrep)
        return out


    # ---- model plan ------------------------------------------------------------------------------
    def plan(self, *, frequencies: dict | None = None, J: list | None = None, polyad_max: int = 2,
             nmax: int = 2, with_rotation: bool = True):
        """Zeroth-order BuildSpec and initial parameters.

        frequencies: {irrep: [omega, ...]} (cm^-1), e.g. from `harmonic_frequencies`; they set
        the polyad weights (omega / omega_min, rounded) and the harmonic terms' initial values.
        Without them the vibrational space is truncated at nmax quanta and harmonic values are
        left for the data to determine."""
        from .build import BuildSpec, ElecSpace, Mode, RotSpace, SpinSpace, VibSpace
        from . import algebra
        lin = self.rotor_type() == "linear"
        if lin:
            raise ValueError("linear molecules need l-type rotation spaces (not built yet); "
                             "use a vibrational-only plan with with_rotation=False on a bent model")
        g = self.group(double=self.multiplicity % 2 == 0)
        gv = self.gamma_vib()
        modes, omega = [], {}
        k = 0
        for irrep, n in sorted(gv.items(), key=lambda kv: kv[0]):
            fl = sorted((frequencies or {}).get(irrep, []))
            for i in range(n):
                k += 1
                modes.append((str(k), irrep, fl[i] if i < len(fl) else None))
        if frequencies and all(w is not None for _, _, w in modes):
            wmin = min(w for _, _, w in modes)
            wts = [max(1, int(round(w / wmin))) for _, _, w in modes]
            vib = VibSpace([Mode(n, ir, wt) for (n, ir, _), wt in zip(modes, wts)],
                           polyads=list(range(max(polyad_max, max(wts)) + 1)))   # every mode excitable
            omega = {f"w{n}": w for n, _, w in modes}
        else:
            vib = VibSpace([Mode(n, ir, 1) for n, ir, _ in modes], nmax=nmax)
        rot_only = not frequencies and nmax == 0           # ground vibrational state only
        spaces = [] if rot_only else [vib]
        if with_rotation:
            spaces.insert(0, RotSpace())
        if self.multiplicity > 1:
            spaces.append(SpinSpace(self.multiplicity - 1))
            if not with_rotation:
                pass
        if self.electronic_irrep and round(g.chars[self.electronic_irrep][0].real) > 1:
            spaces.append(ElecSpace([("1", self.electronic_irrep)]))
        Js = J if J is not None else ([0.5, 1.5, 2.5] if self.multiplicity % 2 == 0 else [0, 1, 2, 3])
        base = BuildSpec(spaces, g, [], J=Js if with_rotation else [0], name=_formula(self.symbols))
        terms, theta = ([] if rot_only else [f"w{n}: n{n}" for n, _, _ in modes]), dict(omega)
        if with_rotation:
            rot2 = algebra.enumerate_terms(base, [tuple(("rot", 2) if s.kind == "rot" else ("vib", 0)
                                                        if s.kind == "vib" else (s.kind,) for s in
                                                        _runtime_kinds(base))]).terms
            if self.multiplicity > 1:      # spin-free rotational terms only (spin terms come from the data)
                rot2 = [t for t in rot2 if "S" not in t.split(":", 1)[1]]
            rot2 = [t.replace(t.split(":")[0], f"r{i + 1}", 1) for i, t in enumerate(rot2)]
            terms += rot2
            theta.update(_rotor_coefficients(base, rot2, self.rotational_constants()))
        return {"spec": base.with_terms(terms), "theta": theta, "modes": modes, "analysis": self.analyse()}


def _runtime_kinds(spec):
    from .build import runtime_spaces
    return runtime_spaces(spec, spec.group)


def _rotor_coefficients(spec, terms, consts):
    """Initial values of the rotational terms: least squares of the target operator
    A_x Jx^2 + A_y Jy^2 + A_z Jz^2 (principal constants) on the terms' canonical polynomials."""
    from .build import Context, Parser, runtime_spaces
    ctx = Context(spec.group, runtime_spaces(spec, spec.group))
    P = Parser(ctx)
    target = ctx.canon(P.parse_expr(" + ".join(f"{consts[a]:.12g} J{a} J{a}" for a in "xyz")))
    polys = [ctx.canon(P.parse_expr(t.split(":", 1)[1])) for t in terms]
    keys = sorted(set(target) | {w for p in polys for w in p}, key=str)
    Amat = np.array([[p.get(w, 0).real for p in polys] for w in keys])
    b = np.array([target.get(w, 0).real for w in keys])
    c, *_ = np.linalg.lstsq(Amat, b, rcond=None)
    return {t.split(":")[0].strip(): float(x) for t, x in zip(terms, c)}


def run(mol: "Molecule", data=None, *, frequencies=None, proposers=None, max_steps: int = 6, **plan_kw):
    """Plan the model for any molecule, then (with data) grow it with autobuild and the
    symbolic language model; without data, return the zeroth-order levels."""
    from . import autobuild as AB
    from .build import compile_build
    from .symbolic_lm import SymbolicLM
    p = mol.plan(frequencies=frequencies, **plan_kw)
    if data is None:
        bm = compile_build(p["spec"])
        lev = [{"J": b.two_j / 2, "irrep": b.irrep, "polyad": b.polyad, "energies": w.tolist()}
               for b, (w, _) in zip(bm.blocks, bm.eig(p["theta"]))]
        return {"plan": p, "model": bm, "levels": lev}
    lm = SymbolicLM.default()
    res = AB.autobuild(p["spec"], data, p["theta"], proposers or [lm.as_proposer(top=8)], max_steps=max_steps)
    return {"plan": p, "result": res}


def _formula(sym):
    c = {}
    for s in sym:
        c[s] = c.get(s, 0) + 1
    return "".join(f"{k}{v if v > 1 else ''}" for k, v in sorted(c.items(), key=lambda kv: (kv[0] != "C", kv[0])))


def _classify(O):
    d = round(float(np.linalg.det(O)))
    R = d * O
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    t = math.acos(c)
    if t < 1e-6:
        return ("E", 1, np.array([0, 0, 1.0])) if d == 1 else ("i", 2, np.array([0, 0, 1.0]))
    w, v = np.linalg.eig(R)
    ax = np.real(v[:, int(np.argmin(np.abs(w - 1)))])
    ax = ax / np.linalg.norm(ax)
    n = int(round(2 * math.pi / t))
    if d == 1:
        return ("C", n, ax)
    if abs(t - math.pi) < 1e-6:
        return ("sigma", 1, ax)
    return ("S", n, ax)


def _minimal_generators(ops):
    """A small generating set: greedily add operations not yet generated."""
    def closure(gens):
        S = [np.eye(3)]
        frontier = [np.eye(3)]
        while frontier:
            nxt = []
            for A in frontier:
                for G in gens:
                    C = A @ G
                    if not any(np.allclose(C, P, atol=1e-6) for P in S):
                        S.append(C)
                        nxt.append(C)
            frontier = nxt
        return S
    gens = []
    have = [np.eye(3)]
    for O in sorted(ops, key=lambda O: -_classify(O)[1]):
        if not any(np.allclose(O, P, atol=1e-6) for P in have):
            gens.append(O)
            have = closure(gens)
        if len(have) == len(ops):
            break
    return gens


def _mulliken_names(g: Group, ops):
    """Rename the auto-named irreps: A/B (principal C_n symmetric/antisymmetric), E, F, G, H by
    dimension; subscript 1/2 from the secondary operation (C2' perpendicular, else sigma_v;
    1-D: + -> 1, degenerate: + -> 2, matching F1/F2 of Td and T1/T2 of Oh); g/u from the
    inversion; '/'' from sigma_h when there is no inversion."""
    info = [(_classify(el.R * el.parity), el) for el in g.elements]
    proper = [(c[1], c[2], i) for i, (c, el) in enumerate(info) if c[0] == "C"]
    if not proper:
        nmax, prin_i, prin_ax = 1, None, np.array([0, 0, 1.0])
    else:
        # principal: highest order axis; for cubic groups the C2/S4 axis (Mulliken convention)
        nmax = max(n for n, _, _ in proper)
        cubic = len({tuple(np.round(np.abs(ax), 4)) for n, ax, _ in proper if n >= 3}) >= 2
        if cubic:
            cands = [(n, ax, i) for n, ax, i in proper if n == (4 if any(n == 4 for n, _, _ in proper) else 2)]
        else:
            cands = [(n, ax, i) for n, ax, i in proper if n == nmax]
        _, prin_ax, prin_i = cands[0]
    sec = None
    high_axes = [ax for n, ax, _ in proper if n > 2]
    for i, (c, el) in enumerate(info):
        if c[0] == "C" and c[1] == 2 and abs(c[2] @ prin_ax) < 1e-4 and \
                not any(abs(abs(c[2] @ h) - 1) < 1e-4 for h in high_axes):
            sec = i
            break
    if sec is None:
        for i, (c, el) in enumerate(info):
            if c[0] == "sigma" and abs(c[2] @ prin_ax) < 1e-4:
                sec = i
                break
    inv = next((i for i, (c, _) in enumerate(info) if c[0] == "i"), None)
    sh = next((i for i, (c, _) in enumerate(info) if c[0] == "sigma" and abs(abs(c[2] @ prin_ax) - 1) < 1e-4),
              None)
    letters = {1: None, 2: "E", 3: "F", 4: "G", 5: "H"}
    c2s = [(ax, i) for n, ax, i in proper if n == 2]
    cubic_g = len({tuple(np.round(np.abs(ax), 4)) for n, ax, _ in proper if n >= 3}) >= 2
    d2like = nmax == 2 and len(c2s) == 3
    new = {}
    spinors = {nm for nm, chi in g.chars.items() if g.double and g._is_spinor(chi)}
    for nm, chi in g.chars.items():
        d = round(chi[0].real)
        if nm in spinors:              # spinor irreps of the double group keep S<d>_<k> names
            new[nm] = nm
            continue
        if d2like:                                    # D2, D2h: B1/B2/B3 by the C2 axis kept
            sym = [k for k, (ax, i) in enumerate(sorted(c2s, key=lambda t: -abs(t[0][2]) * 4 - abs(t[0][1]) * 2))
                   if chi[i].real > 0]
            base = "A" if len(sym) == 3 else f"B{sym[0] + 1}"
            suf = "" if inv is None else ("g" if chi[inv].real > 0 else "u")
            new[nm] = base + suf
            continue
        if d == 1:
            base = "A" if (prin_i is None or cubic_g or chi[prin_i].real > 0) else "B"
        else:
            base = letters.get(d, f"G{d}")
            if d == 2 and prin_i is not None and nmax >= 5:     # E1, E2, ... from chi(C_n) = 2 cos(2 pi k/n)
                n_pr = _classify(info[prin_i][1].R * info[prin_i][1].parity)[1]
                k = int(round(math.acos(np.clip(chi[prin_i].real / 2, -1, 1)) * n_pr / (2 * math.pi)))
                base = f"E{k}"
        sub = ""
        if base.startswith("E") and len(base) > 1:
            pass
        elif sec is not None and abs(chi[sec]) > 1e-6:
            pos = chi[sec].real > 0
            sub = ("1" if pos else "2") if d == 1 else ("2" if pos else "1")
        suf = ""
        if inv is not None:
            suf = "g" if chi[inv].real > 0 else "u"
        elif sh is not None:
            suf = "'" if chi[sh].real > 0 else "''"
        new[nm] = base + sub + suf
    # drop subscripts that are not needed, disambiguate collisions
    names = list(new.values())
    for nm, v in list(new.items()):
        base = v.rstrip("gu'")
        if base[-1:] in "12":
            stem = base[:-1] + v[len(base):]
            if sum(1 for w in names if w.rstrip("gu'")[:-1] + w[len(w.rstrip("gu'")):] == stem
                   and w.rstrip("gu'")[-1:] in "12") == 1:
                new[nm] = stem
    seen = {}
    for nm in g.chars:
        v = new[nm]
        if v in seen.values():
            k = 2
            while f"{v}_{k}" in seen.values():
                k += 1
            v = f"{v}_{k}"
        seen[nm] = v
    g.irreps = {seen[k]: v for k, v in g.irreps.items()}
    g.chars = {seen[k]: v for k, v in g.chars.items()}
    g.trivial = seen[g.trivial]


def harmonic_frequencies(mol: Molecule, method: str = "hf", basis: str = "6-31g") -> dict:
    """Harmonic frequencies (cm^-1) per irrep from PySCF (RHF/UHF Hessian), assigned by projecting
    the normal modes on the irreps of the detected group."""
    from pyscf import gto, hessian, scf  # noqa: F401
    X, _ = mol.principal_frame()
    pmol = gto.M(atom=[[s, tuple(x)] for s, x in zip(mol.symbols, X)], basis=basis, charge=mol.charge,
                 spin=mol.multiplicity - 1, verbose=0)
    mf = (scf.RHF(pmol) if mol.multiplicity == 1 else scf.UHF(pmol)).run(conv_tol=1e-10)
    H = mf.Hessian().kernel()                          # (natm, natm, 3, 3) Eh/bohr^2
    n = len(mol.symbols)
    Hc = H.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)
    m = np.repeat(mol.masses, 3)
    Hm = Hc / np.sqrt(np.outer(m, m))
    # project translations and rotations
    Xb = X / 0.529177210903
    vecs = []
    for c in range(3):
        v = np.zeros((n, 3))
        v[:, c] = 1
        vecs.append((v * np.sqrt(mol.masses)[:, None]).ravel())
        e = np.zeros(3)
        e[c] = 1
        vecs.append((np.cross(e, Xb) * np.sqrt(mol.masses)[:, None]).ravel())
    Q, s, _ = np.linalg.svd(np.array(vecs).T, full_matrices=False)
    Q = Q[:, s > 1e-8 * s[0]]
    P = np.eye(3 * n) - Q @ Q.T
    w, V = np.linalg.eigh(P @ Hm @ P)
    keep = np.argsort(np.abs(w))[Q.shape[1]:]
    g = mol.group()
    reps = []
    for el in g.elements:
        O = el.R * el.parity
        D = np.zeros((3 * n, 3 * n))
        for i, j in enumerate(el.perm):
            D[3 * j:3 * j + 3, 3 * i:3 * i + 3] = O
        reps.append(D)
    out = {}
    for k in keep:
        v = V[:, k]
        best, bw = None, -1
        for nm, chi in g.chars.items():
            d = round(chi[0].real)
            Pv = sum(np.conj(chi[gi]) * (reps[gi] @ v) for gi in range(g.order)) * d / g.order
            wgt = float(np.real(v @ Pv))
            if wgt > bw:
                best, bw = nm, wgt
        f = math.copysign(math.sqrt(abs(w[k])) * 5140.48714, w[k])
        out.setdefault(best, []).append(f)
    # degenerate modes appear d times: keep one per degenerate set
    res = {}
    for nm, fs in out.items():
        d = round(g.chars[nm][0].real)
        fs = sorted(fs)
        res[nm] = [float(np.mean(fs[i:i + d])) for i in range(0, len(fs), d)]
    return res
