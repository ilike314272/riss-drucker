#pragma once
// Large-amplitude-motion (LAM) models beyond the 1D torsion of M1.
//
// 1) Inversion (NH3-type double well), coordinate x (dimensionless), with an
//    optional symmetric-top rotor and inversion-rotation coupling:
//      H = G p^2 + V2 x^2 + V4 x^4 + V6 x^6
//        + B (J(J+1) - K^2) + C K^2 + Bx2 x^2 (J(J+1) - K^2) + Cx2 x^2 K^2
//    Basis: harmonic-oscillator functions of q = x / basis_scale, n < n_basis.
//    Polynomial operators are built in an enlarged basis and truncated, so
//    their matrix elements are exact. Blocks: (J, K >= 0, inversion parity
//    "+" / "-"); levels with K > 0 are doubly degenerate (+-K).
//
// 2) Rho-axis-method internal rotation in an asymmetric rotor (full K mixing):
//      H = A J_a^2 + B J_b^2 + C J_c^2 + D_ab (J_a J_b + J_b J_a)
//        + F p^2 + D_pa p J_a + sum_L V_L (1 - cos L gamma)/2
//    Basis |J K> |m>, m = sigma mod n_fold, |m| <= m_max. Blocks (J, sigma).
//    With rho along a: F (p - rho J_a)^2 = F p^2 - 2 F rho p J_a + F rho^2 J_a^2,
//    so D_pa = -2 F rho and A absorbs F rho^2 (same gauge as torsion_rotor).
#include <string>
#include <vector>

#include "riss_drucker/model.hpp"

namespace riss_drucker::models {

struct InversionSpec {
  int n_basis = 40;
  double basis_scale = 1.0;  // x = basis_scale * q
  int potential_order = 4;   // 2, 4 or 6: highest even power of x
  int j_max = 0;
  bool rotation_coupling = false;  // Bx2, Cx2 terms (needs j_max > 0)
};
CompiledModel compile_inversion(const InversionSpec& spec);

struct RamTorsionSpec {
  int n_fold = 3;
  std::vector<int> v_orders{1};
  int m_max = 15;
  int j_max = 5;
  std::string axis = "Ir";
  bool d_ab = true;
};
CompiledModel compile_ram_torsion(const RamTorsionSpec& spec);

}  // namespace riss_drucker::models
