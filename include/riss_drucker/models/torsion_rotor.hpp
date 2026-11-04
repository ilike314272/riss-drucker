#pragma once
// M1 exact reduced baseline: one-dimensional Fourier torsion plus a rigid
// symmetric-top rotor, optionally coupled through K p (README section 11, M1).
//
//   H = F p^2 + sum_L V_L (1 - cos L phi)/2 + A K^2 + B (J(J+1) - K^2) + D_Kp K p
//
// Basis: e^{i m phi}/sqrt(2 pi), |m| <= m_max, times |J K> (K is a good quantum
// number in this reduced model). Potential terms are L = k * n_fold for k in
// v_orders. Blocks: (J, K, sigma = m mod n_fold), and additionally parity
// (cos/sin combinations) when 2 sigma = 0 mod n_fold and the block has no K p
// term (K = 0 or no coupling). Units: cm^-1 (docs/conventions.md).
// Gauge note: with coupling, F (p - rho K)^2 expands to F p^2 - 2 F rho K p +
// F rho^2 K^2, so D_Kp = -2 F rho and A absorbs F rho^2.
#include <vector>

#include "riss_drucker/model.hpp"

namespace riss_drucker::models {

struct TorsionRotorSpec {
  int n_fold = 3;
  std::vector<int> v_orders{1};
  int m_max = 30;
  int j_max = 0;
  bool kp_coupling = false;
};

// Throws std::invalid_argument on an invalid spec, std::runtime_error if a
// compile certificate (Hermiticity) fails.
CompiledModel compile_torsion_rotor(const TorsionRotorSpec& spec);

}  // namespace riss_drucker::models
