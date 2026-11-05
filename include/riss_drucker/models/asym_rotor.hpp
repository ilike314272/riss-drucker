#pragma once
// M2 asymmetric rotor in a Watson reduction (README 11, M2).
//
//   H = sum_k t_k O_k in the symmetric-top basis |J K>, blocked by J and the
//   four Wang (D2) blocks E+, E-, O+, O- (K even/odd, (|K> +- |-K>)/sqrt 2).
//
// Representation maps (x, y, z) to principal axes:
//   Ir: (b, c, a)   IIr: (c, a, b)   IIIr: (a, b, c)
// Rigid: A J_a^2 + B J_b^2 + C J_c^2.
// A reduction (quartic): -DeltaJ J^4 - DeltaJK J^2 Jz^2 - DeltaK Jz^4
//   - 2 deltaJ J^2 (Jx^2 - Jy^2) - deltaK [Jz^2 (Jx^2 - Jy^2) + (Jx^2 - Jy^2) Jz^2]
// A reduction (sextic): PhiJ J^6 + PhiJK J^4 Jz^2 + PhiKJ J^2 Jz^4 + PhiK Jz^6
//   + 2 phiJ J^4 (Jx^2-Jy^2) + phiJK J^2 [Jz^2, Jx^2-Jy^2]_+ + phiK [Jz^4, Jx^2-Jy^2]_+
// S reduction (quartic): -DJ J^4 - DJK J^2 Jz^2 - DK Jz^4 + d1 J^2 (J+^2 + J-^2) + d2 (J+^4 + J-^4)
// S reduction (sextic): HJ J^6 + HJK J^4 Jz^2 + HKJ J^2 Jz^4 + HK Jz^6
//   + h1 J^4 (J+^2 + J-^2) + h2 J^2 (J+^4 + J-^4) + h3 (J+^6 + J-^6)
// Octic (both reductions, all with + sign like the sextic terms):
//   LJ J^8 + LJJK J^6 Jz^2 + LJK J^4 Jz^4 + LKKJ J^2 Jz^6 + LK Jz^8
//   A: + 2 lJ J^6 (Jx^2-Jy^2) + lJK J^4 [Jz^2, X]_+ + lKJ J^2 [Jz^4, X]_+ + lK [Jz^6, X]_+
//   S: + l1 J^6 (J+^2+J-^2) + l2 J^4 (J+^4+J-^4) + l3 J^2 (J+^6+J-^6) + l4 (J+^8+J-^8)
// Papers that write some terms with the opposite sign map through
// convention.name_map scale = -1 in round-trip validation.
// ([X, Y]_+ = XY + YX.) Units: cm^-1 throughout (docs/conventions.md).
// Extra operators (gauge tests): "J2" = J^2, linearly dependent on A, B, C.
#include <string>
#include <vector>

#include "riss_drucker/model.hpp"

namespace riss_drucker::models {

struct AsymRotorSpec {
  int j_min = 0;
  int j_max = 10;
  std::string reduction = "A";  // "A" | "S"
  std::string axis = "Ir";      // "Ir" | "IIr" | "IIIr"
  int order = 4;                // 2 (rigid), 4 (quartic), 6 (sextic), 8 (octic)
  std::vector<std::string> extra_operators;
};

// Parameter names for a spec, in operator order.
std::vector<std::string> asym_rotor_param_names(const AsymRotorSpec& spec);

// Throws std::invalid_argument on an invalid spec; std::runtime_error if a
// compile certificate fails (the compile is aborted).
CompiledModel compile_asym_rotor(const AsymRotorSpec& spec);

}  // namespace riss_drucker::models
