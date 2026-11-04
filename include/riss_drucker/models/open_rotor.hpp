#pragma once
// Open-shell asymmetric rotor: rotation + electron spin S (M8: open-shell spin
// terms; half-integer J for odd electron counts, Kramers / GSE class).
//
// Basis per J: Hund's case (a) product |J P> |S Sigma> (P, Sigma molecule-fixed
// projections), complete for fixed J (dimension (2J+1)(2S+1) = sum_N (2N+1),
// N = |J-S| .. J+S). Operators are written exactly in molecule-fixed components:
// J has anomalous commutation ([Jx, Jy] = -i Jz), S normal, N = J - S
// (anomalous). Represented with normal matrices as J_mf = (Jx, -Jy, Jz).
//
//   H = A N_a^2 + B N_b^2 + C N_c^2                         rigid rotor in N
//     + eps_aa N_a S_a + eps_bb N_b S_b + eps_cc N_c S_c     spin-rotation (diagonal tensor)
//     + D_ss (S_z^2 - S^2/3) + E_ss (S_x^2 - S_y^2)          spin-spin, S >= 1 (representation axes)
//     + Watson A quartic distortion in N (order 4):
//       -DeltaN N^4 - DeltaNK N^2 N_z^2 - DeltaK N_z^4 - 2 deltaN N^2 (N_x^2 - N_y^2)
//       - deltaK [N_z^2, N_x^2 - N_y^2]_+
//     + DeltaN_S N^2 (N.S)  (spin-rotation distortion, Brown-Sears; = gamma_D in the linear limit)
//     + D_ss_N [N^2, S_z^2 - S^2/3]_+ / 2  (spin-spin distortion; = 2 lambda_D in the linear limit)
// Linear Sigma molecule limit: B = C, A large (pushes K != 0 away), eps_bb = eps_cc = gamma,
// D_ss = 2 lambda (since (2/3) lambda (3 S_z^2 - S^2) = 2 lambda (S_z^2 - S^2/3)).
// Representation (x, y, z) -> principal axes: Ir (b,c,a), IIr (c,a,b), IIIr (a,b,c).
// Units cm^-1. Blocks: one per J (two_j = 2J), irrep "J", basis labels "2P=..,2Sig=..".
#include <string>
#include <vector>

#include "riss_drucker/model.hpp"

namespace riss_drucker::models {

struct OpenRotorSpec {
  int two_s = 1;            // 2S: 1 doublet, 2 triplet, ...
  int n_max = 10;           // all J with J <= n_max + S are built
  std::string axis = "Ir";
  int order = 4;            // 2 or 4 (distortion of N)
  bool spin_rotation = true;
  bool spin_spin = false;   // requires S >= 1
};

CompiledModel compile_open_rotor(const OpenRotorSpec& spec);

}  // namespace riss_drucker::models
