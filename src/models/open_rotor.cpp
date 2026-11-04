#include "riss_drucker/models/open_rotor.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
#include <sstream>
#include <stdexcept>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "riss_drucker/angular.hpp"

namespace riss_drucker::models {
namespace {

using CMat = Eigen::MatrixXcd;
using Eigen::MatrixXd;
using cd = std::complex<double>;

// Normal angular momentum matrices for spin j = two_j / 2, basis m = -j..j.
struct AM {
  CMat x, y, z;
};
AM angmom(int two_j) {
  const int n = two_j + 1;
  const double j = two_j / 2.0;
  CMat jp = CMat::Zero(n, n), jz = CMat::Zero(n, n);
  for (int i = 0; i < n; ++i) {
    const double m = -j + i;
    jz(i, i) = m;
    if (i + 1 < n) jp(i + 1, i) = std::sqrt(j * (j + 1) - m * (m + 1));
  }
  const CMat jm = jp.adjoint();
  return {0.5 * (jp + jm), cd(0, -0.5) * (jp - jm), jz};
}

CMat kron(const CMat& a, const CMat& b) {
  CMat k(a.rows() * b.rows(), a.cols() * b.cols());
  for (Eigen::Index i = 0; i < a.rows(); ++i)
    for (Eigen::Index j = 0; j < a.cols(); ++j) k.block(i * b.rows(), j * b.cols(), b.rows(), b.cols()) = a(i, j) * b;
  return k;
}

struct Ops {  // molecule-fixed components on |J P> x |S Sigma>
  CMat N[3], S[3], N2, S2;
};

Ops make_ops(int two_j, int two_s) {
  const AM J = angmom(two_j), Sm = angmom(two_s);
  const CMat IJ = CMat::Identity(two_j + 1, two_j + 1), IS = CMat::Identity(two_s + 1, two_s + 1);
  const CMat Jmf[3] = {J.x, -J.y, J.z};  // anomalous molecule-fixed components
  const CMat Smf[3] = {Sm.x, Sm.y, Sm.z};
  Ops o;
  for (int a = 0; a < 3; ++a) {
    o.S[a] = kron(IJ, Smf[a]);
    o.N[a] = kron(Jmf[a], IS) - o.S[a];
  }
  o.N2 = o.N[0] * o.N[0] + o.N[1] * o.N[1] + o.N[2] * o.N[2];
  o.S2 = o.S[0] * o.S[0] + o.S[1] * o.S[1] + o.S[2] * o.S[2];
  return o;
}

}  // namespace

CompiledModel compile_open_rotor(const OpenRotorSpec& s) {
  if (s.two_s < 1) throw std::invalid_argument("open_rotor: two_s must be >= 1 (use asym_rotor for S = 0)");
  if (s.n_max < 0) throw std::invalid_argument("open_rotor: n_max must be >= 0");
  if (s.order != 2 && s.order != 4) throw std::invalid_argument("open_rotor: order must be 2 or 4");
  if (s.spin_spin && s.two_s < 2) throw std::invalid_argument("open_rotor: spin-spin needs S >= 1");
  int ax_a, ax_b, ax_c;  // representation index (0 x, 1 y, 2 z) of the a, b, c axes
  if (s.axis == "Ir") { ax_b = 0; ax_c = 1; ax_a = 2; }
  else if (s.axis == "IIr") { ax_c = 0; ax_a = 1; ax_b = 2; }
  else if (s.axis == "IIIr") { ax_a = 0; ax_b = 1; ax_c = 2; }
  else throw std::invalid_argument("open_rotor: axis must be Ir, IIr or IIIr");

  using Builder = std::function<CMat(const Ops&)>;
  std::vector<std::pair<std::string, Builder>> table;
  table.push_back({"A", [=](const Ops& o) -> CMat { return o.N[ax_a] * o.N[ax_a]; }});
  table.push_back({"B", [=](const Ops& o) -> CMat { return o.N[ax_b] * o.N[ax_b]; }});
  table.push_back({"C", [=](const Ops& o) -> CMat { return o.N[ax_c] * o.N[ax_c]; }});
  if (s.order >= 4) {
    auto X = [](const Ops& o) -> CMat { return o.N[0] * o.N[0] - o.N[1] * o.N[1]; };
    auto Nz2 = [](const Ops& o) -> CMat { return o.N[2] * o.N[2]; };
    table.push_back({"DeltaN", [](const Ops& o) -> CMat { return -(o.N2 * o.N2); }});
    table.push_back({"DeltaNK", [=](const Ops& o) -> CMat { return -0.5 * (o.N2 * Nz2(o) + Nz2(o) * o.N2); }});
    table.push_back({"DeltaK", [=](const Ops& o) -> CMat { return -(Nz2(o) * Nz2(o)); }});
    table.push_back({"deltaN", [=](const Ops& o) -> CMat { return -(o.N2 * X(o) + X(o) * o.N2); }});
    table.push_back({"deltaK", [=](const Ops& o) -> CMat { return -(Nz2(o) * X(o) + X(o) * Nz2(o)); }});
  }
  if (s.spin_rotation) {
    const int axes[3] = {ax_a, ax_b, ax_c};
    const char* names[3] = {"eps_aa", "eps_bb", "eps_cc"};
    for (int k = 0; k < 3; ++k) {
      const int a = axes[k];
      table.push_back({names[k], [a](const Ops& o) -> CMat { return 0.5 * (o.N[a] * o.S[a] + o.S[a] * o.N[a]); }});
    }
  }
  auto NdotS = [](const Ops& o) -> CMat {
    CMat r = CMat::Zero(o.N2.rows(), o.N2.cols());
    for (int a = 0; a < 3; ++a) r += 0.5 * (o.N[a] * o.S[a] + o.S[a] * o.N[a]);
    return r;
  };
  if (s.spin_rotation && s.order >= 4)  // Brown-Sears DeltaN^S: N^2 (N.S), symmetrized
    table.push_back({"DeltaN_S", [=](const Ops& o) -> CMat { return 0.5 * (o.N2 * NdotS(o) + NdotS(o) * o.N2); }});
  if (s.spin_spin) {
    table.push_back({"D_ss", [](const Ops& o) -> CMat { return o.S[2] * o.S[2] - o.S2 / 3.0; }});
    table.push_back({"E_ss", [](const Ops& o) -> CMat { return o.S[0] * o.S[0] - o.S[1] * o.S[1]; }});
    if (s.order >= 4)  // centrifugal distortion of the axial spin-spin term
      table.push_back({"D_ss_N", [](const Ops& o) -> CMat {
        const CMat d = o.S[2] * o.S[2] - o.S2 / 3.0;
        return 0.5 * (o.N2 * d + d * o.N2);
      }});
  }

  CompiledModel model;
  model.name = "open_rotor";
  {
    std::ostringstream js;
    js << "{\"axis\":\"" << s.axis << "\",\"model\":\"open_rotor\",\"n_max\":" << s.n_max << ",\"order\":" << s.order
       << ",\"spin_rotation\":" << (s.spin_rotation ? "true" : "false")
       << ",\"spin_spin\":" << (s.spin_spin ? "true" : "false") << ",\"two_s\":" << s.two_s << "}";
    model.spec_json = js.str();
  }
  for (const auto& t : table) model.param_names.push_back(t.first);

  double herm = 0, imag = 0, casimir = 0, sixj = 0;
  const double S = s.two_s / 2.0;
  for (int two_j = s.two_s % 2; two_j <= 2 * s.n_max + s.two_s; two_j += 2) {
    const Ops o = make_ops(two_j, s.two_s);
    const auto n = o.N2.rows();
    const double J = two_j / 2.0;
    CompiledBlock b;
    b.id.two_j = two_j;
    b.id.irrep = "J";
    for (int tp = -two_j; tp <= two_j; tp += 2)
      for (int ts = -s.two_s; ts <= s.two_s; ts += 2)
        b.basis_labels.push_back("2P=" + std::to_string(tp) + ",2Sig=" + std::to_string(ts));
    for (const auto& t : table) {
      const CMat m = t.second(o);
      herm = std::max(herm, (m - m.adjoint()).cwiseAbs().maxCoeff());
      imag = std::max(imag, m.imag().cwiseAbs().maxCoeff());
      const MatrixXd r = 0.5 * (m.real() + m.real().transpose());
      std::vector<Triplet> trip;
      for (Eigen::Index i = 0; i < n; ++i)
        for (Eigen::Index j = 0; j < n; ++j)
          if (std::abs(r(i, j)) > 1e-12) trip.push_back({i, j, r(i, j)});
      b.ops.push_back(CsrMatrix::from_triplets(n, n, std::move(trip)));
    }
    // Casimir certificate: N^2 has eigenvalues N(N+1), N = |J-S|..J+S, multiplicity 2N+1.
    Eigen::SelfAdjointEigenSolver<MatrixXd> es(o.N2.real());
    std::vector<double> expect;
    for (double N = std::abs(J - S); N <= J + S + 1e-9; N += 1.0)
      for (int k = 0; k < static_cast<int>(2 * N + 1.5); ++k) expect.push_back(N * (N + 1));
    std::sort(expect.begin(), expect.end());
    for (Eigen::Index i = 0; i < n; ++i) casimir = std::max(casimir, std::abs(es.eigenvalues()[i] - expect[i]));
    // Angular-algebra certificate: eigenvalues of N.S equal the 6j-based value
    // <N S J|N.S|N S J> = (-1)^{N+S+J} {N S J; S N 1} sqrt(N(N+1)(2N+1) S(S+1)(2S+1)),
    // which must equal (J(J+1) - N(N+1) - S(S+1))/2.
    CMat ns = CMat::Zero(n, n);
    for (int a = 0; a < 3; ++a) ns += 0.5 * (o.N[a] * o.S[a] + o.S[a] * o.N[a]);
    Eigen::SelfAdjointEigenSolver<MatrixXd> en(ns.real());
    std::vector<double> want;
    for (double N = std::abs(J - S); N <= J + S + 1e-9; N += 1.0) {
      const int tN = static_cast<int>(std::lround(2 * N));
      const int ph = static_cast<int>(std::lround(N + S + J));
      const double v = ((ph % 2) ? -1.0 : 1.0) *
                       angular::wigner6j(tN, s.two_s, two_j, s.two_s, tN, 2) *
                       std::sqrt(N * (N + 1) * (2 * N + 1) * S * (S + 1) * (2 * S + 1));
      sixj = std::max(sixj, std::abs(v - 0.5 * (J * (J + 1) - N * (N + 1) - S * (S + 1))));
      for (int k = 0; k < static_cast<int>(2 * N + 1.5); ++k) want.push_back(v);
    }
    std::sort(want.begin(), want.end());
    for (Eigen::Index i = 0; i < n; ++i) sixj = std::max(sixj, std::abs(en.eigenvalues()[i] - want[i]));
    model.blocks.push_back(std::move(b));
  }
  const double scale = 1.0 + std::pow((s.n_max + S + 1) * (s.n_max + S + 2), 2);
  auto require = [&](const char* name, double v, double bound, const char* detail) {
    model.compile_certificates.push_back({name, v <= bound, v, bound, detail});
    if (v > bound) throw std::runtime_error(std::string("open_rotor compile certificate '") + name + "' failed");
  };
  require("hermiticity", herm, 1e-11 * scale, "max |O - O^dagger|");
  require("reality", imag, 1e-11 * scale, "operators real in this basis (time-reversal-even Hamiltonian)");
  require("casimir_N2", casimir, 1e-10 * scale, "N^2 eigenvalues N(N+1) with multiplicity 2N+1");
  require("sixj_spin_rotation", sixj, 1e-10 * scale, "N.S eigenvalues vs 6j formula vs (J(J+1)-N(N+1)-S(S+1))/2");
  return model;
}

}  // namespace riss_drucker::models
