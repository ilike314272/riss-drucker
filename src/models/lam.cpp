#include "riss_drucker/models/lam.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <set>
#include <sstream>
#include <stdexcept>

#include <Eigen/Dense>

namespace riss_drucker::models {
namespace {

using Eigen::MatrixXd;

CsrMatrix to_csr(const MatrixXd& m, double drop = 1e-13) {
  std::vector<Triplet> t;
  const double scale = std::max(1.0, m.cwiseAbs().maxCoeff());
  for (Eigen::Index i = 0; i < m.rows(); ++i)
    for (Eigen::Index j = 0; j < m.cols(); ++j)
      if (std::abs(m(i, j)) > drop * scale) t.push_back({i, j, m(i, j)});
  return CsrMatrix::from_triplets(m.rows(), m.cols(), std::move(t));
}

void certify(CompiledModel& model, const char* name, double v, double bound, const char* detail) {
  model.compile_certificates.push_back({name, v <= bound, v, bound, detail});
  if (v > bound) throw std::runtime_error(model.name + " compile certificate '" + name + "' failed");
}

}  // namespace

// ---------------------------------------------------------------------------
CompiledModel compile_inversion(const InversionSpec& s) {
  if (s.n_basis < 4) throw std::invalid_argument("inversion: n_basis must be >= 4");
  if (!(s.basis_scale > 0)) throw std::invalid_argument("inversion: basis_scale must be > 0");
  if (s.potential_order != 2 && s.potential_order != 4 && s.potential_order != 6)
    throw std::invalid_argument("inversion: potential_order must be 2, 4 or 6");
  if (s.rotation_coupling && s.j_max == 0) throw std::invalid_argument("inversion: rotation_coupling needs j_max > 0");

  // HO ladder in an enlarged basis; truncate after forming polynomials.
  const int nb = s.n_basis, big = s.n_basis + 8;
  MatrixXd a = MatrixXd::Zero(big, big);
  for (int n = 1; n < big; ++n) a(n - 1, n) = std::sqrt(static_cast<double>(n));
  const MatrixXd q = (a + a.transpose()) / std::sqrt(2.0);
  const MatrixXd pq2 = -((a.transpose() - a) * (a.transpose() - a)) / 2.0;  // p_q^2 (real, symmetric)
  const double sc = s.basis_scale;
  const MatrixXd x2 = (sc * sc) * (q * q);
  const MatrixXd X2 = x2.topLeftCorner(nb, nb);
  const MatrixXd X4 = (x2 * x2).topLeftCorner(nb, nb);
  const MatrixXd X6 = (x2 * x2 * x2).topLeftCorner(nb, nb);
  const MatrixXd P2 = pq2.topLeftCorner(nb, nb) / (sc * sc);

  CompiledModel model;
  model.name = "inversion";
  {
    std::ostringstream js;
    js << "{\"basis_scale\":" << s.basis_scale << ",\"j_max\":" << s.j_max << ",\"model\":\"inversion\",\"n_basis\":"
       << s.n_basis << ",\"potential_order\":" << s.potential_order
       << ",\"rotation_coupling\":" << (s.rotation_coupling ? "true" : "false") << "}";
    model.spec_json = js.str();
  }
  std::vector<std::string> names{"G", "V2"};
  if (s.potential_order >= 4) names.push_back("V4");
  if (s.potential_order >= 6) names.push_back("V6");
  if (s.j_max > 0) { names.push_back("B"); names.push_back("C"); }
  if (s.rotation_coupling) { names.push_back("Bx2"); names.push_back("Cx2"); }
  model.param_names = names;

  double parity_leak = 0.0;
  for (const MatrixXd* m : {&X2, &X4, &X6, &P2})
    for (int i = 0; i < nb; ++i)
      for (int j = 0; j < nb; ++j)
        if ((i + j) % 2) parity_leak = std::max(parity_leak, std::abs((*m)(i, j)));

  for (int J = 0; J <= s.j_max; ++J)
    for (int K = 0; K <= J; ++K)
      for (int par = 0; par < 2; ++par) {
        std::vector<int> idx;
        for (int n = par; n < nb; n += 2) idx.push_back(n);
        const auto d = static_cast<Eigen::Index>(idx.size());
        auto sub = [&](const MatrixXd& m) {
          MatrixXd r(d, d);
          for (Eigen::Index i = 0; i < d; ++i)
            for (Eigen::Index j = 0; j < d; ++j) r(i, j) = m(idx[i], idx[j]);
          return r;
        };
        const MatrixXd I = MatrixXd::Identity(d, d);
        const double rotB = J * (J + 1.0) - double(K) * K, rotC = double(K) * K;
        CompiledBlock b;
        b.id.two_j = 2 * J;
        b.id.irrep = par == 0 ? "+" : "-";
        b.id.sub = "K=" + std::to_string(K);
        for (int n : idx) b.basis_labels.push_back("n=" + std::to_string(n));
        for (const auto& nm : names) {
          MatrixXd m;
          if (nm == "G") m = sub(P2);
          else if (nm == "V2") m = sub(X2);
          else if (nm == "V4") m = sub(X4);
          else if (nm == "V6") m = sub(X6);
          else if (nm == "B") m = rotB * I;
          else if (nm == "C") m = rotC * I;
          else if (nm == "Bx2") m = rotB * sub(X2);
          else m = rotC * sub(X2);  // Cx2
          b.ops.push_back(to_csr(m));
        }
        model.blocks.push_back(std::move(b));
      }
  certify(model, "parity", parity_leak, 1e-12, "even operators couple only same-parity HO functions");
  double herm = 0;
  for (const MatrixXd* m : {&X2, &X4, &X6, &P2}) herm = std::max(herm, (*m - m->transpose()).cwiseAbs().maxCoeff());
  certify(model, "hermiticity", herm, 1e-9, "max |O - O^T|");
  return model;
}

// ---------------------------------------------------------------------------
CompiledModel compile_ram_torsion(const RamTorsionSpec& s) {
  if (s.n_fold < 1 || s.m_max < 0 || s.j_max < 0) throw std::invalid_argument("ram_torsion: invalid spec");
  int ax_a, ax_b, ax_c;
  if (s.axis == "Ir") { ax_b = 0; ax_c = 1; ax_a = 2; }
  else if (s.axis == "IIIr") { ax_a = 0; ax_b = 1; ax_c = 2; }
  else throw std::invalid_argument("ram_torsion: axis must be Ir or IIIr");
  std::set<int> orders(s.v_orders.begin(), s.v_orders.end());
  for (int k : orders)
    if (k < 1) throw std::invalid_argument("ram_torsion: v_orders must be >= 1");

  CompiledModel model;
  model.name = "ram_torsion";
  {
    std::ostringstream js;
    js << "{\"axis\":\"" << s.axis << "\",\"d_ab\":" << (s.d_ab ? "true" : "false") << ",\"j_max\":" << s.j_max
       << ",\"m_max\":" << s.m_max << ",\"model\":\"ram_torsion\",\"n_fold\":" << s.n_fold << ",\"v_orders\":[";
    bool first = true;
    for (int k : orders) { js << (first ? "" : ",") << k; first = false; }
    js << "]}";
    model.spec_json = js.str();
  }
  std::vector<std::string> names{"A", "B", "C"};
  if (s.d_ab) names.push_back("D_ab");
  names.push_back("F");
  names.push_back("D_pa");
  for (int k : orders) names.push_back("V" + std::to_string(k * s.n_fold));
  model.param_names = names;

  double herm = 0;
  for (int J = 0; J <= s.j_max; ++J) {
    const int nk = 2 * J + 1;
    MatrixXd Jp = MatrixXd::Zero(nk, nk), Jz = MatrixXd::Zero(nk, nk);
    for (int K = -J; K <= J; ++K) {
      Jz(K + J, K + J) = K;
      if (K < J) Jp(K + 1 + J, K + J) = std::sqrt(J * (J + 1.0) - K * (K + 1.0));
    }
    const MatrixXd Jm = Jp.transpose();
    const MatrixXd Jx = 0.5 * (Jp + Jm);
    const MatrixXd Jy2 = -0.25 * (Jp - Jm) * (Jp - Jm);  // J_y^2 (real)
    const MatrixXd comp2[3] = {Jx * Jx, Jy2, Jz * Jz};
    const MatrixXd comp1[3] = {Jx, MatrixXd(), Jz};      // J_y alone is imaginary: not used linearly
    if (ax_a == 1 || (s.d_ab && ax_b == 1))
      throw std::invalid_argument("ram_torsion: the a axis and D_ab must not involve the y axis");
    for (int sigma = 0; sigma < s.n_fold; ++sigma) {
      std::vector<int> ms;
      for (int m = -s.m_max; m <= s.m_max; ++m)
        if (((m - sigma) % s.n_fold + s.n_fold) % s.n_fold == 0) ms.push_back(m);
      if (ms.empty()) continue;
      const int nm = static_cast<int>(ms.size());
      MatrixXd P2 = MatrixXd::Zero(nm, nm), P = MatrixXd::Zero(nm, nm), Im = MatrixXd::Identity(nm, nm);
      for (int i = 0; i < nm; ++i) { P(i, i) = ms[i]; P2(i, i) = double(ms[i]) * ms[i]; }
      auto kronm = [&](const MatrixXd& r, const MatrixXd& t) {
        MatrixXd k(nk * nm, nk * nm);
        for (int i = 0; i < nk; ++i)
          for (int j = 0; j < nk; ++j) k.block(i * nm, j * nm, nm, nm) = r(i, j) * t;
        return k;
      };
      const MatrixXd Ik = MatrixXd::Identity(nk, nk);
      CompiledBlock b;
      b.id.two_j = 2 * J;
      b.id.irrep = "sigma=" + std::to_string(sigma);
      for (int K = -J; K <= J; ++K)
        for (int m : ms) b.basis_labels.push_back("K=" + std::to_string(K) + ",m=" + std::to_string(m));
      for (const auto& name : names) {
        MatrixXd op;
        if (name == "A") op = kronm(comp2[ax_a], Im);
        else if (name == "B") op = kronm(comp2[ax_b], Im);
        else if (name == "C") op = kronm(comp2[ax_c], Im);
        else if (name == "D_ab") op = kronm(comp1[ax_a] * comp1[ax_b] + comp1[ax_b] * comp1[ax_a], Im);
        else if (name == "F") op = kronm(Ik, P2);
        else if (name == "D_pa") op = kronm(comp1[ax_a], P);
        else {  // V_L: (1 - cos L gamma)/2 couples m <-> m +- L
          const int L = std::stoi(name.substr(1));
          MatrixXd V = 0.5 * Im;
          for (int i = 0; i < nm; ++i)
            for (int j = 0; j < nm; ++j)
              if (std::abs(ms[i] - ms[j]) == L) V(i, j) = -0.25;
          op = kronm(Ik, V);
        }
        herm = std::max(herm, (op - op.transpose()).cwiseAbs().maxCoeff());
        b.ops.push_back(to_csr(op));
      }
      model.blocks.push_back(std::move(b));
    }
  }
  certify(model, "hermiticity", herm, 1e-12, "max |O - O^T|");
  return model;
}

}  // namespace riss_drucker::models
