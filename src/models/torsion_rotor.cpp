#include "riss_drucker/models/torsion_rotor.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace riss_drucker::models {
namespace {

// A basis function: real combination of free-rotor states |m>.
using Component = std::pair<int, double>;  // (m, coefficient)
using BasisFn = std::vector<Component>;

struct Term {  // operator acting on |m>: returns (m', value) pairs
  enum Kind { P2, VL, KSQ, ROTB, KP } kind;
  int L = 0;
};

void apply(const Term& t, int m, int J, int K, int m_max, std::vector<Component>& out) {
  out.clear();
  switch (t.kind) {
    case Term::P2: out.push_back({m, double(m) * m}); break;
    case Term::VL:  // (1 - cos L phi) / 2
      out.push_back({m, 0.5});
      if (std::abs(m + t.L) <= m_max) out.push_back({m + t.L, -0.25});
      if (std::abs(m - t.L) <= m_max) out.push_back({m - t.L, -0.25});
      break;
    case Term::KSQ: out.push_back({m, double(K) * K}); break;
    case Term::ROTB: out.push_back({m, double(J) * (J + 1) - double(K) * K}); break;
    case Term::KP: out.push_back({m, double(K) * m}); break;
  }
}

CsrMatrix build_op(const Term& t, const std::vector<BasisFn>& basis, int J, int K, int m_max) {
  std::map<int, std::pair<std::int64_t, double>> where;  // m -> (basis index, coefficient)
  for (std::size_t i = 0; i < basis.size(); ++i)
    for (const auto& [m, c] : basis[i]) where[m] = {static_cast<std::int64_t>(i), c};
  std::vector<Triplet> trip;
  std::vector<Component> img;
  for (std::size_t j = 0; j < basis.size(); ++j)
    for (const auto& [m, cj] : basis[j]) {
      apply(t, m, J, K, m_max, img);
      for (const auto& [mp, v] : img) {
        const auto it = where.find(mp);
        if (it == where.end()) continue;  // outside this block (cannot happen by symmetry) or truncated
        trip.push_back({it->second.first, static_cast<std::int64_t>(j), it->second.second * cj * v});
      }
    }
  const auto n = static_cast<std::int64_t>(basis.size());
  CsrMatrix op = CsrMatrix::from_triplets(n, n, std::move(trip));
  // Drop rounding-level entries from parity combinations (e.g. 1/2 - 1/2).
  std::vector<Triplet> clean;
  for (std::int64_t r = 0; r < n; ++r)
    for (auto k = op.row_ptr[r]; k < op.row_ptr[r + 1]; ++k)
      if (std::abs(op.values[k]) > 1e-15) clean.push_back({r, op.col_idx[k], op.values[k]});
  return CsrMatrix::from_triplets(n, n, std::move(clean));
}

std::string label(const BasisFn& f, bool parity, int sign) {
  const int m = f.front().first;
  if (!parity || f.size() == 1) return "m=" + std::to_string(m);
  return "m=+-" + std::to_string(std::abs(m)) + (sign > 0 ? ",cos" : ",sin");
}

}  // namespace

CompiledModel compile_torsion_rotor(const TorsionRotorSpec& spec) {
  if (spec.n_fold < 1) throw std::invalid_argument("torsion_rotor: n_fold must be >= 1");
  if (spec.m_max < 0) throw std::invalid_argument("torsion_rotor: m_max must be >= 0");
  if (spec.j_max < 0) throw std::invalid_argument("torsion_rotor: j_max must be >= 0");
  if (spec.kp_coupling && spec.j_max == 0)
    throw std::invalid_argument("torsion_rotor: kp_coupling needs j_max > 0");
  std::set<int> orders(spec.v_orders.begin(), spec.v_orders.end());
  for (int k : orders)
    if (k < 1) throw std::invalid_argument("torsion_rotor: v_orders must be >= 1");

  CompiledModel model;
  model.name = "torsion_rotor";
  {
    std::ostringstream js;  // canonical: sorted keys, sorted unique orders
    js << "{\"j_max\":" << spec.j_max << ",\"kp_coupling\":" << (spec.kp_coupling ? "true" : "false")
       << ",\"m_max\":" << spec.m_max << ",\"model\":\"torsion_rotor\",\"n_fold\":" << spec.n_fold
       << ",\"v_orders\":[";
    bool first = true;
    for (int k : orders) { js << (first ? "" : ",") << k; first = false; }
    js << "]}";
    model.spec_json = js.str();
  }

  std::vector<Term> terms{{Term::P2, 0}};
  model.param_names.push_back("F");
  for (int k : orders) {
    terms.push_back({Term::VL, k * spec.n_fold});
    model.param_names.push_back("V" + std::to_string(k * spec.n_fold));
  }
  if (spec.j_max > 0) {
    terms.push_back({Term::KSQ, 0});
    model.param_names.push_back("A");
    terms.push_back({Term::ROTB, 0});
    model.param_names.push_back("B");
  }
  if (spec.kp_coupling) {
    terms.push_back({Term::KP, 0});
    model.param_names.push_back("D_Kp");
  }

  const int n = spec.n_fold;
  const double s = 1.0 / std::sqrt(2.0);
  double max_defect = 0.0;
  for (int J = 0; J <= spec.j_max; ++J)
    for (int K = -J; K <= J; ++K)
      for (int sigma = 0; sigma < n; ++sigma) {
        std::vector<int> ms;
        for (int m = -spec.m_max; m <= spec.m_max; ++m)
          if (((m - sigma) % n + n) % n == 0) ms.push_back(m);
        if (ms.empty()) continue;
        const bool self_conj = (2 * sigma) % n == 0;
        const bool kp_active = spec.kp_coupling && K != 0;
        const bool parity = self_conj && !kp_active;

        std::vector<std::pair<std::string, std::vector<BasisFn>>> parts;
        if (parity) {
          std::vector<BasisFn> plus, minus;
          for (int m : ms) {
            if (m == 0) plus.push_back({{0, 1.0}});
            else if (m > 0) {
              plus.push_back({{m, s}, {-m, s}});
              minus.push_back({{m, s}, {-m, -s}});
            }
          }
          parts.push_back({"sigma=" + std::to_string(sigma) + ",+", std::move(plus)});
          parts.push_back({"sigma=" + std::to_string(sigma) + ",-", std::move(minus)});
        } else {
          std::vector<BasisFn> plain;
          for (int m : ms) plain.push_back({{m, 1.0}});
          parts.push_back({"sigma=" + std::to_string(sigma), std::move(plain)});
        }

        for (auto& [irrep, basis] : parts) {
          if (basis.empty()) continue;
          CompiledBlock b;
          b.id.two_j = 2 * J;
          b.id.irrep = irrep;
          b.id.sub = "K=" + std::to_string(K);
          const int sign = irrep.back() == '-' ? -1 : 1;
          for (const auto& f : basis) b.basis_labels.push_back(label(f, parity, sign));
          for (const auto& t : terms) {
            CsrMatrix op = build_op(t, basis, J, K, spec.m_max);
            const double d = op.symmetry_defect();
            max_defect = std::max(max_defect, d);
            if (d > 1e-14)
              throw std::runtime_error("torsion_rotor compile certificate failed: operator " +
                                       model.param_names[&t - terms.data()] + " in block " +
                                       b.id.key() + " is not Hermitian (defect " + std::to_string(d) + ")");
            b.ops.push_back(std::move(op));
          }
          model.blocks.push_back(std::move(b));
        }
      }
  model.compile_certificates.push_back(
      {"hermiticity", true, max_defect, 1e-14, "max |O_ij - O_ji| over all operators and blocks"});
  return model;
}

}  // namespace riss_drucker::models
