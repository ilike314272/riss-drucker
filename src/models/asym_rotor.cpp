#include "riss_drucker/models/asym_rotor.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <sstream>
#include <stdexcept>

#include <Eigen/Dense>

#include "riss_drucker/angular.hpp"

namespace riss_drucker::models {
namespace {

using Eigen::MatrixXd;

struct JOps {  // dense operators on |J K>, K = -J..J (index K + J)
  int J;
  double j2;
  MatrixXd Jz2, P2, P4, P6, P8, Jx2, Jy2, X;  // X = Jx^2 - Jy^2 = P2 / 2
};

JOps make_jops(int J) {
  const int n = 2 * J + 1;
  const double j2 = J * (J + 1.0);
  MatrixXd Jp = MatrixXd::Zero(n, n);  // <K+1|J+|K>
  for (int K = -J; K < J; ++K) Jp(K + 1 + J, K + J) = std::sqrt(j2 - K * (K + 1.0));
  const MatrixXd Jm = Jp.transpose();
  JOps o;
  o.J = J;
  o.j2 = j2;
  o.Jz2 = MatrixXd::Zero(n, n);
  for (int K = -J; K <= J; ++K) o.Jz2(K + J, K + J) = double(K) * K;
  const MatrixXd Jp2 = Jp * Jp, Jm2 = Jm * Jm;
  o.P2 = Jp2 + Jm2;
  o.P4 = Jp2 * Jp2 + Jm2 * Jm2;
  o.P6 = Jp2 * Jp2 * Jp2 + Jm2 * Jm2 * Jm2;
  o.P8 = Jp2 * Jp2 * Jp2 * Jp2 + Jm2 * Jm2 * Jm2 * Jm2;
  const MatrixXd I = MatrixXd::Identity(n, n);
  o.X = 0.5 * o.P2;
  o.Jx2 = 0.5 * (j2 * I - o.Jz2 + o.X);
  o.Jy2 = 0.5 * (j2 * I - o.Jz2 - o.X);
  return o;
}

struct OpDef {
  std::string name;
  int degree;  // total degree in J components (time reversal: (-1)^degree)
  std::function<MatrixXd(const JOps&)> build;
};

MatrixXd anti(const MatrixXd& a, const MatrixXd& b) { return a * b + b * a; }

std::vector<OpDef> operator_table(const AsymRotorSpec& s) {
  // Rigid part: which representation axis carries a, b, c.
  // axis index 0 = x, 1 = y, 2 = z.
  int ax_a, ax_b, ax_c;
  if (s.axis == "Ir") { ax_b = 0; ax_c = 1; ax_a = 2; }
  else if (s.axis == "IIr") { ax_c = 0; ax_a = 1; ax_b = 2; }
  else if (s.axis == "IIIr") { ax_a = 0; ax_b = 1; ax_c = 2; }
  else throw std::invalid_argument("asym_rotor: axis must be Ir, IIr or IIIr");
  auto sq = [](int ax) {
    return [ax](const JOps& o) -> MatrixXd { return ax == 0 ? o.Jx2 : ax == 1 ? o.Jy2 : o.Jz2; };
  };
  std::vector<OpDef> t{{"A", 2, sq(ax_a)}, {"B", 2, sq(ax_b)}, {"C", 2, sq(ax_c)}};
  auto I = [](const JOps& o) { return MatrixXd::Identity(2 * o.J + 1, 2 * o.J + 1); };
  if (s.order >= 4) {
    if (s.reduction == "A") {
      t.push_back({"DeltaJ", 4, [I](const JOps& o) -> MatrixXd { return -o.j2 * o.j2 * I(o); }});
      t.push_back({"DeltaJK", 4, [](const JOps& o) -> MatrixXd { return -o.j2 * o.Jz2; }});
      t.push_back({"DeltaK", 4, [](const JOps& o) -> MatrixXd { return -o.Jz2 * o.Jz2; }});
      t.push_back({"deltaJ", 4, [](const JOps& o) -> MatrixXd { return -2.0 * o.j2 * o.X; }});
      t.push_back({"deltaK", 4, [](const JOps& o) -> MatrixXd { return -anti(o.Jz2, o.X); }});
    } else {
      t.push_back({"DJ", 4, [I](const JOps& o) -> MatrixXd { return -o.j2 * o.j2 * I(o); }});
      t.push_back({"DJK", 4, [](const JOps& o) -> MatrixXd { return -o.j2 * o.Jz2; }});
      t.push_back({"DK", 4, [](const JOps& o) -> MatrixXd { return -o.Jz2 * o.Jz2; }});
      t.push_back({"d1", 4, [](const JOps& o) -> MatrixXd { return o.j2 * o.P2; }});
      t.push_back({"d2", 4, [](const JOps& o) -> MatrixXd { return o.P4; }});
    }
  }
  if (s.order >= 6) {
    const auto common = std::vector<OpDef>{
        {s.reduction == "A" ? "PhiJ" : "HJ", 6, [I](const JOps& o) -> MatrixXd { return o.j2 * o.j2 * o.j2 * I(o); }},
        {s.reduction == "A" ? "PhiJK" : "HJK", 6, [](const JOps& o) -> MatrixXd { return o.j2 * o.j2 * o.Jz2; }},
        {s.reduction == "A" ? "PhiKJ" : "HKJ", 6, [](const JOps& o) -> MatrixXd { return o.j2 * o.Jz2 * o.Jz2; }},
        {s.reduction == "A" ? "PhiK" : "HK", 6, [](const JOps& o) -> MatrixXd { return o.Jz2 * o.Jz2 * o.Jz2; }}};
    t.insert(t.end(), common.begin(), common.end());
    if (s.reduction == "A") {
      t.push_back({"phiJ", 6, [](const JOps& o) -> MatrixXd { return 2.0 * o.j2 * o.j2 * o.X; }});
      t.push_back({"phiJK", 6, [](const JOps& o) -> MatrixXd { return o.j2 * anti(o.Jz2, o.X); }});
      t.push_back({"phiK", 6, [](const JOps& o) -> MatrixXd { return anti(o.Jz2 * o.Jz2, o.X); }});
    } else {
      t.push_back({"h1", 6, [](const JOps& o) -> MatrixXd { return o.j2 * o.j2 * o.P2; }});
      t.push_back({"h2", 6, [](const JOps& o) -> MatrixXd { return o.j2 * o.P4; }});
      t.push_back({"h3", 6, [](const JOps& o) -> MatrixXd { return o.P6; }});
    }
  }
  if (s.order >= 8) {
    const bool A = s.reduction == "A";
    t.push_back({"LJ", 8, [I](const JOps& o) -> MatrixXd { return std::pow(o.j2, 4) * I(o); }});
    t.push_back({"LJJK", 8, [](const JOps& o) -> MatrixXd { return std::pow(o.j2, 3) * o.Jz2; }});
    t.push_back({"LJK", 8, [](const JOps& o) -> MatrixXd { return o.j2 * o.j2 * o.Jz2 * o.Jz2; }});
    t.push_back({"LKKJ", 8, [](const JOps& o) -> MatrixXd { return o.j2 * o.Jz2 * o.Jz2 * o.Jz2; }});
    t.push_back({"LK", 8, [](const JOps& o) -> MatrixXd { return o.Jz2 * o.Jz2 * o.Jz2 * o.Jz2; }});
    if (A) {
      t.push_back({"lJ", 8, [](const JOps& o) -> MatrixXd { return 2.0 * std::pow(o.j2, 3) * o.X; }});
      t.push_back({"lJK", 8, [](const JOps& o) -> MatrixXd { return o.j2 * o.j2 * anti(o.Jz2, o.X); }});
      t.push_back({"lKJ", 8, [](const JOps& o) -> MatrixXd { return o.j2 * anti(o.Jz2 * o.Jz2, o.X); }});
      t.push_back({"lK", 8, [](const JOps& o) -> MatrixXd { return anti(o.Jz2 * o.Jz2 * o.Jz2, o.X); }});
    } else {
      t.push_back({"l1", 8, [](const JOps& o) -> MatrixXd { return std::pow(o.j2, 3) * o.P2; }});
      t.push_back({"l2", 8, [](const JOps& o) -> MatrixXd { return o.j2 * o.j2 * o.P4; }});
      t.push_back({"l3", 8, [](const JOps& o) -> MatrixXd { return o.j2 * o.P6; }});
      t.push_back({"l4", 8, [](const JOps& o) -> MatrixXd { return o.P8; }});
    }
  }
  for (const auto& e : s.extra_operators) {
    if (e == "J2") t.push_back({"J2", 2, [I](const JOps& o) -> MatrixXd { return o.j2 * I(o); }});
    else throw std::invalid_argument("asym_rotor: unknown extra operator '" + e + "' (known: J2)");
  }
  return t;
}

void validate(const AsymRotorSpec& s) {
  if (s.j_min < 0 || s.j_max < s.j_min) throw std::invalid_argument("asym_rotor: need 0 <= j_min <= j_max");
  if (s.reduction != "A" && s.reduction != "S") throw std::invalid_argument("asym_rotor: reduction must be A or S");
  if (s.order != 2 && s.order != 4 && s.order != 6 && s.order != 8)
    throw std::invalid_argument("asym_rotor: order must be 2, 4, 6 or 8");
}

// Wang basis: columns of U, grouped into blocks E+, E-, O+, O-.
struct WangBlock {
  std::string label;
  std::vector<int> cols;
  std::vector<std::string> basis_labels;
};

}  // namespace

std::vector<std::string> asym_rotor_param_names(const AsymRotorSpec& spec) {
  validate(spec);
  std::vector<std::string> names;
  for (const auto& d : operator_table(spec)) names.push_back(d.name);
  return names;
}

CompiledModel compile_asym_rotor(const AsymRotorSpec& spec) {
  validate(spec);
  const auto table = operator_table(spec);
  CompiledModel model;
  model.name = "asym_rotor";
  {
    std::ostringstream js;
    std::vector<std::string> extra = spec.extra_operators;
    js << "{\"axis\":\"" << spec.axis << "\",\"extra_operators\":[";
    for (std::size_t i = 0; i < extra.size(); ++i) js << (i ? "," : "") << "\"" << extra[i] << "\"";
    js << "],\"j_max\":" << spec.j_max << ",\"j_min\":" << spec.j_min
       << ",\"model\":\"asym_rotor\",\"order\":" << spec.order << ",\"reduction\":\"" << spec.reduction
       << "\"}";
    model.spec_json = js.str();
  }
  for (const auto& d : table) model.param_names.push_back(d.name);

  double max_herm = 0.0, max_offblock = 0.0;
  for (int J = spec.j_min; J <= spec.j_max; ++J) {
    const JOps jo = make_jops(J);
    const int n = 2 * J + 1;
    MatrixXd U = MatrixXd::Zero(n, n);
    std::vector<WangBlock> blocks{{"E+", {}, {}}, {"E-", {}, {}}, {"O+", {}, {}}, {"O-", {}, {}}};
    int col = 0;
    const double s = 1.0 / std::sqrt(2.0);
    for (int K = 0; K <= J; ++K) {
      const int par = K % 2;  // 0 even, 1 odd
      if (K == 0) {
        U(J, col) = 1.0;
        blocks[0].cols.push_back(col++);
        blocks[0].basis_labels.push_back("K=0");
        continue;
      }
      for (int g : {+1, -1}) {
        U(K + J, col) = s;
        U(-K + J, col) = g * s;
        auto& b = blocks[par * 2 + (g > 0 ? 0 : 1)];
        b.cols.push_back(col++);
        b.basis_labels.push_back("K=" + std::to_string(K) + (g > 0 ? ",+" : ",-"));
      }
    }
    std::vector<MatrixXd> wang;
    for (const auto& d : table) {
      const MatrixXd m = d.build(jo);
      max_herm = std::max(max_herm, (m - m.transpose()).cwiseAbs().maxCoeff());
      wang.push_back(U.transpose() * m * U);
    }
    // Total symmetry: every operator must be block diagonal in the Wang (D2) basis.
    std::vector<int> owner(n, -1);
    for (int b = 0; b < 4; ++b)
      for (int c : blocks[b].cols) owner[c] = b;
    for (const auto& w : wang)
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
          if (owner[i] != owner[j]) max_offblock = std::max(max_offblock, std::abs(w(i, j)));

    for (const auto& b : blocks) {
      if (b.cols.empty()) continue;
      CompiledBlock cb;
      cb.id.two_j = 2 * J;
      cb.id.irrep = b.label;
      cb.basis_labels = b.basis_labels;
      const auto nb = static_cast<std::int64_t>(b.cols.size());
      for (const auto& w : wang) {
        std::vector<Triplet> t;
        for (std::int64_t i = 0; i < nb; ++i)
          for (std::int64_t j = 0; j < nb; ++j) {
            const double v = w(b.cols[i], b.cols[j]);
            if (std::abs(v) > 1e-12 * (1.0 + jo.j2 * jo.j2 * jo.j2)) t.push_back({i, j, v});
          }
        const CsrMatrix op = CsrMatrix::from_triplets(nb, nb, std::move(t));
        // Symmetrize exactly (rounding in U^T M U); the defect is certified above.
        Eigen::MatrixXd dd = op.to_dense();
        dd = 0.5 * (dd + dd.transpose());
        std::vector<Triplet> ts;
        for (std::int64_t i = 0; i < nb; ++i)
          for (std::int64_t j = 0; j < nb; ++j)
            if (dd(i, j) != 0.0) ts.push_back({i, j, dd(i, j)});
        cb.ops.push_back(CsrMatrix::from_triplets(nb, nb, std::move(ts)));
      }
      model.blocks.push_back(std::move(cb));
    }
  }

  const double scale = 1.0 + std::pow(spec.j_max * (spec.j_max + 1.0), spec.order / 2.0);
  auto require = [&](const std::string& name, double value, double bound, const std::string& detail) {
    const bool ok = value <= bound;
    model.compile_certificates.push_back({name, ok, value, bound, detail});
    if (!ok)
      throw std::runtime_error("asym_rotor compile certificate '" + name + "' failed: " +
                               std::to_string(value) + " > " + std::to_string(bound) + " (" + detail + ")");
  };
  require("hermiticity", max_herm, 1e-12 * scale, "max |O - O^T| in the |J K> basis");
  require("total_symmetry", max_offblock, 1e-11 * scale,
          "max coupling between Wang (D2) blocks: every O_k must be totally symmetric");
  int odd = 0;
  for (const auto& d : table) odd += d.degree % 2;
  require("time_reversal", odd, 0, "operators of odd degree in J are T-odd and not allowed");

  // Wigner-Eckart cross-check of the ladder matrix elements against the
  // angular-momentum library (space-fixed J_0 and J_+1 = -J_+ / sqrt 2).
  double we = 0.0;
  for (int J = std::max(spec.j_min, 1); J <= std::min(spec.j_max, 30); ++J) {
    const double red = std::sqrt(J * (J + 1.0) * (2 * J + 1.0));
    for (int M = -J; M <= J; ++M) {
      const double ph = ((J - M) % 2 == 0) ? 1.0 : -1.0;
      const double jz = ph * angular::wigner3j(2 * J, 2, 2 * J, -2 * M, 0, 2 * M) * red;
      we = std::max(we, std::abs(jz - M));
      if (M < J) {
        const double ph1 = ((J - M - 1) % 2 == 0) ? 1.0 : -1.0;
        const double jp = -std::sqrt(2.0) * ph1 * angular::wigner3j(2 * J, 2, 2 * J, -2 * (M + 1), 2, 2 * M) * red;
        we = std::max(we, std::abs(jp - std::sqrt(J * (J + 1.0) - M * (M + 1.0))));
      }
    }
  }
  require("wigner_eckart", we, 1e-12 * (1.0 + spec.j_max), "ladder elements vs 3j * reduced matrix element");
  const int tj = std::min(2 * spec.j_max, 6);
  require("cg_orthogonality", angular::cg_orthogonality_defect(tj), 1e-13, "2j <= " + std::to_string(tj));
  require("cg_completeness", angular::cg_completeness_defect(tj), 1e-13, "2j <= " + std::to_string(tj));
  return model;
}

}  // namespace riss_drucker::models
