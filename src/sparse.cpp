#include "riss_drucker/sparse.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "riss_drucker/kernels.hpp"

namespace riss_drucker {

CsrMatrix CsrMatrix::from_triplets(std::int64_t rows, std::int64_t cols, std::vector<Triplet> t,
                                   bool keep_zeros) {
  if (rows < 0 || cols < 0) throw std::invalid_argument("CsrMatrix: negative shape");
  if (cols > std::numeric_limits<std::int32_t>::max())
    throw std::invalid_argument("CsrMatrix: column count exceeds int32 index range");
  for (const auto& e : t)
    if (e.row < 0 || e.row >= rows || e.col < 0 || e.col >= cols)
      throw std::out_of_range("CsrMatrix: triplet (" + std::to_string(e.row) + ", " +
                              std::to_string(e.col) + ") outside the matrix");
  std::sort(t.begin(), t.end(), [](const Triplet& a, const Triplet& b) {
    return a.row != b.row ? a.row < b.row : a.col < b.col;
  });
  CsrMatrix m;
  m.rows = rows;
  m.cols = cols;
  m.row_ptr.assign(static_cast<std::size_t>(rows) + 1, 0);
  for (std::size_t i = 0; i < t.size();) {
    std::size_t j = i;
    double v = 0.0;
    while (j < t.size() && t[j].row == t[i].row && t[j].col == t[i].col) v += t[j++].value;
    if (keep_zeros || v != 0.0) {
      m.col_idx.push_back(static_cast<std::int32_t>(t[i].col));
      m.values.push_back(v);
      ++m.row_ptr[static_cast<std::size_t>(t[i].row) + 1];
    }
    i = j;
  }
  for (std::int64_t r = 0; r < rows; ++r) m.row_ptr[r + 1] += m.row_ptr[r];
  return m;
}

CsrMatrix CsrMatrix::zero(std::int64_t rows, std::int64_t cols) {
  return from_triplets(rows, cols, {});
}

void CsrMatrix::matvec(const double* x, double* y) const {
  kernels::csr_matvec(static_cast<std::size_t>(rows), row_ptr.data(), col_idx.data(),
                      values.data(), x, y);
}

void CsrMatrix::matvec(SimdLevel level, const double* x, double* y) const {
  kernels::csr_matvec(level, static_cast<std::size_t>(rows), row_ptr.data(), col_idx.data(),
                      values.data(), x, y);
}

Eigen::MatrixXd CsrMatrix::to_dense() const {
  Eigen::MatrixXd d = Eigen::MatrixXd::Zero(rows, cols);
  for (std::int64_t r = 0; r < rows; ++r)
    for (auto k = row_ptr[r]; k < row_ptr[r + 1]; ++k) d(r, col_idx[k]) += values[k];
  return d;
}

std::int64_t CsrMatrix::bandwidth() const {
  std::int64_t bw = 0;
  for (std::int64_t r = 0; r < rows; ++r)
    for (auto k = row_ptr[r]; k < row_ptr[r + 1]; ++k)
      bw = std::max<std::int64_t>(bw, std::abs(r - static_cast<std::int64_t>(col_idx[k])));
  return bw;
}

double CsrMatrix::inf_norm() const {
  double n = 0.0;
  for (std::int64_t r = 0; r < rows; ++r) {
    double s = 0.0;
    for (auto k = row_ptr[r]; k < row_ptr[r + 1]; ++k) s += std::abs(values[k]);
    n = std::max(n, s);
  }
  return n;
}

namespace {
// Value at (r, c), 0 if not stored. Column indices are sorted per row.
double lookup(const CsrMatrix& m, std::int64_t r, std::int32_t c) {
  const auto b = m.col_idx.begin() + m.row_ptr[r], e = m.col_idx.begin() + m.row_ptr[r + 1];
  const auto it = std::lower_bound(b, e, c);
  return (it != e && *it == c) ? m.values[static_cast<std::size_t>(it - m.col_idx.begin())] : 0.0;
}
}  // namespace

double CsrMatrix::symmetry_defect() const {
  if (rows != cols) return std::numeric_limits<double>::infinity();
  double d = 0.0;
  for (std::int64_t r = 0; r < rows; ++r)
    for (auto k = row_ptr[r]; k < row_ptr[r + 1]; ++k)
      d = std::max(d, std::abs(values[k] - lookup(*this, col_idx[k], static_cast<std::int32_t>(r))));
  return d;
}

Eigen::VectorXd CsrMatrix::diagonal() const {
  const std::int64_t n = std::min(rows, cols);
  Eigen::VectorXd d(n);
  for (std::int64_t i = 0; i < n; ++i) d[i] = lookup(*this, i, static_cast<std::int32_t>(i));
  return d;
}

Eigen::VectorXd CsrMatrix::subdiagonal() const {
  const std::int64_t n = std::min(rows, cols);
  Eigen::VectorXd e(std::max<std::int64_t>(n - 1, 0));
  for (std::int64_t i = 0; i + 1 < n; ++i) e[i] = lookup(*this, i + 1, static_cast<std::int32_t>(i));
  return e;
}

OperatorSum::OperatorSum(const std::vector<CsrMatrix>& ops) {
  if (ops.empty()) throw std::invalid_argument("OperatorSum: no operators");
  const auto rows = ops.front().rows, cols = ops.front().cols;
  std::vector<Triplet> all;
  for (const auto& op : ops) {
    if (op.rows != rows || op.cols != cols)
      throw std::invalid_argument("OperatorSum: operators must share one shape");
    for (std::int64_t r = 0; r < rows; ++r)
      for (auto k = op.row_ptr[r]; k < op.row_ptr[r + 1]; ++k) all.push_back({r, op.col_idx[k], 0.0});
  }
  pattern_ = CsrMatrix::from_triplets(rows, cols, std::move(all), /*keep_zeros=*/true);
  for (const auto& op : ops) {
    std::vector<std::int64_t> sc(op.values.size());
    for (std::int64_t r = 0; r < rows; ++r)
      for (auto k = op.row_ptr[r]; k < op.row_ptr[r + 1]; ++k) {
        const auto b = pattern_.col_idx.begin() + pattern_.row_ptr[r];
        const auto e = pattern_.col_idx.begin() + pattern_.row_ptr[r + 1];
        sc[k] = std::lower_bound(b, e, op.col_idx[k]) - pattern_.col_idx.begin();
      }
    scatter_.push_back(std::move(sc));
    values_.push_back(op.values);
  }
}

CsrMatrix OperatorSum::assemble(const std::vector<double>& theta) const {
  if (theta.size() != values_.size())
    throw std::invalid_argument("OperatorSum: expected " + std::to_string(values_.size()) +
                                " parameters, got " + std::to_string(theta.size()));
  CsrMatrix h = pattern_;
  std::fill(h.values.begin(), h.values.end(), 0.0);
  for (std::size_t k = 0; k < values_.size(); ++k) {
    const double t = theta[k];
    if (t == 0.0) continue;
    for (std::size_t e = 0; e < values_[k].size(); ++e) h.values[scatter_[k][e]] += t * values_[k][e];
  }
  return h;
}

}  // namespace riss_drucker
