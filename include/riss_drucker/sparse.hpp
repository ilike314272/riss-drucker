#pragma once
// CSR operators and the fixed-pattern sum H(theta) = sum_k t_k O_k (README 5.3).
// Layout as in kernels.hpp: int64 row pointers, int32 column indices.
#include <cstdint>
#include <vector>

#include <Eigen/Dense>

#include "riss_drucker/cpu_features.hpp"

namespace riss_drucker {

struct Triplet {
  std::int64_t row;
  std::int64_t col;
  double value;
};

struct CsrMatrix {
  std::int64_t rows = 0;
  std::int64_t cols = 0;
  std::vector<std::int64_t> row_ptr{0};
  std::vector<std::int32_t> col_idx;
  std::vector<double> values;

  std::int64_t nnz() const { return static_cast<std::int64_t>(values.size()); }

  // Sorts, sums duplicates. Exact zeros are dropped unless keep_zeros.
  static CsrMatrix from_triplets(std::int64_t rows, std::int64_t cols, std::vector<Triplet> t,
                                 bool keep_zeros = false);
  static CsrMatrix zero(std::int64_t rows, std::int64_t cols);

  void matvec(const double* x, double* y) const;                   // active SIMD level
  void matvec(SimdLevel level, const double* x, double* y) const;  // explicit level
  Eigen::MatrixXd to_dense() const;
  std::int64_t bandwidth() const;      // max |i - j| over stored entries
  double inf_norm() const;             // max absolute row sum (bounds ||H||_2)
  double symmetry_defect() const;      // max |A_ij - A_ji|
  Eigen::VectorXd diagonal() const;
  Eigen::VectorXd subdiagonal() const;  // A(i+1, i), for tridiagonal blocks
};

// Union sparsity pattern of a list of same-shape operators plus a scatter map,
// so assembling H(theta) is one pass over the operator values. The pattern is
// fixed by the model structure; theta only changes the values (README 1).
class OperatorSum {
 public:
  explicit OperatorSum(const std::vector<CsrMatrix>& ops);
  CsrMatrix assemble(const std::vector<double>& theta) const;
  std::size_t num_terms() const { return values_.size(); }
  const CsrMatrix& pattern() const { return pattern_; }

 private:
  CsrMatrix pattern_;
  std::vector<std::vector<std::int64_t>> scatter_;
  std::vector<std::vector<double>> values_;
};

}  // namespace riss_drucker
