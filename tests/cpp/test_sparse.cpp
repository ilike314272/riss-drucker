#include <random>

#include "riss_drucker/block_id.hpp"
#include "riss_drucker/sparse.hpp"
#include "test_harness.hpp"
#include "test_util.hpp"

using namespace riss_drucker;

TEST_CASE("from_triplets sums duplicates, sorts, drops zeros, checks bounds") {
  auto m = CsrMatrix::from_triplets(3, 3, {{2, 1, 1.0}, {0, 2, 2.0}, {2, 1, 0.5}, {1, 1, 0.0}});
  CHECK(m.nnz() == 2);
  const Eigen::MatrixXd d = m.to_dense();
  CHECK(d(2, 1) == 1.5 && d(0, 2) == 2.0 && d(1, 1) == 0.0);
  CHECK(CsrMatrix::from_triplets(3, 3, {{1, 1, 0.0}}, true).nnz() == 1);
  CHECK_THROWS(CsrMatrix::from_triplets(2, 2, {{2, 0, 1.0}}));
  CHECK(CsrMatrix::zero(4, 4).row_ptr.size() == 5);
}

TEST_CASE("matvec, norms, bandwidth, diagonals, symmetry defect") {
  std::mt19937_64 rng(1);
  const auto m = rd_test::random_banded(30, 2, rng);
  const Eigen::MatrixXd d = m.to_dense();
  Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(30, -1, 2), y(30);
  m.matvec(x.data(), y.data());
  CHECK_CLOSE((y - d * x).cwiseAbs().maxCoeff(), 0.0, 1e-13);
  CHECK(m.bandwidth() == 2);
  CHECK_CLOSE(m.inf_norm(), d.cwiseAbs().rowwise().sum().maxCoeff(), 1e-14);
  CHECK(m.symmetry_defect() == 0.0);
  CHECK((m.diagonal() - d.diagonal()).cwiseAbs().maxCoeff() == 0.0);
  CHECK((m.subdiagonal() - d.diagonal(-1)).cwiseAbs().maxCoeff() == 0.0);
  const auto asym = CsrMatrix::from_triplets(2, 2, {{0, 1, 1.0}, {1, 0, 0.75}});
  CHECK_CLOSE(asym.symmetry_defect(), 0.25, 1e-16);
}

TEST_CASE("OperatorSum equals sum_k t_k dense(O_k) for disjoint and overlapping patterns") {
  std::mt19937_64 rng(2);
  std::vector<CsrMatrix> ops{rd_test::random_banded(20, 1, rng), rd_test::random_banded(20, 3, rng),
                             CsrMatrix::from_triplets(20, 20, {{5, 5, 1.0}}), CsrMatrix::zero(20, 20)};
  const std::vector<double> theta{0.3, -1.7, 2.5, 9.0};
  const OperatorSum sum(ops);
  const Eigen::MatrixXd h = sum.assemble(theta).to_dense();
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(20, 20);
  for (std::size_t k = 0; k < ops.size(); ++k) ref += theta[k] * ops[k].to_dense();
  CHECK_CLOSE((h - ref).cwiseAbs().maxCoeff(), 0.0, 1e-14);
  // Pattern fixed by structure: theta = 0 keeps the pattern.
  CHECK(sum.assemble({0, 0, 0, 0}).nnz() == sum.pattern().nnz());
  CHECK_THROWS(sum.assemble({1.0}));
  CHECK_THROWS(OperatorSum({CsrMatrix::zero(2, 2), CsrMatrix::zero(3, 3)}));
}

TEST_CASE("BlockId key is canonical and the hash is stable") {
  BlockId a;
  a.two_j = 4;
  a.irrep = "sigma=1";
  a.sub = "K=-2";
  CHECK(a.key() == "J2=4|irrep=sigma=1|spin=|el=0|sub=K=-2");
  CHECK(stable_hash64("") == 14695981039346656037ull);
  CHECK(stable_hash64("a") == 0xaf63dc4c8601ec8cull);  // FNV-1a reference value
}

RD_TEST_MAIN
