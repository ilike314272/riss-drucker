#include <string>

#include "riss_drucker/build_info.hpp"
#include "riss_drucker/c_api.h"
#include "test_harness.hpp"

using namespace riss_drucker;

TEST_CASE("portable baseline flags recorded; forbidden flags absent") {
  const auto& b = build_info();
  CHECK(b.cxx_flags.find("-march=x86-64-v2") != std::string::npos);
  CHECK(b.cxx_flags.find("-ffp-contract=off") != std::string::npos);
  for (const char* bad : {"march=native", "ffast-math", "Ofast", "mavx"})
    CHECK(b.cxx_flags.find(bad) == std::string::npos);
}

TEST_CASE("provenance fields populated") {
  const auto& b = build_info();
  CHECK(!b.version.empty());
  CHECK(!b.git_commit.empty());
  CHECK(!b.compiler_id.empty());
  CHECK(!b.linalg_backends.empty() && b.linalg_backends.front() == "eigen");
}

TEST_CASE("C API smoke") {
  CHECK(std::string(rd_version()) == build_info().version);
  const double x[3] = {1, 2, 3};
  CHECK(rd_dot(3, x, x) == 14.0);
  const double a[4] = {2, 1, 1, 2};  // eigenvalues 1, 3
  double w[2];
  CHECK(rd_eigvalsh(2, a, w) == 0);
  CHECK_CLOSE(w[0], 1.0, 1e-14);
  CHECK_CLOSE(w[1], 3.0, 1e-14);
}

RD_TEST_MAIN
