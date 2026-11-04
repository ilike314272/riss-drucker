// Dense eigh baseline per linalg backend (solver path "dense", README 3.1).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>

#include "riss_drucker/linalg_backend.hpp"

using namespace riss_drucker;

int main() {
  std::mt19937_64 rng(11);
  std::normal_distribution<double> d;
  for (int n : {100, 400, 1000}) {
    Eigen::MatrixXd a(n, n);
    for (int i = 0; i < n; ++i)
      for (int j = 0; j <= i; ++j) a(i, j) = a(j, i) = d(rng);
    for (auto b : linalg::available_backends()) {
      double best = 1e300;
      for (int r = 0; r < 3; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        auto res = linalg::eigh(a, b, true);
        best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
      }
      std::printf("{\"kernel\":\"eigh\",\"n\":%d,\"backend\":\"%s\",\"seconds\":%.3e}\n", n,
                  linalg::to_string(b), best);
    }
  }
  return 0;
}
