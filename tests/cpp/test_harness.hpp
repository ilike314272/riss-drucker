#pragma once
// Minimal self-contained test harness (avoids a test-framework dependency).
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace rd_test {

struct Case {
  const char* name;
  std::function<void()> fn;
};
inline std::vector<Case>& registry() {
  static std::vector<Case> r;
  return r;
}
inline int& failures() {
  static int f = 0;
  return f;
}
struct Register {
  Register(const char* n, std::function<void()> f) { registry().push_back({n, std::move(f)}); }
};

inline int run_all() {
  for (const auto& c : registry()) {
    const int before = failures();
    try {
      c.fn();
    } catch (const std::exception& e) {
      std::printf("  exception: %s\n", e.what());
      ++failures();
    }
    std::printf("[%s] %s\n", failures() == before ? " OK " : "FAIL", c.name);
  }
  std::printf("%zu cases, %d failed checks\n", registry().size(), failures());
  return failures() == 0 ? 0 : 1;
}

}  // namespace rd_test

#define RD_CAT2(a, b) a##b
#define RD_CAT(a, b) RD_CAT2(a, b)
#define TEST_CASE(name)                                                   \
  static void RD_CAT(rd_test_fn_, __LINE__)();                            \
  static rd_test::Register RD_CAT(rd_test_reg_, __LINE__)(name, RD_CAT(rd_test_fn_, __LINE__)); \
  static void RD_CAT(rd_test_fn_, __LINE__)()

#define CHECK(cond)                                                             \
  do {                                                                          \
    if (!(cond)) {                                                              \
      std::printf("  %s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #cond);    \
      ++rd_test::failures();                                                    \
    }                                                                           \
  } while (0)

#define CHECK_CLOSE(a, b, tol)                                                          \
  do {                                                                                  \
    const double rd_a_ = (a), rd_b_ = (b), rd_t_ = (tol);                               \
    if (!(std::abs(rd_a_ - rd_b_) <= rd_t_)) {                                          \
      std::printf("  %s:%d: |%s - %s| = %.3e > %.3e (%.17g vs %.17g)\n", __FILE__,      \
                  __LINE__, #a, #b, std::abs(rd_a_ - rd_b_), rd_t_, rd_a_, rd_b_);       \
      ++rd_test::failures();                                                            \
    }                                                                                   \
  } while (0)

#define CHECK_THROWS(expr)                                                      \
  do {                                                                          \
    bool rd_threw_ = false;                                                     \
    try { (void)(expr); } catch (...) { rd_threw_ = true; }                     \
    if (!rd_threw_) {                                                           \
      std::printf("  %s:%d: expected exception from %s\n", __FILE__, __LINE__, #expr); \
      ++rd_test::failures();                                                    \
    }                                                                           \
  } while (0)

#define RD_TEST_MAIN \
  int main() { return rd_test::run_all(); }
