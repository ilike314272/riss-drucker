#include "riss_drucker/angular.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace riss_drucker::angular {
namespace {

// ---- Minimal unsigned big integer (little-endian 32-bit limbs) -------------
class BigUInt {
 public:
  BigUInt() = default;
  explicit BigUInt(std::uint64_t v) {
    while (v) { d_.push_back(static_cast<std::uint32_t>(v)); v >>= 32; }
  }
  bool zero() const { return d_.empty(); }
  void mul_small(std::uint32_t m) {
    std::uint64_t carry = 0;
    for (auto& x : d_) {
      const std::uint64_t t = static_cast<std::uint64_t>(x) * m + carry;
      x = static_cast<std::uint32_t>(t);
      carry = t >> 32;
    }
    if (carry) d_.push_back(static_cast<std::uint32_t>(carry));
    if (m == 0) d_.clear();
  }
  static int cmp(const BigUInt& a, const BigUInt& b) {
    if (a.d_.size() != b.d_.size()) return a.d_.size() < b.d_.size() ? -1 : 1;
    for (std::size_t i = a.d_.size(); i-- > 0;)
      if (a.d_[i] != b.d_[i]) return a.d_[i] < b.d_[i] ? -1 : 1;
    return 0;
  }
  void add(const BigUInt& o) {
    if (d_.size() < o.d_.size()) d_.resize(o.d_.size(), 0);
    std::uint64_t carry = 0;
    for (std::size_t i = 0; i < d_.size(); ++i) {
      const std::uint64_t t = static_cast<std::uint64_t>(d_[i]) + (i < o.d_.size() ? o.d_[i] : 0) + carry;
      d_[i] = static_cast<std::uint32_t>(t);
      carry = t >> 32;
    }
    if (carry) d_.push_back(static_cast<std::uint32_t>(carry));
  }
  void sub(const BigUInt& o) {  // requires *this >= o
    std::int64_t borrow = 0;
    for (std::size_t i = 0; i < d_.size(); ++i) {
      std::int64_t t = static_cast<std::int64_t>(d_[i]) - (i < o.d_.size() ? o.d_[i] : 0) - borrow;
      borrow = t < 0;
      if (t < 0) t += (std::int64_t(1) << 32);
      d_[i] = static_cast<std::uint32_t>(t);
    }
    trim();
  }
  static BigUInt mul(const BigUInt& a, const BigUInt& b) {
    BigUInt r;
    if (a.zero() || b.zero()) return r;
    r.d_.assign(a.d_.size() + b.d_.size(), 0);
    for (std::size_t i = 0; i < a.d_.size(); ++i) {
      std::uint64_t carry = 0;
      for (std::size_t j = 0; j < b.d_.size(); ++j) {
        const std::uint64_t t = static_cast<std::uint64_t>(a.d_[i]) * b.d_[j] + r.d_[i + j] + carry;
        r.d_[i + j] = static_cast<std::uint32_t>(t);
        carry = t >> 32;
      }
      r.d_[i + b.d_.size()] += static_cast<std::uint32_t>(carry);
    }
    r.trim();
    return r;
  }
  std::uint32_t divmod_small(std::uint32_t m) {  // *this /= m, returns remainder
    std::uint64_t rem = 0;
    for (std::size_t i = d_.size(); i-- > 0;) {
      const std::uint64_t cur = (rem << 32) | d_[i];
      d_[i] = static_cast<std::uint32_t>(cur / m);
      rem = cur % m;
    }
    trim();
    return static_cast<std::uint32_t>(rem);
  }
  std::string to_string() const {
    if (zero()) return "0";
    BigUInt t = *this;
    std::vector<std::uint32_t> chunks;
    while (!t.zero()) chunks.push_back(t.divmod_small(1000000000u));
    std::string s = std::to_string(chunks.back());
    for (std::size_t i = chunks.size() - 1; i-- > 0;) {
      std::string c = std::to_string(chunks[i]);
      s += std::string(9 - c.size(), '0') + c;
    }
    return s;
  }
  // value = mantissa * 2^exp2, using the top 96 bits.
  long double mantissa(int& exp2) const {
    long double m = 0;
    const std::size_t n = d_.size(), top = std::min<std::size_t>(n, 3);
    for (std::size_t i = 0; i < top; ++i) m = m * 4294967296.0L + d_[n - 1 - i];
    exp2 = static_cast<int>(32 * (n - top));
    int e = 0;
    m = std::frexp(m, &e);
    exp2 += e;
    return m;
  }

 private:
  void trim() { while (!d_.empty() && d_.back() == 0) d_.pop_back(); }
  std::vector<std::uint32_t> d_;
};

// ---- Prime-factorized factorials -----------------------------------------
const std::vector<int>& primes_upto(int n) {
  static std::vector<int> primes;
  static int limit = 1;
  static std::mutex mu;
  std::lock_guard<std::mutex> lock(mu);
  if (n > limit) {
    const int lim = std::max(n, 2 * limit);
    std::vector<char> sieve(static_cast<std::size_t>(lim) + 1, 1);
    primes.clear();
    for (int i = 2; i <= lim; ++i) {
      if (!sieve[i]) continue;
      primes.push_back(i);
      for (long long j = 1LL * i * i; j <= lim; j += i) sieve[static_cast<std::size_t>(j)] = 0;
    }
    limit = lim;
  }
  return primes;
}

using Exps = std::vector<int>;  // exponent per prime (index into prime table)

void add_factorial(Exps& e, const std::vector<int>& primes, int n, int sign) {
  for (std::size_t i = 0; i < primes.size() && primes[i] <= n; ++i)
    for (long long pk = primes[i]; pk <= n; pk *= primes[i]) e[i] += sign * static_cast<int>(n / pk);
}

struct Term {
  int sign;
  Exps e;  // term = sign * prod p^{-e}
};

// value = phase * sqrt(prod p^{pref}) * sum_t term_t
ExactValue racah(int phase, const Exps& pref, const std::vector<Term>& terms,
                 const std::vector<int>& primes) {
  ExactValue out;
  if (terms.empty()) return out;
  const std::size_t np = pref.size();
  Exps E(np, std::numeric_limits<int>::min());
  for (const auto& t : terms)
    for (std::size_t i = 0; i < np; ++i) E[i] = std::max(E[i], t.e[i]);
  BigUInt pos, neg;
  for (const auto& t : terms) {
    BigUInt v(1);
    for (std::size_t i = 0; i < np; ++i)
      for (int k = E[i] - t.e[i]; k > 0; --k) v.mul_small(static_cast<std::uint32_t>(primes[i]));
    (t.sign > 0 ? pos : neg).add(v);
  }
  int s = BigUInt::cmp(pos, neg);
  if (s == 0) return out;
  BigUInt S = s > 0 ? pos : neg;
  S.sub(s > 0 ? neg : pos);
  out.sign = phase * s;

  // value = sign * S * prod p^{h/2}, h = pref - 2E
  Exps h(np);
  for (std::size_t i = 0; i < np; ++i) h[i] = pref[i] - 2 * E[i];
  int e2 = 0;
  long double acc = S.mantissa(e2);
  for (std::size_t i = 0; i < np; ++i) {
    if (h[i] == 0) continue;
    acc *= std::pow(static_cast<long double>(primes[i]), 0.5L * h[i]);
    int e = 0;
    acc = std::frexp(acc, &e);
    e2 += e;
  }
  out.value = out.sign * static_cast<double>(std::ldexp(acc, e2));

  // Exact: value^2 = S^2 prod p^h, reduced.
  BigUInt num = BigUInt::mul(S, S);
  for (std::size_t i = 0; i < np; ++i)
    for (int k = 0; k < h[i]; ++k) num.mul_small(static_cast<std::uint32_t>(primes[i]));
  for (std::size_t i = 0; i < np; ++i) {
    while (h[i] < 0) {
      BigUInt q = num;
      if (q.divmod_small(static_cast<std::uint32_t>(primes[i])) != 0) break;
      num = q;
      ++h[i];
    }
  }
  BigUInt den(1);
  for (std::size_t i = 0; i < np; ++i)
    for (int k = 0; k < -h[i]; ++k) den.mul_small(static_cast<std::uint32_t>(primes[i]));
  out.num = num.to_string();
  out.den = den.to_string();
  return out;
}

bool triangle(int ta, int tb, int tc) {
  return tc <= ta + tb && tc >= std::abs(ta - tb) && (ta + tb + tc) % 2 == 0;
}

void check_args(std::initializer_list<int> tjs) {
  for (int t : tjs)
    if (t < 0) throw std::invalid_argument("angular: 2j arguments must be non-negative");
}

// Triangle coefficient Delta(abc) exponents (squared): (a+b-c)!(a-b+c)!(-a+b+c)!/(a+b+c+1)!
void add_delta(Exps& e, const std::vector<int>& p, int ta, int tb, int tc) {
  add_factorial(e, p, (ta + tb - tc) / 2, 1);
  add_factorial(e, p, (ta - tb + tc) / 2, 1);
  add_factorial(e, p, (-ta + tb + tc) / 2, 1);
  add_factorial(e, p, (ta + tb + tc) / 2 + 1, -1);
}

}  // namespace

ExactValue wigner3j_exact(int tj1, int tj2, int tj3, int tm1, int tm2, int tm3) {
  check_args({tj1, tj2, tj3});
  if (tm1 + tm2 + tm3 != 0) return {};
  const int tj[3] = {tj1, tj2, tj3}, tm[3] = {tm1, tm2, tm3};
  for (int i = 0; i < 3; ++i)
    if (std::abs(tm[i]) > tj[i] || (tj[i] + tm[i]) % 2 != 0) return {};
  if (!triangle(tj1, tj2, tj3)) return {};

  const int nmax = (tj1 + tj2 + tj3) / 2 + 1;
  const auto& primes = primes_upto(std::max(nmax, 2));
  std::size_t np = 0;
  while (np < primes.size() && primes[np] <= nmax) ++np;
  const std::vector<int> p(primes.begin(), primes.begin() + static_cast<long>(np));

  Exps pref(np, 0);
  add_delta(pref, p, tj1, tj2, tj3);
  for (int i = 0; i < 3; ++i) {
    add_factorial(pref, p, (tj[i] + tm[i]) / 2, 1);
    add_factorial(pref, p, (tj[i] - tm[i]) / 2, 1);
  }
  const int a1 = (tj3 - tj2 + tm1) / 2, a2 = (tj3 - tj1 - tm2) / 2;
  const int b1 = (tj1 + tj2 - tj3) / 2, b2 = (tj1 - tm1) / 2, b3 = (tj2 + tm2) / 2;
  const int kmin = std::max({0, -a1, -a2}), kmax = std::min({b1, b2, b3});
  std::vector<Term> terms;
  for (int k = kmin; k <= kmax; ++k) {
    Term t{(k % 2 == 0) ? 1 : -1, Exps(np, 0)};
    for (int n : {k, a1 + k, a2 + k, b1 - k, b2 - k, b3 - k}) add_factorial(t.e, p, n, 1);
    terms.push_back(std::move(t));
  }
  const int ph = (tj1 - tj2 - tm3) / 2;
  return racah((ph % 2 == 0) ? 1 : -1, pref, terms, p);
}

double wigner3j(int tj1, int tj2, int tj3, int tm1, int tm2, int tm3) {
  return wigner3j_exact(tj1, tj2, tj3, tm1, tm2, tm3).value;
}

ExactValue wigner6j_exact(int tj1, int tj2, int tj3, int tj4, int tj5, int tj6) {
  check_args({tj1, tj2, tj3, tj4, tj5, tj6});
  if (!triangle(tj1, tj2, tj3) || !triangle(tj1, tj5, tj6) || !triangle(tj4, tj2, tj6) ||
      !triangle(tj4, tj5, tj3))
    return {};
  const int al[4] = {(tj1 + tj2 + tj3) / 2, (tj1 + tj5 + tj6) / 2, (tj4 + tj2 + tj6) / 2,
                     (tj4 + tj5 + tj3) / 2};
  const int be[3] = {(tj1 + tj2 + tj4 + tj5) / 2, (tj2 + tj3 + tj5 + tj6) / 2,
                     (tj3 + tj1 + tj6 + tj4) / 2};
  const int tmin = *std::max_element(al, al + 4), tmax = *std::min_element(be, be + 3);
  const int nmax = std::max(tmax + 1, *std::max_element(al, al + 4) + 1);
  const auto& primes = primes_upto(std::max(nmax, 2));
  std::size_t np = 0;
  while (np < primes.size() && primes[np] <= nmax) ++np;
  const std::vector<int> p(primes.begin(), primes.begin() + static_cast<long>(np));

  Exps pref(np, 0);
  add_delta(pref, p, tj1, tj2, tj3);
  add_delta(pref, p, tj1, tj5, tj6);
  add_delta(pref, p, tj4, tj2, tj6);
  add_delta(pref, p, tj4, tj5, tj3);
  std::vector<Term> terms;
  for (int t = tmin; t <= tmax; ++t) {
    Term term{(t % 2 == 0) ? 1 : -1, Exps(np, 0)};
    for (int a : al) add_factorial(term.e, p, t - a, 1);
    for (int b : be) add_factorial(term.e, p, b - t, 1);
    add_factorial(term.e, p, t + 1, -1);
    terms.push_back(std::move(term));
  }
  return racah(1, pref, terms, p);
}

double wigner6j(int tj1, int tj2, int tj3, int tj4, int tj5, int tj6) {
  return wigner6j_exact(tj1, tj2, tj3, tj4, tj5, tj6).value;
}

double wigner9j(int tj1, int tj2, int tj3, int tj4, int tj5, int tj6, int tj7, int tj8, int tj9) {
  check_args({tj1, tj2, tj3, tj4, tj5, tj6, tj7, tj8, tj9});
  // sum_x (-1)^{2x}(2x+1) {j1 j4 j7; j8 j9 x}{j2 j5 j8; j4 x j6}{j3 j6 j9; x j1 j2}
  const int lo = std::max({std::abs(tj1 - tj9), std::abs(tj4 - tj8), std::abs(tj2 - tj6)});
  const int hi = std::min({tj1 + tj9, tj4 + tj8, tj2 + tj6});
  long double s = 0;
  for (int tx = lo; tx <= hi; tx += 2) {
    if ((tx + tj1 + tj9) % 2 != 0) continue;
    const double a = wigner6j(tj1, tj4, tj7, tj8, tj9, tx);
    if (a == 0.0) continue;
    const double b = wigner6j(tj2, tj5, tj8, tj4, tx, tj6);
    if (b == 0.0) continue;
    const double c = wigner6j(tj3, tj6, tj9, tx, tj1, tj2);
    s += static_cast<long double>((tx % 2 == 0) ? 1 : -1) * (tx + 1) * a * b * c;
  }
  return static_cast<double>(s);
}

double clebsch_gordan(int tj1, int tm1, int tj2, int tm2, int tJ, int tM) {
  if (tm1 + tm2 != tM) return 0.0;
  const int ph = (tj1 - tj2 + tM) / 2;
  if ((tj1 - tj2 + tM) % 2 != 0) return 0.0;
  return ((ph % 2 == 0) ? 1.0 : -1.0) * std::sqrt(tJ + 1.0) * wigner3j(tj1, tj2, tJ, tm1, tm2, -tM);
}

double cg_orthogonality_defect(int two_jmax) {
  double dev = 0.0;
  for (int t1 = 0; t1 <= two_jmax; ++t1)
    for (int t2 = 0; t2 <= two_jmax; ++t2)
      for (int tJ = std::abs(t1 - t2); tJ <= t1 + t2; tJ += 2)
        for (int tJp = std::abs(t1 - t2); tJp <= t1 + t2; tJp += 2)
          for (int tM = -std::min(tJ, tJp); tM <= std::min(tJ, tJp); tM += 2) {
            double s = 0.0;
            for (int tm1 = -t1; tm1 <= t1; tm1 += 2) {
              const int tm2 = tM - tm1;
              if (std::abs(tm2) > t2 || (t2 + tm2) % 2) continue;
              s += clebsch_gordan(t1, tm1, t2, tm2, tJ, tM) * clebsch_gordan(t1, tm1, t2, tm2, tJp, tM);
            }
            dev = std::max(dev, std::abs(s - (tJ == tJp ? 1.0 : 0.0)));
          }
  return dev;
}

double cg_completeness_defect(int two_jmax) {
  double dev = 0.0;
  for (int t1 = 0; t1 <= two_jmax; ++t1)
    for (int t2 = 0; t2 <= two_jmax; ++t2)
      for (int tm1 = -t1; tm1 <= t1; tm1 += 2)
        for (int tm1p = -t1; tm1p <= t1; tm1p += 2)
          for (int tm2 = -t2; tm2 <= t2; tm2 += 2) {
            const int tM = tm1 + tm2, tm2p = tM - tm1p;
            if (std::abs(tm2p) > t2) continue;
            double s = 0.0;
            for (int tJ = std::abs(t1 - t2); tJ <= t1 + t2; tJ += 2)
              s += clebsch_gordan(t1, tm1, t2, tm2, tJ, tM) * clebsch_gordan(t1, tm1p, t2, tm2p, tJ, tM);
            dev = std::max(dev, std::abs(s - (tm1 == tm1p ? 1.0 : 0.0)));
          }
  return dev;
}

double sixj_orthogonality_defect(int two_jmax) {
  double dev = 0.0;
  for (int ta = 0; ta <= two_jmax; ++ta)
    for (int tb = 0; tb <= two_jmax; ++tb)
      for (int tc = 0; tc <= two_jmax; ++tc)
        for (int td = 0; td <= two_jmax; ++td)
          for (int tf = 0; tf <= 2 * two_jmax; ++tf)
            for (int tfp = 0; tfp <= 2 * two_jmax; ++tfp) {
              if ((tf - tfp) % 2) continue;
              // {a b x; c d f}: f couples (a, d) and (c, b).
              if (!triangle(ta, td, tf) || !triangle(tc, tb, tf)) continue;
              if (!triangle(ta, td, tfp) || !triangle(tc, tb, tfp)) continue;
              double s = 0.0;
              for (int tx = std::abs(ta - tb); tx <= ta + tb; tx += 2)
                s += (tx + 1.0) * (tf + 1.0) * wigner6j(ta, tb, tx, tc, td, tf) * wigner6j(ta, tb, tx, tc, td, tfp);
              dev = std::max(dev, std::abs(s - (tf == tfp ? 1.0 : 0.0)));
            }
  return dev;
}

}  // namespace riss_drucker::angular
