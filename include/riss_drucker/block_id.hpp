#pragma once
// Block identifier: blocks (J, irrep, spin species, ...) are independent and
// are the unit of caching, parallelism and checkpointing (README 3.1, 5.3).
#include <cstdint>
#include <string>

namespace riss_drucker {

struct BlockId {
  int two_j = 0;               // 2J, so half-integer J (double groups, M8) fits
  std::string irrep;           // symmetry label in the model's group chain
  std::string spin_species;    // nuclear-spin species, "" if not resolved
  int electronic_state = 0;    // reserved for multi-surface models (Phase B)
  std::string sub;             // model-specific good quantum numbers, "k=v;k=v"

  // Canonical, stable string form: part of cache keys and result files.
  std::string key() const;
  bool operator==(const BlockId& o) const { return key() == o.key(); }
  bool operator<(const BlockId& o) const { return key() < o.key(); }
};

// FNV-1a 64-bit: stable across processes and platforms (unlike std::hash).
std::uint64_t stable_hash64(const std::string& s);

}  // namespace riss_drucker
