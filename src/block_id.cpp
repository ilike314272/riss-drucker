#include "riss_drucker/block_id.hpp"

namespace riss_drucker {

std::string BlockId::key() const {
  return "J2=" + std::to_string(two_j) + "|irrep=" + irrep + "|spin=" + spin_species +
         "|el=" + std::to_string(electronic_state) + "|sub=" + sub;
}

std::uint64_t stable_hash64(const std::string& s) {
  std::uint64_t h = 14695981039346656037ull;
  for (unsigned char c : s) {
    h ^= c;
    h *= 1099511628211ull;
  }
  return h;
}

}  // namespace riss_drucker
