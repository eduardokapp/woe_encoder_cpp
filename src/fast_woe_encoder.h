#ifndef FAST_WOE_ENCODER_H_
#define FAST_WOE_ENCODER_H_

#include <vector>

namespace fast_woe_encoder {

namespace fast_woe_encoder_internal {

// Default precision when smoothing.
constexpr double kEpsilon = 0;

} // namespace fast_woe_encoder_internal

// Input args for WoEEncoder
struct WoEEncoderOptions {
  // Verbose level.
  int verbose = 0;
  // Smoothing precision.
  double epsilon = fast_woe_encoder_internal::kEpsilon;
  // Value to use for default WOE.
  double default_woe = 0.0;
};

// A simple implementation for WOE encoding.
class WoEEncoder {
public:
  explicit WoEEncoder(WoEEncoderOptions options = {}) : options_(options){};

  // Given a matrix of integers, populates a map of WOE values.
  // The matrix should be represented columnwise.
  // Only supports boolean targets.
  void Fit(const std::vector<std::vector<int>> &features,
           const std::vector<bool> &targets);

  // Transforms a matrix of integers to a matrix of WOE values.
  [[nodiscard]] std::vector<std::vector<double>>
  Transform(const std::vector<std::vector<int>> &features) const;

private:
  // Populate WoEMap for a given column.
  void PopulateWoEMap(size_t column_index, const std::vector<int> &column,
                      const std::vector<bool> &targets);

  // Each column has its own vector of WoE values. We assume that
  // categories are always (represented as) integers.
  std::vector<std::vector<double>> woe_map_;
  int64_t total_pos_{};
  int64_t total_neg_{};
  WoEEncoderOptions options_;
};

} // namespace fast_woe_encoder

#endif // FAST_WOE_ENCODER_H_