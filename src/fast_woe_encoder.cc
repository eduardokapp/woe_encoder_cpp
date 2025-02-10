#include "fast_woe_encoder.h"

#include <cmath>
#include <iostream>
#include <ostream>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fast_woe_encoder {
namespace {

// Helper function for verbose logging.
void VLog(int verbose, const std::string &message) {
  if (verbose > 0) {
    std::cout << message << std::endl;
  }
}

// Helper function to calculate category counts (now separated).
std::vector<std::pair<int64_t, int64_t>>
CalculateCategoryCounts(const std::vector<int> &column,
                        const std::vector<bool> &targets, int &max_category) {

  max_category = 0;
  for (int cat : column) {
    max_category = std::max(max_category, cat);
  }
  std::vector<std::pair<int64_t, int64_t>> counts(max_category + 1, {0, 0});

  for (size_t i = 0; i < column.size(); ++i) {
    size_t cat = column[i];
    if (cat < counts.size()) {
      counts[cat].first += static_cast<int64_t>(targets[i]);
      counts[cat].second += 1 - static_cast<int64_t>(targets[i]);
    } else {
      // Handle out-of-bounds category (log or throw an error).
      VLog(0, "Category out of range: " +
                  std::to_string(
                      cat)); // Use 0 or options_.verbose if available here
    }
  }
  return counts;
}

} // namespace

using std::count;

void WoEEncoder::PopulateWoEMap(size_t column_index,
                                const std::vector<int> &column,
                                const std::vector<bool> &targets) {

  int max_category = 0;
  std::vector<std::pair<int64_t, int64_t>> counts =
      CalculateCategoryCounts(column, targets, max_category);

  std::vector<double> woe_values(counts.size(), options_.default_woe);

  // Calculate WOE for each category.
  woe_values.reserve(counts.size());
  for (size_t cat = 0; cat < counts.size(); ++cat) {
    if (counts[cat].first + counts[cat].second > 1) {
      woe_values[cat] = std::log(
          ((static_cast<double>(counts[cat].first) + options_.epsilon) /
           (static_cast<double>(total_pos_) + 2 * options_.epsilon)) /
          (((static_cast<double>(counts[cat].first + counts[cat].second) -
             static_cast<double>(counts[cat].first)) +
            options_.epsilon) /
           (static_cast<double>(total_neg_) + 2 * options_.epsilon)));
    }
  }
  woe_map_[static_cast<int>(column_index)] = woe_values;
}

void WoEEncoder::Fit(const std::vector<std::vector<int>> &features,
                     const std::vector<bool> &targets) {
  // Get total counts of positives and negatives.
  total_pos_ =
      static_cast<int64_t>(count(targets.begin(), targets.end(), true));
  total_neg_ = static_cast<int64_t>(targets.size() - total_pos_);
  VLog(options_.verbose, "Total positives: " + std::to_string(total_pos_));
  VLog(options_.verbose, "Total negatives: " + std::to_string(total_pos_));
  // There needs to be at least one positive and one negative.
  if (total_pos_ == 0 || total_neg_ == 0) {
    throw std::invalid_argument(
        "Binary target needs at least one positive and one negative");
  }

  // Initialize woe_map_ with the number of columns.
  woe_map_.clear();
  woe_map_.resize(features.size());

  // For each column, populate a map of WOE values.
  for (size_t i = 0; i < features.size(); ++i) {
    VLog(options_.verbose, "Fitting column " + std::to_string(i));
    PopulateWoEMap(i, features[static_cast<int>(i)], targets);
  }
}

std::vector<std::vector<double>>
WoEEncoder::Transform(const std::vector<std::vector<int>> &features) const {
  std::vector<std::vector<double>> transformed_features(features.size());

  for (size_t i = 0; i < features.size(); ++i) {
    // Check if the column exists (important for robustness).
    if (i < woe_map_.size()) { // Check if the column exists
      const std::vector<double> &woe_values = woe_map_[i];

      transformed_features[i].reserve(features[i].size());

      for (int cat : features[i]) {
        if (cat < woe_values.size()) {
          transformed_features[i].push_back(woe_values[cat]);
        } else {
          transformed_features[i].push_back(options_.default_woe);
        }
      }
    } else {
      VLog(options_.verbose,
           "Column not found in training data: " + std::to_string(i));
      // Handle missing column appropriately (e.g., fill with default woe or
      // throw an error).
      transformed_features[i].resize(
          features[i].size(), options_.default_woe); // Fill with default value
    }
  }
  return transformed_features;
}

} // namespace fast_woe_encoder