#pragma once

#include <string>
#include <utility>
#include <vector>

#include "matrix.hpp"
#include "random.hpp"

namespace nn::data {

/// Fast CSV reader (numeric only). Skips empty lines; `skip_header` drops the first line.
Matrix load_csv(const std::string& path, char delimiter = ',', bool skip_header = false);

/// Split columns: X = all except `label_col`, labels = that column. label_col < 0 counts from end.
std::pair<Matrix, Matrix> split_xy(const Matrix& m, int label_col = -1);

/// Map arbitrary label values (one column) to dense class indices 0..K-1 (first-seen order).
/// Returns the number of classes.
std::size_t encode_labels(Matrix& labels, std::vector<real>* classes = nullptr);

/// Class indices (one column) -> one-hot [rows x num_classes].
Matrix one_hot(const Matrix& labels, std::size_t num_classes);

/// Shuffle rows of X and Y together, then split off `test_fraction` rows.
struct Split { Matrix x_train, y_train, x_test, y_test; };
Split train_test_split(const Matrix& x, const Matrix& y, real test_fraction, Rng& rng);

/// Feature scaling fitted on training data, reusable for new data.
class MinMaxScaler {
public:
    void fit(const Matrix& x);
    void transform(Matrix& x) const;
    void fit_transform(Matrix& x) { fit(x); transform(x); }
    std::vector<real> min, range;
};

class StandardScaler {
public:
    void fit(const Matrix& x);
    void transform(Matrix& x) const;
    void fit_transform(Matrix& x) { fit(x); transform(x); }
    std::vector<real> mean, inv_std;
};

}  // namespace nn::data
