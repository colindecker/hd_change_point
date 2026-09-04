#include <RcppArmadillo.h>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>

using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]

// -----------------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------------

int empirical_quantile_index(int B, double prob) {
  int idx = static_cast<int>(std::ceil(prob * static_cast<double>(B))) - 1;
  return std::max(0, std::min(B - 1, idx));
}

double empirical_quantile_vec(const arma::vec& x, double prob) {
  arma::vec finite_x = x.elem(arma::find_finite(x));
  if (finite_x.n_elem == 0) return arma::datum::nan;

  arma::vec xs = arma::sort(finite_x);
  return xs(empirical_quantile_index(static_cast<int>(xs.n_elem), prob));
}

arma::cube list_to_cube(Rcpp::List X_list) {
  int K = X_list.size();
  if (K < 1) stop("X_list must have positive length.");

  Rcpp::NumericMatrix X0_r(X_list[0]);
  int n = X0_r.nrow();
  int T = X0_r.ncol();

  if (n < 4) stop("Each matrix in X_list must have at least 4 rows.");
  if (T < 2) stop("Each matrix in X_list must have at least 2 columns.");

  arma::cube X(n, T, K, arma::fill::zeros);
  for (int k = 0; k < K; k++) {
    Rcpp::NumericMatrix Xk_r(X_list[k]);
    if (Xk_r.nrow() != n || Xk_r.ncol() != T) {
      stop("All matrices in X_list must have the same dimensions.");
    }
    arma::mat Xk(Xk_r.begin(), n, T, false);
    X.slice(k) = Xk;
  }

  return X;
}

// -----------------------------------------------------------------------------
// Observed functional CUSUM and change-point estimates
// -----------------------------------------------------------------------------

arma::mat compute_cusum_matrix_fast(const arma::mat& X) {
  int n = X.n_rows;
  arma::mat prefix = arma::cumsum(X, 0);
  arma::rowvec total = prefix.row(n - 1);
  arma::vec frac = arma::regspace<arma::vec>(1, n - 1) / static_cast<double>(n);

  return (prefix.rows(0, n - 2) - frac * total) /
    std::sqrt(static_cast<double>(n));
}

arma::vec row_l2_norm_sq(const arma::mat& X) {
  return arma::sum(arma::square(X), 1) / static_cast<double>(X.n_cols);
}

// -----------------------------------------------------------------------------
// Article residual construction: tau = 0 and g = (q + r) / 2
// -----------------------------------------------------------------------------

arma::mat make_residual_prefix(
    const arma::mat& Xk,
    const arma::rowvec& mu_left,
    const arma::rowvec& mu_right,
    int N_L,
    int N_R
) {
  int n = Xk.n_rows;
  int T = Xk.n_cols;

  arma::mat resid(n, T, arma::fill::zeros);
  resid.rows(0, N_L - 1) = Xk.rows(0, N_L - 1);
  resid.rows(0, N_L - 1).each_row() -= mu_left;
  resid.rows(N_R, n - 1) = Xk.rows(N_R, n - 1);
  resid.rows(N_R, n - 1).each_row() -= mu_right;

  arma::mat prefix(n + 1, T, arma::fill::zeros);
  prefix.rows(1, n) = arma::cumsum(resid, 0);
  return prefix;
}

bool build_article_residual_prefixes(
    const arma::cube& X,
    const arma::ivec& change_point_estimates,
    int q,
    int r,
    std::vector<arma::mat>& residual_prefix_list
) {
  int n = X.n_rows;
  int K = X.n_slices;
  int h = q + r;
  double g = static_cast<double>(h) / 2.0;

  residual_prefix_list.resize(K);

  for (int k = 0; k < K; k++) {
    double omega = static_cast<double>(change_point_estimates(k));
    int L = static_cast<int>(std::floor((omega - g) / static_cast<double>(h)));
    int R = static_cast<int>(std::ceil((omega + g) / static_cast<double>(h)));
    int N_L = h * L;
    int N_R = h * R;

    if (N_L <= 0 || N_L > n || N_R < 0 || N_R >= n || N_L > N_R) {
      return false;
    }

    arma::mat Xk = X.slice(k);
    arma::rowvec mu_left = arma::mean(Xk.rows(0, N_L - 1), 0);
    arma::rowvec mu_right = arma::mean(Xk.rows(N_R, n - 1), 0);
    residual_prefix_list[k] =
      make_residual_prefix(Xk, mu_left, mu_right, N_L, N_R);
  }

  return true;
}

// -----------------------------------------------------------------------------
// Article multiplier bootstrap CUSUM
// -----------------------------------------------------------------------------

arma::mat multiplier_bridge_from_prefix(
    const arma::mat& prefix,
    const arma::vec& multipliers,
    int q,
    int r,
    int m
) {
  int n = prefix.n_rows - 1;
  int T = prefix.n_cols;
  int h = q + r;

  arma::mat partial_sum(n, T, arma::fill::zeros);

  // Only the q observations in each big block I_ell enter S_k^*(s).
  for (int ell = 0; ell < m; ell++) {
    int start = ell * h;
    int end = start + q;

    for (int s = 0; s < n; s++) {
      int upper = std::min(end, s + 1);
      if (upper > start) {
        partial_sum.row(s) +=
          multipliers(ell) * (prefix.row(upper) - prefix.row(start));
      }
    }
  }

  arma::rowvec final_sum = partial_sum.row(n - 1);
  arma::vec frac = arma::regspace<arma::vec>(1, n - 1) / static_cast<double>(n);
  arma::mat centered = partial_sum.rows(0, n - 2) - frac * final_sum;

  return centered /
    std::sqrt(static_cast<double>(m) * static_cast<double>(q));
}

// -----------------------------------------------------------------------------
// Article block-size selector
// -----------------------------------------------------------------------------

struct BlockSelection {
  arma::ivec Q;
  arma::ivec R;
  arma::ivec M;
  arma::vec bootstrap_quantiles;
  arma::vec rolling_sd;
  arma::mat bootstrap_statistics;  // B x G global maxima
  int selected_idx;                // zero-based index in Q
  int selected_rolling_idx;        // zero-based index in rolling_sd
};

BlockSelection select_block_size_article(
    const arma::cube& X,
    const arma::ivec& change_point_estimates,
    int B,
    double quantile_prob,
    double q_search_lower,
    double q_search_upper,
    bool verbose,
    double progress_interval,
    int progress_check_every
) {
  int n = X.n_rows;
  int K = X.n_slices;
  double n13 = std::pow(static_cast<double>(n), 1.0 / 3.0);

  int q_min = std::max(
    2,
    static_cast<int>(std::floor(q_search_lower * n13))
  );
  int q_max = std::min(
    static_cast<int>(std::floor(static_cast<double>(n) / 4.0)),
    static_cast<int>(std::ceil(q_search_upper * n13))
  );

  if (q_max < q_min) {
    stop("The candidate set Q_n is empty for this sample size/search region.");
  }

  int G = q_max - q_min + 1;
  if (G < 3) {
    stop("The candidate set Q_n contains fewer than 3 q values for this search region.");
  }

  arma::ivec Q(G), R(G), M(G);
  for (int j = 0; j < G; j++) {
    int q = q_min + j;
    int r = static_cast<int>(std::floor(std::sqrt(static_cast<double>(q))));
    Q(j) = q;
    R(j) = r;
    M(j) = n / (q + r);
  }

  arma::vec bootstrap_quantiles(G, arma::fill::value(arma::datum::nan));
  arma::mat bootstrap_statistics(B, G, arma::fill::value(arma::datum::nan));

  long long total = static_cast<long long>(G) * static_cast<long long>(B);
  long long completed = 0;
  int check_every = std::max(1, progress_check_every);
  double report_interval = std::max(1.0, progress_interval);
  auto last_report = std::chrono::steady_clock::now();

  if (verbose) {
    Rcpp::Rcout << "[article block selection bootstrap] started; total steps = "
                << total << "\n";
  }

  for (int j = 0; j < G; j++) {
    int q = Q(j);
    int r = R(j);
    int m = M(j);

    std::vector<arma::mat> residual_prefix_list;
    if (!build_article_residual_prefixes(
          X, change_point_estimates, q, r, residual_prefix_list)) {
      stop(
        "For at least one coordinate, tau = 0 gives an empty left or right "
        "trimmed sample for a candidate q. The article's residual construction "
        "is therefore not well defined for this data/candidate grid."
      );
    }

    arma::vec global_bootstrap(B, arma::fill::zeros);

    for (int b = 0; b < B; b++) {
      arma::vec multipliers = arma::randn(m);
      double global_b = 0.0;

      // The same multiplier vector is used for every coordinate.
      for (int k = 0; k < K; k++) {
        arma::mat C_star = multiplier_bridge_from_prefix(
          residual_prefix_list[k], multipliers, q, r, m
        );
        global_b = std::max(global_b, row_l2_norm_sq(C_star).max());
      }

      global_bootstrap(b) = global_b;
      completed++;

      if (completed % check_every == 0 || completed == total) {
        Rcpp::checkUserInterrupt();
        auto now = std::chrono::steady_clock::now();
        double seconds_since_report =
          std::chrono::duration<double>(now - last_report).count();

        if (verbose && (seconds_since_report >= report_interval || completed == total)) {
          int pct = static_cast<int>(
            std::round(100.0 * static_cast<double>(completed) /
                       static_cast<double>(total))
          );
          Rcpp::Rcout << "[article block selection bootstrap] "
                      << completed << "/" << total << " (" << pct << "%)\n";
          last_report = now;
        }
      }
    }

    bootstrap_statistics.col(j) = global_bootstrap;
    bootstrap_quantiles(j) = empirical_quantile_vec(global_bootstrap, quantile_prob);
  }

  int H = G - 2;
  arma::vec rolling_sd(H, arma::fill::value(arma::datum::nan));
  double best_sd = arma::datum::inf;
  int selected_rolling_idx = -1;

  for (int j = 0; j < H; j++) {
    arma::vec window = bootstrap_quantiles.subvec(j, j + 2);
    if (!window.is_finite()) continue;

    double sd_value = arma::stddev(window);
    rolling_sd(j) = sd_value;

    if (sd_value < best_sd) {
      best_sd = sd_value;
      selected_rolling_idx = j;
    }
  }

  if (selected_rolling_idx < 0) {
    stop(
      "No valid three-q stability window was available. With tau = 0, this can "
      "occur when an estimated change point is too close to a boundary for one "
      "or more candidate q values."
    );
  }

  BlockSelection out;
  out.Q = Q;
  out.R = R;
  out.M = M;
  out.bootstrap_quantiles = bootstrap_quantiles;
  out.rolling_sd = rolling_sd;
  out.bootstrap_statistics = bootstrap_statistics;
  out.selected_rolling_idx = selected_rolling_idx;
  out.selected_idx = selected_rolling_idx + 1;
  return out;
}

// -----------------------------------------------------------------------------
// Main exported function
// -----------------------------------------------------------------------------

// [[Rcpp::export]]
Rcpp::List cusum_bootstrap_functional_test(
    Rcpp::List X_list,
    double gamma = 0.05,
    int B = 1000,
    int seed = 1,
    double q_search_lower = 0.5,
    double q_search_upper = 1.5,
    bool verbose = true,
    double progress_interval = 120.0,
    int progress_check_every = 25
) {
  Function set_seed("set.seed");
  set_seed(seed);

  if (gamma <= 0.0 || gamma >= 1.0) stop("gamma must be in (0, 1).");
  if (B <= 1) stop("B must be larger than 1.");

  if (!std::isfinite(q_search_lower) || !std::isfinite(q_search_upper)) {
    stop("q_search_lower and q_search_upper must be finite.");
  }
  if (q_search_lower <= 0.0 || q_search_upper <= 0.0) {
    stop("q_search_lower and q_search_upper must be positive.");
  }
  if (q_search_lower > q_search_upper) {
    stop("q_search_lower must not exceed q_search_upper.");
  }

  arma::cube X = list_to_cube(X_list);
  int K = X.n_slices;
  double quantile_prob = 1.0 - gamma;

  // Observed coordinatewise CUSUM maxima and change-point estimates.
  arma::ivec change_point_estimates(K);
  arma::vec coordinate_test_statistics(K);

  for (int k = 0; k < K; k++) {
    arma::vec normsq = row_l2_norm_sq(compute_cusum_matrix_fast(X.slice(k)));
    arma::uword argmax = normsq.index_max();
    change_point_estimates(k) = static_cast<int>(argmax) + 1;
    coordinate_test_statistics(k) = normsq(argmax);
  }

  double global_test_statistic = coordinate_test_statistics.max();

  // Select q using the article's three-q rolling-SD stability rule.
  BlockSelection sel = select_block_size_article(
    X,
    change_point_estimates,
    B,
    quantile_prob,
    q_search_lower,
    q_search_upper,
    verbose,
    progress_interval,
    progress_check_every
  );

  int q = sel.Q(sel.selected_idx);
  int r = sel.R(sel.selected_idx);
  int m = sel.M(sel.selected_idx);
  double g = static_cast<double>(q + r) / 2.0;
  double selected_rolling_sd = sel.rolling_sd(sel.selected_rolling_idx);

  arma::vec global_bootstrap_statistics =
    sel.bootstrap_statistics.col(sel.selected_idx);
  double global_critical_value = sel.bootstrap_quantiles(sel.selected_idx);
  bool global_rejection = global_test_statistic > global_critical_value;

  // Single-step maxT adjusted coordinate p-values.
  arma::vec coordinate_maxT_p_values(K);
  for (int k = 0; k < K; k++) {
    int exceedances = static_cast<int>(
      arma::accu(global_bootstrap_statistics >= coordinate_test_statistics(k))
    );
    coordinate_maxT_p_values(k) =
      static_cast<double>(1 + exceedances) / static_cast<double>(B + 1);
  }

  arma::uvec selected_u = arma::find(
    coordinate_test_statistics > global_critical_value
  );
  arma::ivec selected_coordinates(selected_u.n_elem);
  for (arma::uword j = 0; j < selected_u.n_elem; j++) {
    selected_coordinates(j) = static_cast<int>(selected_u(j)) + 1;
  }

  int global_exceedances = static_cast<int>(
    arma::accu(global_bootstrap_statistics >= global_test_statistic)
  );
  double global_p_value =
    static_cast<double>(1 + global_exceedances) / static_cast<double>(B + 1);

  // Compact coordinate results table.
  Rcpp::IntegerVector coordinate_index = Rcpp::seq(1, K);
  Rcpp::DataFrame coordinate_results = Rcpp::DataFrame::create(
    Named("coordinate") = coordinate_index,
    Named("statistic") = coordinate_test_statistics,
    Named("maxT_p_value") = coordinate_maxT_p_values
  );

  // Full rolling-SD diagnostics retained by agreement.
  int H = sel.rolling_sd.n_elem;
  Rcpp::LogicalVector selected_flags(H, false);
  selected_flags[sel.selected_rolling_idx] = true;

  Rcpp::DataFrame rolling_sd_table = Rcpp::DataFrame::create(
    Named("q_left") = sel.Q.subvec(0, H - 1),
    Named("q_center") = sel.Q.subvec(1, H),
    Named("q_right") = sel.Q.subvec(2, H + 1),
    Named("rolling_sd") = sel.rolling_sd,
    Named("selected") = selected_flags
  );

  arma::vec g_column =
    arma::conv_to<arma::vec>::from(sel.Q + sel.R) / 2.0;

  Rcpp::DataFrame block_grid = Rcpp::DataFrame::create(
    Named("q") = sel.Q,
    Named("r") = sel.R,
    Named("m") = sel.M,
    Named("g") = g_column,
    Named("bootstrap_quantile") = sel.bootstrap_quantiles
  );

  // Agreed compact 14-item return object.
  return Rcpp::List::create(
    Named("change_point_estimates") = change_point_estimates,
    Named("coordinate_results") = coordinate_results,
    Named("selected_coordinates") = selected_coordinates,
    Named("global_test_statistic") = global_test_statistic,
    Named("global_p_value") = global_p_value,
    Named("global_critical_value") = global_critical_value,
    Named("global_rejection") = global_rejection,
    Named("selected_q") = q,
    Named("selected_r") = r,
    Named("selected_m") = m,
    Named("selected_g") = g,
    Named("selected_rolling_sd") = selected_rolling_sd,
    Named("rolling_sd_table") = rolling_sd_table,
    Named("block_grid") = block_grid
  );
}
