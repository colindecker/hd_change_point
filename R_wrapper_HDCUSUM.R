library(Rcpp)
library(RcppArmadillo)
#here data stands for a list of $K$ functional time-seres. Each list entry is a matrix. 
#the rows of the matrix correspond to time, and the columns to the series evaluated on a grid. 
sourceCpp("/Users/daviddecker/Documents/HDCUSUM/HDCUSUM.cpp")
fit_rivers<-cusum_bootstrap_functional_test(
  data,
  gamma = 0.05,
  B = 1000,
  seed = 0,
  q_search_lower = 0.5,
  q_search_upper = 2,
  verbose = TRUE,
  progress_interval = 120.0,
  progress_check_every = 25
)