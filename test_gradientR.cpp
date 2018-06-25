
#include <Rcpp.h>
#include <RcppEigen.h>
#include <Eigen/Core>

#include "ThresholdParameters.h"



extern "C" lmGradient(const SEXP X_, const SEXP y_,
		      const SEXP tauSq_, const SEXP lambda_) {
  const Eigen::Map<Eigen::MatrixXd> X(Rcpp::as<Eigen::Map<Eigen::MatrixXd> >(X_));
  const Eigen::Map<Eigen::VectorXd> y(Rcpp::as<Eigen::Map<Eigen::VectorXd> >(y_));
  const double tauSq(Rcpp::as<double>(tauSq_));
  
  LMFixedLambda beta(Rcpp::as<Map<Eigen::ArrayXd> >(beta_),
		     Rcpp::as<double>(lambda_)
		     );

  Eigen::ArrayXd grad = Eigen::ArrayXd::Zeros(beta.size());
  for (int i = 0; i < X.rows(); i++)
    grad += lmUnitGradient(beta, i, X, y, tauSq);
  
  return (Rcpp::wrap(grad));
};
