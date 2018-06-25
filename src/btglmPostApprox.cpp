
#include <cstdlib>
#include <Eigen/Core>
#include <Rcpp.h>
#include <RcppEigen.h>
#include <random>
#include <vector>

#include "AdaM.h"
#include "formatSgdOutput.h"
#include "ThresholdParameters.h"
// #include "startingPoints.h"
// #include "tglmApproxPostModeSGD.h"


// -I /Library/Frameworks/R.framework/Versions/3.5/Resources/library/RcppEigen/include/ -std=c++14


// LMFixedLambda initializeBeta(const SEXP beta_, const SEXP lambda_) {
//   const Eigen::ArrayXd beta(Rcpp::as<Eigen::Map<Eigen::ArrayXd> >(beta_));
//   const double lambda(Rcpp::as<double>(lambda_));
//   LMFixedLambda out(beta, lambda);
//   return (out);
// };


// extern "C" SEXP lmGradient(const SEXP beta_, const SEXP X_, const SEXP y_,
// 		      const SEXP tauSq_, const SEXP lambda_) {
//   const Eigen::Map<Eigen::MatrixXd> X(Rcpp::as<Eigen::Map<Eigen::MatrixXd> >(X_));
//   const Eigen::Map<Eigen::VectorXd> y(Rcpp::as<Eigen::Map<Eigen::VectorXd> >(y_));
//   const double tauSq(Rcpp::as<double>(tauSq_));

//   LMFixedLambda beta = initializeBeta(beta_, lambda_);

//   Eigen::ArrayXd grad = Eigen::ArrayXd::Zero(beta.size());
//   for (int i = 0; i < X.rows(); i++)
//     grad += lmUnitGradient(beta, i, X, y, tauSq);
  
//   return (Rcpp::wrap(grad));
// };







// .CAll'able R interface
extern "C" SEXP btlmPostApprox(
  const SEXP X_,
  const SEXP y_,
  const SEXP beta_,
  const SEXP lambda_,
  const SEXP tauSqBeta_,
  const SEXP M_,
  // const SEXP include_,
  const SEXP batchSize_,
  const SEXP iterMax_,
  const SEXP eps_,
  const SEXP tol_,
  const SEXP learningRate_,
  const SEXP momentumDecay_,
  const SEXP velocityDecay_,
  const SEXP seed_
) {
  try {
    const Eigen::Map<Eigen::MatrixXd> X(Rcpp::as<Eigen::Map<Eigen::MatrixXd> >(X_));
    const Eigen::Map<Eigen::VectorXd> y(Rcpp::as<Eigen::Map<Eigen::VectorXd> >(y_));
    const double tauSqBeta(Rcpp::as<double>(tauSqBeta_));
    const double tol(Rcpp::as<double>(tol_));
    const int batchSize(Rcpp::as<int>(batchSize_));
    const int iterMax(Rcpp::as<int>(iterMax_));

    std::mt19937 rng(Rcpp::as<int>(seed_));
    
    // !!!!!!
    // const Rcpp::IntegerVector include(Rcpp::as<Rcpp::IntegerVector>(include));
    LMFixedLambda beta(
      Rcpp::as<Eigen::Map<Eigen::ArrayXd> >(beta_),
      Rcpp::as<double>(lambda_),
      Rcpp::as<double>(M_)
    );
    
    AdaM<Eigen::ArrayXd> sgd(
      Eigen::ArrayXd::Zero(beta.size() + 1),
      Rcpp::as<double>(learningRate_),
      Rcpp::as<double>(momentumDecay_),
      Rcpp::as<double>(velocityDecay_),
      Rcpp::as<double>(eps_)
    );

    std::vector<int> dataInd(y.size());
    for (int i = 0; i < y.size(); i++)
      dataInd[i] = i;
    
    
    while (!sgd.converged(tol) && sgd.iteration() < iterMax) {
      sgd.minibatchUpdate
	<LMFixedLambda, Eigen::ArrayXd,
	 const Eigen::MatrixXd, const Eigen::VectorXd, const double>
        (beta, lmUnitGradient, rng, batchSize, dataInd, X, y, tauSqBeta);
    };

    return (Rcpp::wrap(Rcpp::List::create(
		       Rcpp::Named("coefficients") = beta,
		       Rcpp::Named("lambda") = beta.lambda(),
		       Rcpp::Named("objective") =
		         beta.objective(X, y, tauSqBeta),
		       Rcpp::Named("convergance") =
		         formatSgdOutput(sgd, tol)
	   )));

  }
  catch (std::exception& _ex_) {
    forward_exception_to_r(_ex_);
  }
  catch (...) {
    ::Rf_error("C++ exception (unknown cause)");
  }
  return (R_NilValue);  // not reached
};




