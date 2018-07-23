
// #define EIGEN_NO_DEBUG
#include <cstdlib>
#include <Eigen/Core>
#include <Rcpp.h>
#include <RcppEigen.h>
#include <random>
#include <vector>

#include "AdaM.h"
#include "RandomWalk.h"
#include "ThresholdGLM.h"
#include "ThresholdParameters.h"

#include "formatSgdOutput.h"


// -I /Library/Frameworks/R.framework/Versions/3.5/Resources/library/RcppEigen/include/ -std=c++14





extern "C" SEXP btlm(
  const SEXP X_,
  const SEXP y_,
  const SEXP beta_,
  const SEXP lambda_,
  const SEXP tauSqBeta_,
  const SEXP M_,
  const SEXP include_,
  const SEXP batchSize_,
  const SEXP nSave_,
  const SEXP thin_,
  const SEXP burnin_,
  const SEXP iterMaxSgd_,
  const SEXP eps_,
  const SEXP tol_,
  const SEXP learningRate_,
  const SEXP momentumDecay_,
  const SEXP velocityDecay_,
  const SEXP metropolisTarget_,
  const SEXP seed_
) {
  try {
    const Eigen::Map<Eigen::MatrixXd>
      X(Rcpp::as<Eigen::Map<Eigen::MatrixXd> >(X_));
    const Eigen::Map<Eigen::VectorXd>
      y(Rcpp::as<Eigen::Map<Eigen::VectorXd> >(y_));
    const double priorBetaPrecision(1 / Rcpp::as<double>(tauSqBeta_));
    const double tol(Rcpp::as<double>(tol_));
    const double priorPrecShape = 0.1, priorPrecRate = 0.1;
    const double postPrecShape = 0.5 * y.size() + priorPrecShape;
    const int batchSize(Rcpp::as<int>(batchSize_));
    const int nSave(Rcpp::as<int>(nSave_));
    const int thin(Rcpp::as<int>(thin_));
    const int burnin(Rcpp::as<int>(burnin_));
    const int iterMaxSgd(Rcpp::as<int>(iterMaxSgd_));
    const Rcpp::IntegerVector include(include_);
    const double metropolisTarget(Rcpp::as<double>(metropolisTarget_));

    bool updateStepSize = false;
    
    double postPrecRate, residualPrecision;
    double stepSize = 1.0;
    double acceptProb = 0.0;
    int saveCount = 0, mcmcIter = 0;
    std::gamma_distribution<double> _Gamma_(0.1, 0.1);

    ThresholdGLM::_rng_.seed(Rcpp::as<int>(seed_));

    
    LMFixedLambda beta(
      Rcpp::as<Eigen::Map<Eigen::ArrayXd> >(beta_),
      Rcpp::as<double>(lambda_),
      Rcpp::as<double>(M_),
      include  //, tauSqBeta, X, y
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

    // Posterior sample storage
    Eigen::MatrixXd coefSamples(nSave, beta.size());
    Eigen::VectorXd sigmaSamples(nSave);

    while (!sgd.converged(tol) && sgd.iteration() < iterMaxSgd) {
      sgd.minibatchUpdate
    	<LMFixedLambda, Eigen::ArrayXd,
    	 const Eigen::MatrixXd&, const Eigen::VectorXd&, const double&>
        (beta, lmUnitGradient, ThresholdGLM::_rng_, batchSize,
    	 dataInd, X, y, priorBetaPrecision);
    };

    // sgd.clearHistory();
    // sgd.toggleRMSprop(true);
    while (saveCount < nSave) {
      // Adjust adaptive step size settings
      if (mcmcIter == 50)
	updateStepSize = true;
      if (mcmcIter == burnin)
	updateStepSize = false;
      
      // Update residual precision -- full conditional
      postPrecRate = -beta.objective(X, y, priorBetaPrecision) + priorPrecRate;
      _Gamma_.param(std::gamma_distribution<double>
		    ::param_type(postPrecShape, 1 / postPrecRate));
      residualPrecision = _Gamma_(ThresholdGLM::_rng_);

      // Update regression coefficients -- SGLD + metropolis correction
      sgldUpdate(beta, sgd, batchSize, dataInd, stepSize, acceptProb,
		 updateStepSize, X, y, residualPrecision,
		 priorBetaPrecision, metropolisTarget
		 );

      if (mcmcIter % thin == 0 && mcmcIter >= burnin) {
	coefSamples.row(saveCount) = beta;
	sigmaSamples(saveCount) = std::sqrt(1 / residualPrecision);
	saveCount++;
      }
      // if (mcmcIter < burnin)
      // 	sgd.eta(sgd.eta() * beta.updateScale());
	// Rcpp::Rcout << "Iter " << saveCount << "; " << beta.lambda() << "\n";
      mcmcIter++;
    };
    
    return (Rcpp::wrap(Rcpp::List::create(
		       Rcpp::Named("acceptanceRate") = acceptProb,
		       Rcpp::Named("coefficients") = coefSamples,
		       Rcpp::Named("eta") = sgd.eta() * stepSize,
		       Rcpp::Named("include") = include,
		       Rcpp::Named("lambda") = beta.lambda(),
		       Rcpp::Named("N") = X.cols(),
		       Rcpp::Named("p") = X.rows(),
		       Rcpp::Named("sigma") = sigmaSamples
		       // Rcpp::Named("objective") =
		       //   beta.objective(X, y, tauSqBeta)
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













// .CAll'able R interface
extern "C" SEXP btlmPostApprox(
  const SEXP X_,
  const SEXP y_,
  const SEXP beta_,
  const SEXP lambda_,
  const SEXP tauSqBeta_,
  const SEXP M_,
  const SEXP include_,
  const SEXP batchSize_,
  const SEXP iterMaxSgd_,
  const SEXP eps_,
  const SEXP tol_,
  const SEXP learningRate_,
  const SEXP momentumDecay_,
  const SEXP velocityDecay_,
  const SEXP seed_
) {
  try {
    const Eigen::Map<Eigen::MatrixXd>
      X(Rcpp::as<Eigen::Map<Eigen::MatrixXd> >(X_));
    const Eigen::Map<Eigen::VectorXd>
      y(Rcpp::as<Eigen::Map<Eigen::VectorXd> >(y_));
    const double priorBetaPrecision(1 / Rcpp::as<double>(tauSqBeta_));
    const double tol(Rcpp::as<double>(tol_));
    const double priorPrecShape = 0.1, priorPrecRate = 0.1;
    const double postPrecShape = 0.5 * y.size() + priorPrecShape;
    const int batchSize(Rcpp::as<int>(batchSize_));
    const int iterMaxSgd(Rcpp::as<int>(iterMaxSgd_));
    const Rcpp::IntegerVector include(include_);

    // std::mt19937 rng(Rcpp::as<int>(seed_));
    ThresholdGLM::_rng_.seed(Rcpp::as<int>(seed_));
    double precision, postPrecRate;
    
    
    LMFixedLambda beta(
      Rcpp::as<Eigen::Map<Eigen::ArrayXd> >(beta_),
      Rcpp::as<double>(lambda_),
      Rcpp::as<double>(M_),
      include  // , tauSqBeta, X, y
    );
    
    AdaM<Eigen::ArrayXd> sgd(
      Eigen::ArrayXd::Zero(beta.size()),
      Rcpp::as<double>(learningRate_),
      Rcpp::as<double>(momentumDecay_),
      Rcpp::as<double>(velocityDecay_),
      Rcpp::as<double>(eps_)
    );

    std::vector<int> dataInd(y.size());
    for (int i = 0; i < y.size(); i++)
      dataInd[i] = i;

    
    while (!sgd.converged(tol) && sgd.iteration() < iterMaxSgd) {
      sgd.minibatchUpdate
	<LMFixedLambda, Eigen::ArrayXd,
	 const Eigen::MatrixXd&, const Eigen::VectorXd&, const double&>
        (beta, lmUnitGradient, ThresholdGLM::_rng_, batchSize,
	 dataInd, X, y, priorBetaPrecision);
    };
    postPrecRate = -beta.objective(X, y, priorBetaPrecision) + priorPrecRate;
    precision = postPrecShape / postPrecRate;
    
    return (Rcpp::wrap(Rcpp::List::create(
		       Rcpp::Named("coefficients") = beta,
		       Rcpp::Named("convergence") =
  		         formatSgdOutput(sgd, tol),
		       Rcpp::Named("include") = include,
		       Rcpp::Named("lambda") = beta.lambda(),
		       Rcpp::Named("N") = X.cols(),
		       Rcpp::Named("objective") = -postPrecRate + priorPrecRate,
		       Rcpp::Named("p") = X.rows(),
		       Rcpp::Named("sigma") = std::pow(precision, -0.5)
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







// extern "C" SEXP btglmPostApprox(
//   const SEXP X_,
//   const SEXP y_,
//   const SEXP beta_,
//   const SEXP lambda_,
//   const SEXP tauSqBeta_,
//   const SEXP M_,
//   const SEXP include_,
//   const SEXP batchSize_,
//   const SEXP iterMaxSgd_,
//   const SEXP eps_,
//   const SEXP tol_,
//   const SEXP learningRate_,
//   const SEXP momentumDecay_,
//   const SEXP velocityDecay_,
//   const SEXP seed_
// ) {
//   try {
//     const Eigen::Map<Eigen::MatrixXd>
//       X(Rcpp::as<Eigen::Map<Eigen::MatrixXd> >(X_));
//     const Eigen::Map<Eigen::VectorXd>
//       y(Rcpp::as<Eigen::Map<Eigen::VectorXd> >(y_));
//     const double priorBetaPrecision(1 / Rcpp::as<double>(tauSqBeta_));
//     const double tol(Rcpp::as<double>(tol_));
//     // const double priorPrecShape = 0.1, priorPrecRate = 0.1;
//     // const double postPrecShape = 0.5 * y.size() + priorPrecShape;
//     const int batchSize(Rcpp::as<int>(batchSize_));
//     const int iterMaxSgd(Rcpp::as<int>(iterMaxSgd_));
//     const Rcpp::IntegerVector include(include_);

    
//     ThresholdGLM::_rng_.seed(Rcpp::as<int>(seed_));
    
//     GLMFixedLambda<ThresholdGLM::logit> beta(
//       Rcpp::as<Eigen::Map<Eigen::ArrayXd> >(beta_),
//       Rcpp::as<double>(lambda_),
//       Rcpp::as<double>(M_),
//       include  // , tauSqBeta, X, y
//     );
    
//     AdaM<Eigen::ArrayXd> sgd(
//       Eigen::ArrayXd::Zero(beta.size()),
//       Rcpp::as<double>(learningRate_),
//       Rcpp::as<double>(momentumDecay_),
//       Rcpp::as<double>(velocityDecay_),
//       Rcpp::as<double>(eps_)
//     );

//     std::vector<int> dataInd(y.size());
//     for (int i = 0; i < y.size(); i++)
//       dataInd[i] = i;

    
//     while (!sgd.converged(tol) && sgd.iteration() < iterMaxSgd) {
//       sgd.minibatchUpdate
// 	<GLMFixedLambda<ThresholdGLM::logit>, Eigen::ArrayXd,
// 	 const Eigen::MatrixXd&, const Eigen::VectorXd&, const double&>
//         (beta, glmUnitGradient<ThresholdGLM::logit>,
// 	 ThresholdGLM::_rng_, batchSize,
// 	 dataInd, X, y, priorBetaPrecision);
//     };
    
//     return (Rcpp::wrap(Rcpp::List::create(
// 		       Rcpp::Named("coefficients") = beta,
// 		       Rcpp::Named("convergence") =
//   		         formatSgdOutput(sgd, tol),
// 		       Rcpp::Named("include") = include,
// 		       Rcpp::Named("lambda") = beta.lambda(),
// 		       Rcpp::Named("N") = X.cols(),
// 		       Rcpp::Named("objective") = beta.objective(X, y, priorBetaPrecision),
// 		       Rcpp::Named("p") = X.rows()
// 	   )));

//   }
//   catch (std::exception& _ex_) {
//     forward_exception_to_r(_ex_);
//   }
//   catch (...) {
//     ::Rf_error("C++ exception (unknown cause)");
//   }
//   return (R_NilValue);  // not reached
// };








// extern "C" SEXP btglm(
//   const SEXP X_,
//   const SEXP y_,
//   const SEXP beta_,
//   const SEXP lambda_,
//   const SEXP tauSqBeta_,
//   const SEXP M_,
//   const SEXP include_,
//   const SEXP batchSize_,
//   const SEXP nSave_,
//   const SEXP thin_,
//   const SEXP burnin_,
//   const SEXP iterMaxSgd_,
//   const SEXP eps_,
//   const SEXP tol_,
//   const SEXP learningRate_,
//   const SEXP momentumDecay_,
//   const SEXP velocityDecay_,
//   const SEXP metropolisTarget_,
//   const SEXP seed_
// ) {
//   try {
//     const Eigen::Map<Eigen::MatrixXd>
//       X(Rcpp::as<Eigen::Map<Eigen::MatrixXd> >(X_));
//     const Eigen::Map<Eigen::VectorXd>
//       y(Rcpp::as<Eigen::Map<Eigen::VectorXd> >(y_));
//     const double priorBetaPrecision(1 / Rcpp::as<double>(tauSqBeta_));
//     const double tol(Rcpp::as<double>(tol_));
//     // const double priorPrecShape = 0.1, priorPrecRate = 0.1;
//     // const double postPrecShape = 0.5 * y.size() + priorPrecShape;
//     const int batchSize(Rcpp::as<int>(batchSize_));
//     const int nSave(Rcpp::as<int>(nSave_));
//     const int thin(Rcpp::as<int>(thin_));
//     const int burnin(Rcpp::as<int>(burnin_));
//     const int iterMaxSgd(Rcpp::as<int>(iterMaxSgd_));
//     const Rcpp::IntegerVector include(include_);
//     const double metropolisTarget(Rcpp::as<double>(metropolisTarget_));

//     bool updateStepSize = false;
    
//     double stepSize = 1.0;
//     double acceptProb = 0.0;
//     int saveCount = 0, mcmcIter = 0;

//     ThresholdGLM::_rng_.seed(Rcpp::as<int>(seed_));

    
//     GLMFixedLambda<ThresholdGLM::logit> beta(
//       Rcpp::as<Eigen::Map<Eigen::ArrayXd> >(beta_),
//       Rcpp::as<double>(lambda_),
//       Rcpp::as<double>(M_),
//       include  //, tauSqBeta, X, y
//     );
    
//     AdaM<Eigen::ArrayXd> sgd(
//       Eigen::ArrayXd::Zero(beta.size() + 1),
//       Rcpp::as<double>(learningRate_),
//       Rcpp::as<double>(momentumDecay_),
//       Rcpp::as<double>(velocityDecay_),
//       Rcpp::as<double>(eps_)
//     );

    
//     std::vector<int> dataInd(y.size());
//     for (int i = 0; i < y.size(); i++)
//       dataInd[i] = i;

//     // Posterior sample storage
//     Eigen::MatrixXd coefSamples(nSave, beta.size());

//     while (!sgd.converged(tol) && sgd.iteration() < iterMaxSgd) {
//       sgd.minibatchUpdate
//     	<GLMFixedLambda<ThresholdGLM::logit>, Eigen::ArrayXd,
//     	 const Eigen::MatrixXd&, const Eigen::VectorXd&, const double&>
//         (beta, glmUnitGradient<ThresholdGLM::logit>, ThresholdGLM::_rng_,
// 	 batchSize, dataInd, X, y, priorBetaPrecision);
//     };

//     // sgd.clearHistory();
//     // sgd.toggleRMSprop(true);
//     while (saveCount < nSave) {
//       // Adjust adaptive step size settings
//       if (mcmcIter == 50)
// 	updateStepSize = true;
//       if (mcmcIter == burnin)
// 	updateStepSize = false;

//       // Update regression coefficients -- SGLD + metropolis correction
//       beta.sgldUpdate
// 	(sgd, batchSize, dataInd, stepSize, acceptProb,
// 	 updateStepSize, X, y, priorBetaPrecision, metropolisTarget);

//       if (mcmcIter % thin == 0 && mcmcIter >= burnin) {
// 	coefSamples.row(saveCount) = beta;
// 	saveCount++;
//       }
//       mcmcIter++;
//     };
    
//     return (Rcpp::wrap(Rcpp::List::create(
// 		       Rcpp::Named("acceptanceRate") = acceptProb,
// 		       Rcpp::Named("coefficients") = coefSamples,
// 		       Rcpp::Named("eta") = sgd.eta() * stepSize,
// 		       Rcpp::Named("include") = include,
// 		       Rcpp::Named("lambda") = beta.lambda(),
// 		       Rcpp::Named("N") = X.cols(),
// 		       Rcpp::Named("p") = X.rows()
// 		       // Rcpp::Named("objective") =
// 		       //   beta.objective(X, y, tauSqBeta)
// 	   )));

//   }
//   catch (std::exception& _ex_) {
//     forward_exception_to_r(_ex_);
//   }
//   catch (...) {
//     ::Rf_error("C++ exception (unknown cause)");
//   }
//   return (R_NilValue);  // not reached
// };










