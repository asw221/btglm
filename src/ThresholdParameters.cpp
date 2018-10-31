
#include <algorithm>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Rcpp.h>
#include <random>

#include "AdaM.h"
#include "ThresholdGLM.h"
#include "ThresholdParameters.h"


using std::sqrt;
using std::pow;



void updateTheta(LMFixedLambda &theta, Eigen::ArrayXd &delta) {
  theta -= delta;
  // theta.update();
  theta.setSparse();
  theta.lambda(std::max(theta.lambda() * ThresholdGLM::_lambdaDecayRate_,
			ThresholdGLM::_minLambda_));
};


// void sgldUpdate(
//   LMFixedLambda &theta,
//   AdaM<Eigen::ArrayXd> &sgd,
//   const int &batchSize,
//   std::vector<int> &dataIndex,
//   double &learningScale,
//   double &acceptanceProbability,
//   const bool updateLearningScale,
//   const Eigen::MatrixXd &X,
//   const Eigen::VectorXd &y,
//   const double &residualPrecision,
//   const double &priorPrecision,
//   const double &metropolisTarget
// ) {
//   const double decay = 0.9, k = 0.01;
//   double gradScale = residualPrecision * dataIndex.size() / batchSize;
//   double lpCurrent, lpProposal, pAccept;
  
//   // Compute SGD update step -----------------------------------------
//   sgd.virtualMinibatch
//     <LMFixedLambda, Eigen::ArrayXd, const Eigen::MatrixXd&,
//      const Eigen::VectorXd&, const double&>
//     (theta, lmUnitGradient, ThresholdGLM::_rng_, batchSize,
//      dataIndex, X, y, priorPrecision);

//   Eigen::ArrayXd Mass = Eigen::ArrayXd::Constant(theta.size(),
//     theta._lambda * pow(std::log(theta.size()) / theta.size(), 2));
//   for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator
// 	 it(theta._spar, 0); it; ++it)
//     Mass.coeffRef(it.index()) = sqrt(1 / (sgd.velocity().coeffRef(it.index()) + 1e-8));
//   double eta = sgd.eta() * learningScale;
//   Eigen::ArrayXd delta = eta * Mass * sgd.momentum() +
//     sqrt(2 * eta / gradScale) * gaussianNoise(sqrt(Mass), ThresholdGLM::_rng_);

//   lpCurrent = theta.objective(X, y, priorPrecision);
//   theta -= delta;
//   theta.setSparse();
//   lpProposal = theta.objective(X, y, priorPrecision);
//   pAccept = std::min(std::exp(
//     residualPrecision * (lpProposal - lpCurrent)), 1.0);
//   acceptanceProbability = acceptanceProbability * decay +
//     pAccept * (1 - decay);
//   if (ThresholdGLM::_Uniform_(ThresholdGLM::_rng_) < pAccept)
//     theta.computeDeriv();
//   else {  // reject update
//     theta += delta;
//     theta.setSparse();
//   }
//   if (updateLearningScale) {
//     if (acceptanceProbability < metropolisTarget)
//       learningScale *= 1 - k;
//     else
//       learningScale *= 1 + k;
//   }
// };







Eigen::ArrayXd lmUnitGradient(
  const LMFixedLambda &theta,
  const int &i,
  const Eigen::MatrixXd &X,
  const Eigen::VectorXd &y,
  const double &priorPrecision
  ) {
  const double mu = (X.row(i) * theta._spar)(0);
  double j, k;  // loop vars
  Eigen::ArrayXd grad(theta.size());
  grad = (theta._deriv * X.row(i).transpose().array()) * (mu - y(i))
    + (priorPrecision * theta + theta._priorDeriv) / X.rows();
  for (j = 0; j < theta._include.size(); j++) {
    k = theta._include[j];
    grad.coeffRef(k) -= (priorPrecision * theta.coeffRef(k) +
       theta._priorDeriv.coeffRef(k)) / X.rows();
  }
  return (grad);
};




// void LMFixedLambda::computeDeriv() {
//   const double scale = ThresholdGLM::_epsilon_;
//   double threshDeriv = 0.0;
//   for (int i = 0; i < this->size(); i++) {  // loop slower if parallelized?
//     threshDeriv = ThresholdGLM::approxDThreshCauchy
//       (this->coeffRef(i), _lambda, scale);
//     _deriv.coeffRef(i) = _spar.coeffRef(i) + 
//   }
//   _deriv = this->unaryExpr([&](const double &x) {
//       return (ThresholdGLM::approxThresholdCauchy(x, _lambda, scale) +
// 	      ThresholdGLM::approxDThreshCauchy(x, _lambda, scale) * x); });
//   for (int i = 0; i < _include.size(); i++)
//     _deriv.coeffRef(_include[i]) = 1.0;
// };

void LMFixedLambda::computeDeriv() { };

void LMFixedLambda::update() { };




// Public Methods
// -------------------------------------------------------------------

// Setters
void LMFixedLambda::lambda(const double &lambda) {
  if (lambda > 0)
    _lambda = lambda;
};


void LMFixedLambda::setSparse() {
  const double scale = ThresholdGLM::_epsilon_;
  double activeCoeff, threshApprox, threshGradApprox;
  // Adjust _deriv, _priorDeriv, and _spar member data for all
  // coefficients in the model
  for (int i = 0; i < this->size(); i++) {
    // Comptute H(\beta_i) and h(\beta_i) = H'(\beta_i)
    // approximations to indicator function and its derivative
    activeCoeff = this->coeffRef(i);
    threshApprox = ThresholdGLM::approxThresholdCauchy(
      activeCoeff, _lambda, scale);
    threshGradApprox = ThresholdGLM::approxDThreshCauchy(
      activeCoeff, _lambda, scale);
    // Update stored derivative components
    _deriv.coeffRef(i) = threshApprox + threshGradApprox * activeCoeff;
    _priorDeriv.coeffRef(i) = 0.5 * _priorModelSizeScale * threshGradApprox;
    _spar(i) = threshApprox * activeCoeff;
  }
  // "Undo" some of the above steps for coefficients that are always
  // included in the model
  for (int i = 0; i < _include.size(); i++) {
    _deriv.coeffRef(_include[i]) = 1.0;
    _priorDeriv.coeffRef(_include[i]) = 0.0;
    _spar(_include[i]) = this->coeffRef(_include[i]);
  }
};


// void LMFixedLambda::update() {
//   setSparse();
//   // computeDeriv();
// };




// Getters

double LMFixedLambda::lambda() const {
  return (_lambda);
};






// Misc
Eigen::VectorXd LMFixedLambda::residuals(
  const Eigen::MatrixXd &X,
  const Eigen::VectorXd &y
) const {
  return (y - X * _spar);
};


double LMFixedLambda::objective(
  const Eigen::MatrixXd &X,
  const Eigen::VectorXd &y,
  const double &priorPrecision
) const {
  const double likelihood = 0.5 * residuals(X, y).squaredNorm();
  const double modelSize = (_spar.array() / *this).sum();
  const double prior = 0.5 * (priorPrecision *
    this->matrix().squaredNorm() + _priorModelSizeScale * modelSize);
  return (-(likelihood + prior));
};














// -------------------------------------------------------------------


// void updateTheta(LMFixedLambda &theta, Eigen::ArrayXd &delta) {
//   if (!theta._canRejectUpdates) {
//     theta -= delta;
//     theta.update();
//   }
//   else {
//     double lpProposal, pAccept;
//     const double lpCurrent = theta.objective();
//     theta -= delta;
//     theta.setSparse();
//     lpProposal = theta.objective();
//     pAccept = std::min(
//       std::exp(theta._postPrecision * (lpProposal - lpCurrent)), 1.0);
//     theta._rejectRate = theta._rejectDecay * theta._rejectRate +
//       (1 - theta._rejectDecay) * (1 - pAccept);
//     theta._iter++;
//     if (ThresholdGLM::_Uniform_(ThresholdGLM::_rng_) < pAccept)
//       theta.computeDeriv();
//     else {  // reject update
//       theta += delta;
//       theta.setSparse();
//     }
//     if (theta._canUpdateMHScales) {
//       const double k = 0.1;
//       const double rejectRate = theta._rejectRate /
// 	(1 - std::pow(theta._rejectDecay, theta._iter));
//       if (rejectRate > theta._mhTarget)
// 	theta._mhScale = 1 - k;
//       else
// 	theta._mhScale = 1 + k;
//     }
//   }
// };

