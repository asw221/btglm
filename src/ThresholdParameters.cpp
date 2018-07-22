
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
  theta.update();
}


void sgldUpdate(
  LMFixedLambda &theta,
  AdaM<Eigen::ArrayXd> &sgd,
  const int &batchSize,
  std::vector<int> &dataIndex,
  double &learningScale,
  double &acceptanceProbability,
  const bool updateLearningScale,
  const Eigen::MatrixXd &X,
  const Eigen::VectorXd &y,
  const double &residualPrecision,
  const double &priorPrecision,
  const double &metropolisTarget
) {
  const double decay = 0.9, k = 0.01;
  double gradScale = residualPrecision * dataIndex.size() / batchSize;
  double lpCurrent, lpProposal, pAccept;
  
  // Compute SGD update step -----------------------------------------
  sgd.virtualMinibatch
    <LMFixedLambda, Eigen::ArrayXd, const Eigen::MatrixXd&,
     const Eigen::VectorXd&, const double&>
    (theta, lmUnitGradient, ThresholdGLM::_rng_, batchSize,
     dataIndex, X, y, priorPrecision);

  Eigen::ArrayXd Mass = Eigen::ArrayXd::Constant(theta.size(),
    theta._lambda * pow(std::log(theta.size()) / theta.size(), 2));
  for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator
	 it(theta._spar, 0); it; ++it)
    Mass.coeffRef(it.index()) = sqrt(1 / (sgd.velocity().coeffRef(it.index()) + 1e-8));
  double eta = sgd.eta() * learningScale;
  Eigen::ArrayXd delta = eta * Mass * sgd.momentum() +
    sqrt(2 * eta / gradScale) * gaussianNoise(sqrt(Mass), ThresholdGLM::_rng_);

  lpCurrent = theta.objective(X, y, priorPrecision);
  theta -= delta;
  theta.setSparse();
  lpProposal = theta.objective(X, y, priorPrecision);
  pAccept = std::min(std::exp(
    residualPrecision * (lpProposal - lpCurrent)), 1.0);
  acceptanceProbability = acceptanceProbability * decay +
    pAccept * (1 - decay);
  if (ThresholdGLM::_Uniform_(ThresholdGLM::_rng_) < pAccept)
    theta.computeDeriv();
  else {  // reject update
    theta += delta;
    theta.setSparse();
  }
  if (updateLearningScale) {
    if (acceptanceProbability < metropolisTarget)
      learningScale *= 1 - k;
    else
      learningScale *= 1 + k;
  }
};





Eigen::SparseMatrix<double, Eigen::RowMajor> activeCoefGradient(
  const LMFixedLambda &theta,
  const Eigen::MatrixXd &X,
  const Eigen::VectorXd &y,
  const double &priorPrecision
) {
  // do some fancy shuffling with the index values to
  // avoid using SparseMatrix::coeffRef()
  const double scale = 1e-6;
  double gradThresh, logPriorGrad;
  Eigen::SparseMatrix<double, Eigen::RowMajor> grad = theta._spar;
  Eigen::VectorXd residuals = theta.residuals(X, y);
  int i = 0;
  for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator
	 it(grad, 0); it; ++it) {
    logPriorGrad = it.value() * priorPrecision;
    gradThresh = ThresholdGLM::approxDThreshCauchy
      (it.value(), theta._lambda, scale) * it.value();
    it.valueRef() = -(1 + gradThresh) * X.row(it.index()) * residuals;
    if (it.index() == theta._include[i])
      i++;
    else
      it.valueRef() += logPriorGrad;
  }
  return (grad);
}



Eigen::ArrayXd lmUnitGradient(
  const LMFixedLambda &theta,
  const int &i,
  const Eigen::MatrixXd &X,
  const Eigen::VectorXd &y,
  const double &priorPrecision
  ) {
  const double mu = (theta._spar * X.col(i))(0);
  Eigen::ArrayXd grad(theta.size());
  grad = (theta._deriv * X.col(i).array()) * (mu - y(i))
    + (theta * priorPrecision / X.cols());
  for (int i = 0; i < theta._include.size(); i++)
    grad.coeffRef(theta._include[i]) -= theta.coeffRef(theta._include[i]) *
      priorPrecision / X.cols();
  return (grad);
};




void LMFixedLambda::computeDeriv() {
  const double scale = 1e-6;
  _deriv = this->unaryExpr([&](const double &x) {
      return (ThresholdGLM::approxDThreshCauchy(x, _lambda, scale) * x); });
  for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(_spar, 0);
       it; ++it)
    _deriv(it.index())++;
  for (int i = 0; i < _include.size(); i++)
    _deriv.coeffRef(_include[i]) = 1.0;
};






// Public Methods
// -------------------------------------------------------------------

// Setters
void LMFixedLambda::lambda(const double &lambda) {
  _lambda = lambda;
};


void LMFixedLambda::setSparse() {
  _spar = this->matrix().sparseView(1, _lambda).transpose();
  for (int i = 0; i < _include.size(); i++)
    _spar.coeffRef(0, _include[i]) = this->coeffRef(_include[i]);
};

void LMFixedLambda::update() {
  setSparse();
  computeDeriv();
};




// Getters
int LMFixedLambda::nonZeros() const {
  return (_spar.nonZeros());
};

double LMFixedLambda::lambda() const {
  return (_lambda);
};

double LMFixedLambda::lambdaMax() const {
  return (_M);
};


double LMFixedLambda::minSparseCoeff() const {
  double result = _M;
  double val;
  for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(_spar, 0);
       it; ++it) {
    val = std::abs(it.value());
    if (val < result)
      result = val;
  }
  return (result);
};





// Misc
Eigen::VectorXd LMFixedLambda::residuals(
  const Eigen::MatrixXd &X,
  const Eigen::VectorXd &y
) const {
  return (y - (_spar * X).transpose());
};


double LMFixedLambda::objective(
  const Eigen::MatrixXd &X,
  const Eigen::VectorXd &y,
  const double &priorPrecision
) const {
  const double likelihood = 0.5 * residuals(X, y).squaredNorm();
  const double prior = 0.5 * priorPrecision * this->matrix().squaredNorm();
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

