
#include <algorithm>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "ThresholdGLM.h"
#include "ThresholdParameters.h"



void updateTheta(LMFixedLambda &theta, Eigen::ArrayXd &delta) {
  const double M = theta.M();
  const double lambda = theta.lambda();
  theta -= delta.head(theta.size());
  if (theta.iteration() % 20 == 0)
    theta.setLambda(M * lambda /
		    (std::exp(delta.tail(1)(0)) * (M - lambda) + lambda));
  theta.update();
};


// template< typename T, typename S >
Eigen::ArrayXd lmUnitGradient(
  const LMFixedLambda &theta,
  const int &i,
  const Eigen::MatrixXd &X,
  const Eigen::VectorXd &y,
  const double &tauSq
  ) {
  const double mu = (theta._spar * X.row(i).transpose())(0);
  Eigen::ArrayXd grad(theta.size() + 1);
  grad.head(theta.size()) = (theta._deriv * X.row(i).transpose().array())
    + (theta / (tauSq * X.rows()));
  grad.coeffRef(theta._include) -= theta.coeffRef(theta._include) /
    (tauSq * X.rows());
  grad.tail(1) = (X.row(i).transpose() * theta._derivPsi.matrix())(0);
  grad *= (mu - y(i));  // multiply by negative residual
  return (grad);
};




void LMFixedLambda::computeDeriv() {
  const double scale = std::max(std::pow(10.0, -_iter / 100.0), 1e-6);
  _deriv = this->unaryExpr([&](const double &x) {
      return (ThresholdGLM::approxDThreshCauchy(x, _lambda, scale) * x); });
  _derivPsi = this->unaryExpr([&](const double &x) {
      return (ThresholdGLM::approxDPsiCauchy(_lambda, x, _M, scale) * x); });
  for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(_spar, 0);
       it; ++it)
    _deriv(it.index())++;
  _deriv.coeffRef(_include) = 1.0;
  _derivPsi.coeffRef(_include) = 0.0;
};






// Public Methods
// -------------------------------------------------------------------

// Setters
void LMFixedLambda::setLambda(const double &lambda) {
  _lambda = lambda;
};

void LMFixedLambda::setSparse() {
  _spar = this->matrix().sparseView(1, _lambda).transpose();
  _spar.coeffRef(0, _include) = this->coeffRef(_include);
};

void LMFixedLambda::update() {
  setSparse();
  computeDeriv();
  _iter++;
};


// Getters
int LMFixedLambda::iteration() const {
  return (_iter);
};

double LMFixedLambda::lambda() const {
  return (_lambda);
};

double LMFixedLambda::M() const {
  return (_M);
};






// Misc
Eigen::VectorXd LMFixedLambda::residuals(
  const Eigen::MatrixXd &X,
  const Eigen::VectorXd &y
  ) const {
  return (y - (_spar * X.transpose()).transpose());
};

double LMFixedLambda::objective(
  const Eigen::MatrixXd &X,
  const Eigen::VectorXd &y,
  const double &tauSq
  ) const {
  const Eigen::VectorXd resid = residuals(X, y);
  return (-0.5 * (resid.squaredNorm() +
		  this->matrix().squaredNorm() / tauSq)
	  );
};


