
#include <algorithm>
#include <cmath>
#include <random>

#include "ThresholdGLM.h"



std::mt19937 ThresholdGLM::_rng_;
std::uniform_real_distribution<double> ThresholdGLM::_Uniform_(0.0, 1.0);


double ThresholdGLM::_epsilon_ = 0.1;
double ThresholdGLM::_lambdaDecayRate_ = std::pow(0.95, 0.005);
double ThresholdGLM::_minLambda_ = 0.35;
// Threshold approximation function scale

// double ThresholdGLM::_dthreshScale_ = 1e-6;



double ThresholdGLM::approxThresholdCauchy(
  const double &theta,
  const double &lambda,
  const double &eps
) {
  return (1 + ThresholdGLM::pcauchy(theta, lambda, eps) -
	  ThresholdGLM::pcauchy(theta, -lambda, eps));
};



double ThresholdGLM::approxDThreshCauchy(
  const double &theta,
  const double &lambda,
  const double &eps
) {
  return (ThresholdGLM::dcauchy(theta, lambda, eps) -
	  ThresholdGLM::dcauchy(theta, -lambda, eps));
};

double ThresholdGLM::approxDPsiCauchy(  // psi = ln(lambda) - ln(M - lambda)
				      const double &lambda,
				      const double &theta,
				      const double &M,
				      const double &eps
					) {
  return (-lambda * (M - lambda) / M *
	  (ThresholdGLM::dcauchy(theta, lambda, eps) +
	   ThresholdGLM::dcauchy(theta, -lambda, eps)));
};





// glmLink class definitions -----------------------------------------
// link = identity by default

double ThresholdGLM::glmLink::operator()(const double &x) const {
  return x;
};

Eigen::ArrayXd ThresholdGLM::glmLink::operator()(const Eigen::ArrayXd &ary) const {
  return ary;
};

double ThresholdGLM::glmLink::inverse(const double &x) const {
  return x;
};

Eigen::ArrayXd ThresholdGLM::glmLink::inverse(const Eigen::ArrayXd &ary) const {
  return ary;
};






double ThresholdGLM::logit::operator()(const double &x) const {
  return -std::log(1 - 1 / x);
};

Eigen::ArrayXd ThresholdGLM::logit::operator()(const Eigen::ArrayXd &ary) const {
  return ary.unaryExpr([&](const double &x)
		       { return this->operator()(x); });
};

double ThresholdGLM::logit::inverse(const double &x) const {
  return 1 / (1 + std::exp(-x));
};

Eigen::ArrayXd ThresholdGLM::logit::inverse(const Eigen::ArrayXd &ary) const {
  return ary.unaryExpr([&](const double &x)
		       { return this->inverse(x); });
};


// glmLink -----------------------------------------------------------
