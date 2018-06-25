
#include "ThresholdGLM.h"


// double ThresholdGLM::constants::epsCauchy = 1e-6;
// double ThresholdGLM::constants::updateCauchyScale = 100.0;

// int ThresholdGLM::constants::updateLambdaAfter = 20;



double ThresholdGLM::approxDThreshCauchy(
					 const double &theta,
					 const double &lambda,
					 const double &eps
					 ) {
  return (dcauchy(theta, lambda, eps) +
	  dcauchy(theta, -lambda, eps));
};

double ThresholdGLM::approxDPsiCauchy(  // psi = ln(lambda) - ln(M - lambda)
				      const double &lambda,
				      const double &theta,
				      const double &M,
				      const double &eps
					) {
  return (-lambda * (M - lambda) / M *
	  (dcauchy(theta, lambda, eps) +
	   dcauchy(theta, -lambda, eps)));
};

