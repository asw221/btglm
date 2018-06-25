
#define _USE_MATH_DEFINES
#include <cmath>

using std::pow;


#ifndef _THRESHOLD_GLM_
#define _THRESHOLD_GLM_

namespace ThresholdGLM {

  // namespace constants {
  //   extern double epsCauchy;
  //   extern double updateCauchyScale;
  //
  //   extern int updateLambdaAfter;
  // };


  template< typename T = double >
  T dcauchy(const T &x, const T &location = 0.0, const T &scale = 1.0) {
    return (scale / ((pow(scale, 2) + pow(x - location, 2)) * M_PI));
  };


  double approxDThreshCauchy(
    const double &theta,
    const double &lambda,
    const double &eps = 1e-6
  );

  double approxDPsiCauchy(  // psi = ln(lambda) - ln(M - lambda)
    const double &lambda,
    const double &theta,
    const double &M,
    const double &eps = 1e-6
  );
  
}

#endif  // _THRESHOLD_GLM_
