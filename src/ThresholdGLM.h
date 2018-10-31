
#define _USE_MATH_DEFINES
#include <cmath>
#include <Eigen/Core>
#include <random>

using std::pow;
using std::atan;


#ifndef _THRESHOLD_GLM_
#define _THRESHOLD_GLM_

namespace ThresholdGLM {

  extern std::mt19937 _rng_;
  extern std::uniform_real_distribution<double> _Uniform_;

  extern double _epsilon_;  // Threshold approximation function scale
  extern double _lambdaDecayRate_;
  extern double _minLambda_;
  // extern double _dthreshScale_;
  // extern double _pRejectDecay_;
  // extern double _pRejectTarget_[2];


  template< typename T = double >
  T dcauchy(const T &x, const T &location = 0.0, const T &scale = 1.0) {
    return (scale / ((pow(scale, 2) + pow(x - location, 2)) * M_PI));
  };

  template< typename T = double >
  T pcauchy(const T &x, const T &location = 0.0, const T &scale = 1.0) {
    return (0.5 + atan((x - location) / scale) / M_PI);
  };


  double approxThresholdCauchy(
    const double &theta,
    const double &lamba,
    const double &eps = 0.1
  );


  double approxDThreshCauchy(
    const double &theta,
    const double &lambda,
    const double &eps = 0.1
  );

  double approxDPsiCauchy(  // psi = ln(lambda) - ln(M - lambda)
    const double &lambda,
    const double &theta,
    const double &M,
    const double &eps = 0.1
  );



  
  class glmLink
  {
  public:
    glmLink()
    { ; }
      
    virtual double operator()(const double &x) const;
    virtual Eigen::ArrayXd operator()(const Eigen::ArrayXd &ary) const;
    virtual double inverse(const double &x) const;
    virtual Eigen::ArrayXd inverse(const Eigen::ArrayXd &ary) const;
  };



  class logit :
    public glmLink
  {
  public:
    logit() : glmLink()
    { ; }
    
    virtual double operator()(const double &x) const override;
    virtual Eigen::ArrayXd operator()(const Eigen::ArrayXd &ary) const override;
    virtual double inverse(const double &x) const override;
    virtual Eigen::ArrayXd inverse(const Eigen::ArrayXd &ary) const override;
  };

  
}

#endif  // _THRESHOLD_GLM_
