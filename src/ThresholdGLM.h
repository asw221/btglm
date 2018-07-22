
#define _USE_MATH_DEFINES
#include <cmath>
#include <Eigen/Core>
#include <random>

using std::pow;


#ifndef _THRESHOLD_GLM_
#define _THRESHOLD_GLM_

namespace ThresholdGLM {

  extern std::mt19937 _rng_;
  extern std::uniform_real_distribution<double> _Uniform_;

  // extern double _dthreshScale_;
  // extern double _pRejectDecay_;
  // extern double _pRejectTarget_[2];


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
