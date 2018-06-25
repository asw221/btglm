
#include <cmath>
#include <Eigen/Core>
#include <random>
#include <vector>


// May, 2018 - Andrew Whiteman
//
/*! \class AdaM
\brief ADAptive Moment SGD optimization routine

Details.
 */



#ifndef _ADA_M_
#define _ADA_M_


template< typename T = double >
class AdaM
{
private :
  T _mt;             // current momentum
  T _vt;             // current velocity
  double _eta;       // learning rate
  double _gamma[2];  // momentum/velocity decay rates
  double _eps;       // avoid division by zero constant
  double _dtheta;    // || \theta^(t) - \theta^(t-1) ||^2
  int _iter;         // current number of updates

  void updateMomentum(const T &gt);
  void updateVelocity(const T &gt);

  template< typename S >
  void updatePosition(S &theta);


public :
  AdaM(
    const T &theta,                // parameter(s) to optimize
    const double &eta = 0.001,     // SGD learning rate, range (0, 1]
    const double &gamma1 = 0.90,   // momentum decay rate, range (0, 1)
    const double &gamma2 = 0.999,  // velocity decay rate, range (0, 1)
    const double &eps = 1e-8       // small positive constant
  );


  // Update methods
  // type R is the same as the type returned by the gradient function
  template< typename S, typename R, typename... Args >
  void update(S &theta, R gradient(const S &theta, Args&...), Args&... args);

  template< typename S, typename R, typename... Args >
  void minibatchUpdate(
    S &theta,
    R unitGradient(const S &theta, const int &i, Args&...),
    std::mt19937 &rng,
    const int &batchSize,
    std::vector<int> &index,
    Args&... args
  );


  // Simple getter methods
  bool converged(const double &tol = 1e-6) const;
  int iteration() const;
  double dtheta() const;
};


#include "AdaM.inl"

#endif  // _ADA_M_
