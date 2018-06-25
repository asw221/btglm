
#include <algorithm>
#include <cmath>
#include <Eigen/Core>
#include <random>
#include <vector>


using std::sqrt;
using std::pow;


// May, 2018 - Andrew Whiteman
//
// Implementation file for AdaM.h
// ===================================================================


template< typename T, typename S >
void updateTheta(T &theta, S &delta) {
  theta -= delta;
};

// Define:
// void updateTheta(Other &theta, Eigen::ArrayXd &delta);
// As long as you can do math with the Other class, this should be good

// Then define a linear threshold coefficient object (inheriting from Eigen)
//                                            ^ Eigen::ArrayXd (hence namespace)
// along with a type-specific overloaded updateTheta()
// Might require clever handling of namespaces



// Squared 2-norm: these should be relocated at some point
// (general ExtraMath.h/cpp file or something)

template< typename T >
double squaredNorm(const T &x);

double squaredNorm(const Eigen::VectorXd &x);
double squaredNorm(const Eigen::ArrayXd &x);




// Public Methods
// -------------------------------------------------------------------

template< typename T >
AdaM<T>::AdaM(
  const T &theta,
  const double &eta,
  const double &gamma1,
  const double &gamma2,
  const double &eps
) {
  if (eta <= 0 || eta > 1)
    throw (std::logic_error("Learning rate must be between (0, 1]"));
  if (gamma1 <= 0 || gamma1 >= 1)
    throw (std::logic_error("Momentum decay rate must be between (0, 1)"));
  if (gamma2 <= 0 || gamma2 >= 1)
    throw (std::logic_error("Velocity decay rate must be between (0, 1)"));
  if (eps <= 0)
    throw (std::logic_error("eps must be between (0, inf)"));

  _eta = eta;
  _gamma[0] = gamma1;
  _gamma[1] = gamma2;
  _eps = eps;

  _mt = theta * 0;
  _vt = theta * 0;
  _dtheta = 0.0;
  _iter = 1;
};


// Batch update
// type R is the same as the type returned by the gradient function
template< typename T >
template< typename S, typename R, typename... Args >
void AdaM<T>::update(S &theta, R gradient(const S &theta, Args&...), Args&... args) {
  T gt = std::function< R(const S&, Args&...) >
    (gradient)(theta, std::forward< Args >(args)...);  // compute gradient
  updateMomentum(gt);
  updateVelocity(gt);
  updatePosition(theta);
  _iter++;
};


// Minibatch update
// type R is the same as the type returned by the gradient function
template< typename T >
template< typename S, typename R, typename... Args >
void AdaM<T>::minibatchUpdate(
  S &theta,
  R unitGradient(const S &theta, const int &i, Args&...),
  std::mt19937 &rng,
  const int &batchSize,
  std::vector<int> &index,
  Args&... args
) {
  // static std::normal_distribution<double> Z(0.0, 1.0);
  // static std::mt19937 rng(1);
  std::shuffle(index.begin(), index.end(), rng);
  std::vector<int>::iterator it(index.begin());
  T gt = theta * 0;
  for (int j = 1; it != index.end(); it++, j++) {
    gt += std::function< R(const S&, const int&, Args&...) >
      (unitGradient)(theta, (*it), std::forward<Args>(args)...);
    if (j % batchSize == 0) {
      // for (int i = 0; i < gt.size(); i++)
      // 	gt(i) += Z(rng) * sqrt(_eta) / pow(_iter, 0.55);
      updateMomentum(gt);
      updateVelocity(gt);
      updatePosition(theta);
      gt *= 0;
    }
  }
  _iter++;
};


template< typename T >
bool AdaM<T>::converged(const double &tol) const {
  return (_iter > 1 && _dtheta <= tol);
};

template< typename T >
int AdaM<T>::iteration() const {
  return (_iter - 1);
};

template< typename T >
double AdaM<T>::dtheta() const {
  return (_dtheta);
};





// Private Methods
// -------------------------------------------------------------------

template< typename T >
void AdaM<T>::updateMomentum(const T &gt) {
  _mt = _gamma[0] * _mt + (1 - _gamma[0]) * gt;
};


template< typename T >
void AdaM<T>::updateVelocity(const T &gt) {
  _vt = _gamma[1] * _vt + (1 - _gamma[1]) * pow(gt, 2);
};


template< typename T >
template< typename S >
void AdaM<T>::updatePosition(S &theta) {
  double eta_t = _eta * sqrt(1 - pow(_gamma[1], _iter)) /
    (1 - pow(_gamma[0], _iter));
  T delta = eta_t / (sqrt(_vt) + _eps) * _mt;
  _dtheta = squaredNorm(delta);
  updateTheta(theta, delta);
};







template< typename T = double >
double squaredNorm(const T &x) {  // Squared 2-norm
  return (pow(x, 2));
};

double squaredNorm(const Eigen::VectorXd &x) {
  return (x.squaredNorm());
};

double squaredNorm(const Eigen::ArrayXd &x) {
  return (x.matrix().squaredNorm());
};
