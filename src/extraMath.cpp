
#include <Eigen/Core>
#include <random>

#include "extraMath.h"


template< typename T >
double squaredNorm(const T &x) {  // Squared 2-norm
  return (pow(x, 2));
};

double squaredNorm(const Eigen::VectorXd &x) {
  return (x.squaredNorm());
};

double squaredNorm(const Eigen::ArrayXd &x) {
  return (x.matrix().squaredNorm());
};



// template< typename T >
// T gaussianNoise(const T &scl, std::mt19937 &rng) {
//   std::normal_distribution<T> _z(0, 1);
//   return (scl * _z(rng));
// };

Eigen::ArrayXd gaussianNoise(const Eigen::ArrayXd &scl, std::mt19937 &rng) {
  static std::normal_distribution<double> _z(0, 1);
  Eigen::ArrayXd noise = scl.unaryExpr([&](const double &x) {
      return (x * _z(rng)); });
  return (noise);
};

double gaussianNoise(const double &scl, std::mt19937 &rng) {
  static std::normal_distribution<double> _z(0, 1);
  return (scl * _z(rng));
};

// Eigen::VectorXd gaussianNoise(const Eigen::VectorXd &scl, std::mt19937 &rng) {
//   static std::normal_distribution<double> _z(0, 1);
//   Eigen::VectorXd noise = scl.array().unaryExpr([&](const double &x) {
//       return (x * _z(rng)); }).matrix();
//   return (noise);
// };
