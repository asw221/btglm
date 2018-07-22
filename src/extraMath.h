
#include <Eigen/Core>
#include <random>


#ifndef _EXTRA_MATH_
#define _EXTRA_MATH_


// Squared 2-norm: these should be relocated at some point
// (general ExtraMath.h/cpp file or something)

template< typename T = double >
double squaredNorm(const T &x);

double squaredNorm(const Eigen::VectorXd &x);
double squaredNorm(const Eigen::ArrayXd &x);


// template< typename T = double >
// T gaussianNoise(const T &scl, std::mt19937 &rng);

Eigen::ArrayXd gaussianNoise(const Eigen::ArrayXd &scl, std::mt19937 &rng);
double gaussianNoise(const double &scl, std::mt19937 &rng);
// Eigen::VectorXd gaussianNoise(const Eigen::VectorXd &scl, std::mt19937 &rng);

#endif  // _EXTRA_MATH_
