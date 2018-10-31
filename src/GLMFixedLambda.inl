
#include <cmath>
#include <Eigen/Core>
#include <vector>

#include "AdaM.h"
#include "ThresholdGLM.h"




// GLMFixedLambda ----------------------------------------------------


template< typename Link >
void updateTheta(GLMFixedLambda<Link> &theta, Eigen::ArrayXd &delta) {
  theta -= delta;
  theta.update();
};



template< typename Link >
Eigen::VectorXd GLMFixedLambda<Link>::residuals(
  const Eigen::MatrixXd &X,
  const Eigen::VectorXd &y
) const {
  return (y - _link.inverse((_spar * X).transpose().array()).matrix());
};


template< typename Link >
double GLMFixedLambda<Link>::objective(
  const Eigen::MatrixXd &X,
  const Eigen::VectorXd &y,
  const double &priorPrecision
) const {
  Eigen::ArrayXd eta = (_spar * X).transpose().array();
  const double likelihood = y.transpose() * eta.matrix() +
    (1 - _link.inverse(eta)).log().sum();
  const double prior = 0.5 * priorPrecision * this->matrix().squaredNorm();
  return (-(likelihood + prior));
};


template< typename Link >
Eigen::ArrayXd glmUnitGradient(
  const GLMFixedLambda<Link> &theta,
  const int &i,
  const Eigen::MatrixXd &X,
  const Eigen::VectorXd &y,
  const double &priorPrecision
) {
  const double mu = theta._link.inverse((theta._spar * X.col(i))(0));
  // mu = theta._link.inverse(mu);
  Eigen::ArrayXd grad(theta.size());
  grad = (theta._deriv * X.col(i).array()) * (mu - y(i))
    + (theta * priorPrecision / X.cols());
  for (int i = 0; i < theta._include.size(); i++)
    grad.coeffRef(theta._include[i]) -= theta.coeffRef(theta._include[i]) *
      priorPrecision / X.cols();
  return (grad);
};




// template< typename Link >
// void GLMFixedLambda<Link>::sgldUpdate(
//   AdaM<Eigen::ArrayXd> &sgd,
//   const int &batchSize,
//   std::vector<int> &dataIndex,
//   double &learningScale,
//   double &acceptanceProbability,
//   const bool updateLearningScale,
//   const Eigen::MatrixXd &X,
//   const Eigen::VectorXd &y,
//   const double &priorPrecision,
//   const double &metropolisTarget
// ) {
//   const double decay = 0.9, k = 0.01, P = this->size();
//   double gradScale = dataIndex.size() / batchSize;
//   double lpCurrent, lpProposal, pAccept;
    
//   // Compute SGD update step -----------------------------------------
//   sgd.virtualMinibatch
//     <GLMFixedLambda<Link>, Eigen::ArrayXd, const Eigen::MatrixXd&,
//      const Eigen::VectorXd&, const double&>
//     (*this, glmUnitGradient<Link>, ThresholdGLM::_rng_, batchSize,
//      dataIndex, X, y, priorPrecision);

//   Eigen::ArrayXd Mass = Eigen::ArrayXd::Constant(P,
//     _lambda * std::pow(std::log(P) / P, 2));
//   for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator
// 	 it(_spar, 0); it; ++it)
//     Mass.coeffRef(it.index()) = std::sqrt(1 / (sgd.velocity().coeffRef(it.index()) + 1e-8));
//   double eta = sgd.eta() * learningScale;
//   Eigen::ArrayXd delta = eta * Mass * sgd.momentum() +
//     std::sqrt(2 * eta / gradScale) * gaussianNoise(Mass.sqrt(), ThresholdGLM::_rng_);

//   lpCurrent = objective(X, y, priorPrecision);
//   *this -= delta;
//   setSparse();
//   lpProposal = objective(X, y, priorPrecision);
//   pAccept = std::min(std::exp(lpProposal - lpCurrent), 1.0);
//   acceptanceProbability = acceptanceProbability * decay +
//     pAccept * (1 - decay);
//   if (ThresholdGLM::_Uniform_(ThresholdGLM::_rng_) < pAccept)
//     computeDeriv();
//   else {  // reject update
//     *this += delta;
//     setSparse();
//   }
//   if (updateLearningScale) {
//     if (acceptanceProbability < metropolisTarget)
//       learningScale *= 1 - k;
//     else
//       learningScale *= 1 + k;
//   }
// };





