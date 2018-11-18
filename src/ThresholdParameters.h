
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <memory>
#include <Rcpp.h>

#include "AdaM.h"
#include "ThresholdGLM.h"


#ifndef _THRESHOLD_PARAMETERS_
#define _THRESHOLD_PARAMETERS_

class LMFixedLambda;

template< typename Link >
class GLMFixedLambda;





// (Overloaded) updateTheta() allows class to interface with gradient
// update methods
void updateTheta(LMFixedLambda &theta, Eigen::ArrayXd &delta);


template< typename Link >
void updateTheta(GLMFixedLambda<Link> &theta, Eigen::ArrayXd &delta);




Eigen::ArrayXd lmUnitGradient(
  const LMFixedLambda &theta,
  const int &i,
  const Eigen::MatrixXd &X,
  const Eigen::VectorXd &y,
  const double &priorPrecision
);


// void sgldUpdate(
//   LMFixedLambda &theta,
//   AdaM<Eigen::ArrayXd> &sgd,
//   const int &batchSize,
//   std::vector<int> &dataIndex,
//   double &learningScale,
//   double &acceptanceProbability,
//   const bool updateLearningScale,
//   const Eigen::MatrixXd &X,
//   const Eigen::VectorXd &y,
//   const double &residualPrecision,
//   const double &priorPrecision,
//   const double &metropolisTarget = 0.44
// );





template< typename Link >
Eigen::ArrayXd glmUnitGradient(
  const GLMFixedLambda<Link> &theta,
  const int &i,
  const Eigen::MatrixXd &X,
  const Eigen::VectorXd &y,
  const double &priorPrecision
);

// template< typename Link >
// void sgldUpdate(
//   GLMFixedLambda<Link> &theta,
//   AdaM<Eigen::ArrayXd> &sgd,
//   const int &batchSize,
//   std::vector<int> &dataIndex,
//   std::vector<double> &learningScale,
//   std::vector<double> &acceptanceProbability,
//   const bool updateLearningScale,
//   const Eigen::MatrixXd &X,
//   const Eigen::VectorXd &y,
//   const double &priorPrecision,
//   const double &metropolisTarget
// );



double findReasonableLambda(const LMFixedLambda &theta);









// LMFixedLambda -----------------------------------------------------

class LMFixedLambda :
  public Eigen::ArrayXd
{
protected:
  double _lambda;               // threshold parameter
  double _priorModelSizeScale;  // prior hyperparamter, k
  
  Eigen::ArrayXd _deriv;        // diagonal derivative of \sigma(\beta) 
  Eigen::ArrayXd _priorDeriv;   // derivative of k * \sum_j H(\beta_j)
  Eigen::VectorXd _spar;        // \sigma(\beta)
  // Store these because with stochastic gradient update schemes, these
  // parameters need to be evaluated potentially at every data point,
  // but their values don't change between gradient updates
  
  Rcpp::IntegerVector _include;
  // Integer index set of coefficients that should always be active
  
  virtual void computeDeriv();
  
public:
  // Allows construction from Eigen expressions
  template< typename T >
  LMFixedLambda(const Eigen::ArrayBase<T> &other,
		const double &lambda,
		const Rcpp::IntegerVector &include,
		const double priorModelSizeScale = 2.0) :
    Eigen::ArrayXd(other),
    _priorModelSizeScale(priorModelSizeScale),
    _include(include)
  {
    if (lambda < 0)
      throw (std::logic_error("lambda sould be >= 0"));
    _lambda = lambda;
    _deriv = Eigen::ArrayXd::Zero(this->size());
    _priorDeriv = Eigen::ArrayXd::Zero(this->size());
    _spar = Eigen::VectorXd::Zero(this->size());
    // update();
    setSparse();
  };

  

  
  
  // Unit Gradient function controls how class handles gradient updates
  friend Eigen::ArrayXd lmUnitGradient(
    const LMFixedLambda &theta,
    const int &i,
    const Eigen::MatrixXd &X,
    const Eigen::VectorXd &y,
    const double &priorPrecision
  );

  // friend void sgldUpdate(
  //   LMFixedLambda &theta,
  //   AdaM<Eigen::ArrayXd> &sgd,
  //   const int &batchSize,
  //   std::vector<int> &dataIndex,
  //   double &learningScale,
  //   double &acceptanceProbability,
  //   const bool updateLearningScale,
  //   const Eigen::MatrixXd &X,
  //   const Eigen::VectorXd &y,
  //   const double &residualPrecision,
  //   const double &priorPrecision,
  //   const double &metropolisTarget
  // );
  
  

  // Setters
  void lambda(const double &lambda);
  void setSparse();
  void update();

  // Getters
  double lambda() const;
  double minActiveCoeff() const;

  
  // Allows Eigen expressions to be assigned to this class
  template< typename T >
  LMFixedLambda& operator=(const Eigen::ArrayBase<T> &other) {
    this->Eigen::ArrayXd::operator=(other);
    return (*this);
  };

  // Misc
  virtual Eigen::VectorXd residuals(
    const Eigen::MatrixXd &X,
    const Eigen::VectorXd &y
  ) const;

  // Maybe not the best place for this
  virtual double objective(
    const Eigen::MatrixXd &X,
    const Eigen::VectorXd &y,
    const double &priorPrecision
  ) const;
};

// LMFixedLambda -----------------------------------------------------





// GLMFixedLambda ----------------------------------------------------

template< typename Link = ThresholdGLM::glmLink >
class GLMFixedLambda :
  public LMFixedLambda
{
protected:
  Link _link;
  double _phi;  // dispersion parameter

public:
  template< typename T >
  GLMFixedLambda(const Eigen::ArrayBase<T> &other,
		 const double &lambda,
		 const double &lambdaMax,
		 const Rcpp::IntegerVector &include,
		 const double &dispersion = 1.0) :
    LMFixedLambda(other, lambda, lambdaMax, include),
    _phi(dispersion)
  { ; }

  
  // Unit Gradient function controls how class handles gradient updates
  template< typename T >
  friend Eigen::ArrayXd glmUnitGradient(
    const GLMFixedLambda<T> &theta,
    const int &i,
    const Eigen::MatrixXd &X,
    const Eigen::VectorXd &y,
    const double &priorPrecision
  );

  // void sgldUpdate(
  //   AdaM<Eigen::ArrayXd> &sgd,
  //   const int &batchSize,
  //   std::vector<int> &dataIndex,
  //   double &learningScale,
  //   double &acceptanceProbability,
  //   const bool updateLearningScale,
  //   const Eigen::MatrixXd &X,
  //   const Eigen::VectorXd &y,
  //   // const double &residualPrecision,
  //   const double &priorPrecision,
  //   const double &metropolisTarget
  // );
  

  virtual Eigen::VectorXd residuals(
    const Eigen::MatrixXd &X,
    const Eigen::VectorXd &y
  ) const override;

  virtual double objective(
    const Eigen::MatrixXd &X,
    const Eigen::VectorXd &y,
    const double &priorPrecision
  ) const override;
};


#include "GLMFixedLambda.inl"


#endif  // _THRESHOLD_PARAMETERS_


