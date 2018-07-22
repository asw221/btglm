
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


Eigen::SparseMatrix<double, Eigen::RowMajor> activeCoefGradient(
  const LMFixedLambda &theta,
  const Eigen::MatrixXd &X,
  const Eigen::VectorXd &y,
  const double &priorPrecision
);


Eigen::ArrayXd lmUnitGradient(
  const LMFixedLambda &theta,
  const int &i,
  const Eigen::MatrixXd &X,
  const Eigen::VectorXd &y,
  const double &priorPrecision
);


void sgldUpdate(
  LMFixedLambda &theta,
  AdaM<Eigen::ArrayXd> &sgd,
  const int &batchSize,
  std::vector<int> &dataIndex,
  double &learningScale,
  double &acceptanceProbability,
  const bool updateLearningScale,
  const Eigen::MatrixXd &X,
  const Eigen::VectorXd &y,
  const double &residualPrecision,
  const double &priorPrecision,
  const double &metropolisTarget = 0.44
);





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









// LMFixedLambda -----------------------------------------------------

class LMFixedLambda :
  public Eigen::ArrayXd
{
protected:
  double _lambda;
  double _M;       // lambda \in [0, M]
  
  Eigen::ArrayXd _deriv;
  Eigen::SparseMatrix<double, Eigen::RowMajor> _spar;
  Rcpp::IntegerVector _include;  //
  
  virtual void computeDeriv();
  
public:
  // Allows construction from Eigen expressions
  template< typename T >
  LMFixedLambda(const Eigen::ArrayBase<T> &other,
		const double &lambda,
		const double &lambdaMax,
		const Rcpp::IntegerVector &include) :
    Eigen::ArrayXd(other),
    _include(include)
  {
    if (lambdaMax <= 0)
      throw (std::logic_error("Maximal lambda value must be >= 0"));
    if (lambda < 0 || lambda > lambdaMax)
      throw (std::logic_error("lambda sould be between [0, lambdaMax]"));
    _lambda = lambda;
    _M = lambdaMax;
    _deriv = Eigen::ArrayXd::Zero(this->size());
    update();
  };

  

  friend Eigen::SparseMatrix<double, Eigen::RowMajor> activeCoefGradient(
    const LMFixedLambda &theta,
    const Eigen::MatrixXd &X,
    const Eigen::VectorXd &y,
    const double &priorPrecision
  );
  
  // Unit Gradient function controls how class handles gradient updates
  friend Eigen::ArrayXd lmUnitGradient(
    const LMFixedLambda &theta,
    const int &i,
    const Eigen::MatrixXd &X,
    const Eigen::VectorXd &y,
    const double &priorPrecision
  );

  friend void sgldUpdate(
    LMFixedLambda &theta,
    AdaM<Eigen::ArrayXd> &sgd,
    const int &batchSize,
    std::vector<int> &dataIndex,
    double &learningScale,
    double &acceptanceProbability,
    const bool updateLearningScale,
    const Eigen::MatrixXd &X,
    const Eigen::VectorXd &y,
    const double &residualPrecision,
    const double &priorPrecision,
    const double &metropolisTarget
  );  
  

  // Setters
  void lambda(const double &lambda);
  void setSparse();
  void update();

  // Getters
  int nonZeros() const;
  double lambda() const;
  double lambdaMax() const;
  double minSparseCoeff() const;

  
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

  void sgldUpdate(
    AdaM<Eigen::ArrayXd> &sgd,
    const int &batchSize,
    std::vector<int> &dataIndex,
    double &learningScale,
    double &acceptanceProbability,
    const bool updateLearningScale,
    const Eigen::MatrixXd &X,
    const Eigen::VectorXd &y,
    // const double &residualPrecision,
    const double &priorPrecision,
    const double &metropolisTarget
  );
  

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


