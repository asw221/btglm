
#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "ThresholdGLM.h"


#ifndef _THRESHOLD_PARAMETERS_
#define _THRESHOLD_PARAMETERS_


class LMFixedLambda :
  public Eigen::ArrayXd
{
private:
  double _lambda;
  double _M;       // lambda \in [0, M]
  Eigen::SparseMatrix<double, Eigen::RowMajor> _spar;
  Eigen::ArrayXd _deriv;
  Eigen::ArrayXd _derivPsi;
  int _include;  // change to Rcpp::IntegerVector
  int _iter;

  virtual void computeDeriv();
  
public:
  // Unit Gradient function controls how class handles gradient updates
  friend Eigen::ArrayXd lmUnitGradient(
    const LMFixedLambda &theta,
    const int &i,
    const Eigen::MatrixXd &X,
    const Eigen::VectorXd &y,
    const double &tauSq
  );

  
  // Allows construction from Eigen expressions
  template< typename T >
  LMFixedLambda(const Eigen::ArrayBase<T> &other,
		const double &lambda,
		const double &M) :
    Eigen::ArrayXd(other), _lambda(lambda), _M(M), _include(0), _iter(1)
  {
    _deriv.resize(this->size());
    _derivPsi.resize(this->size());
    update();
  };
  

  // Setters
  void setLambda(const double &lambda);
  void setSparse();
  void update();

  // Getters
  int iteration() const;
  double lambda() const;
  double M() const;

  
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
    const double &tauSq
  ) const;

};


// (Overloaded) updateTheta() allows class to interface with gradient
// update methods
void updateTheta(LMFixedLambda &theta, Eigen::ArrayXd &delta);

Eigen::ArrayXd lmUnitGradient(
  const LMFixedLambda &theta,
  const int &i,
  const Eigen::MatrixXd &X,
  const Eigen::VectorXd &y,
  const double &tauSq
);

#endif  // _THRESHOLD_PARAMETERS_


