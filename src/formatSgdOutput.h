
#include <Rcpp.h>
#include <sstream>

#include "AdaM.h"


#ifndef _FORMAT_SGD_OUTPUT_
#define _FORMAT_SGD_OUTPUT_

template< typename T >
Rcpp::List formatSgdOutput(const AdaM<T> &sgd, const double &tol = 1e-6);



template< typename T >
Rcpp::List formatSgdOutput(const AdaM<T> &sgd, const double &tol) {
  std::ostringstream msg;
  if (!sgd.converged(tol))
    msg << "Check diagnostics: Algorithm did not converge after "
	<< sgd.iteration() << " iterations";
  return (Rcpp::List::create(
	    Rcpp::Named("iter") = sgd.iteration(),
	    Rcpp::Named("delta") = sgd.dtheta(),
	    Rcpp::Named("msg") = msg.str()
	  ));
};



#endif  // _FORMAT_SGD_OUTPUT_


