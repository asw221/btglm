
#include <algorithm>
#include <cmath>
#include <random>




template< typename T >
template< typename S, typename... Args >
void RandomWalk<T>::update(
  S &theta,
  double logPosterior(const S &theta, Args&&...),
  std::mt19937 &rng,
  Args&&... args
) {
  if (_iter == 0)
    _logPcurrent = std::function<double(const S&, Args&&...)>
      (logPosterior)(theta, std::forward<Args>(args)...);
  
  // Propose new theta with Normal kernel and Metropolis correction
  const S proposal = theta + _eps * _normal(rng);
  const double lpProposal = std::function<double(const S&, Args&&...)>
      (logPosterior)(proposal, std::forward<Args>(args)...);
  const double ratio = std::exp(lpProposal - _logPcurrent);
  if (_uniform(rng) < ratio) {
    theta = proposal;
    _logPcurrent = lpProposal;
  }

  // Adjust random walk parameters
  _cumJumpProb = _gamma * _cumJumpProb +
    (1 - _gamma) * std::min(ratio, 1.0);
  if (_updateEps && _iter % 100 == 0)
    updateStepSize();
  if (std::abs(jumpProbability() - _targetJumpProb) < _tolJump)
    _updateEps = false;
  _iter++;
};





template< typename T >
double RandomWalk<T>::epsilon() const {
  return _eps;
};

template< typename T >
double RandomWalk<T>::jumpProbability() const {
  double p = 0.0;
  if (_iter)
    p = _cumJumpProb / (1 - std::pow(_gamma, _iter));
  return (p);
};

template< typename T >
int RandomWalk<T>::iteration() const {
  return _iter;
};



template< typename T >
void RandomWalk<T>::fixStepSize() {
  _updateEps = false;
};


template< typename T >
void RandomWalk<T>::updateStepSize(const double &k) {
  if (_cumJumpProb > _targetJumpProb)
    _eps *= 1 - k;
  else if (_cumJumpProb < _targetJumpProb)
    _eps *= 1 + k;
}

