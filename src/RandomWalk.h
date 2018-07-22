
#include <random>


#ifndef _RANDOM_WALK_
#define _RANDOM_WALK_


template< typename T = double >
class RandomWalk {
  bool _updateEps;
  
  double _cumJumpProb;
  double _eps;
  double _gamma;  // jump prob decay parameter
  double _logPcurrent;
  double _targetJumpProb;
  double _tolJump;
  
  int _iter;
  
  std::normal_distribution<double> _normal;
  std::uniform_real_distribution<double> _uniform;

  void updateStepSize(const double &k = 0.1);

public:
  RandomWalk(
    const double &eps = 2.4,
    const double &targetJumpProb = 0.44,
    const double &gamma = 0.9
  ) :
    _updateEps(true), _cumJumpProb(0.0), _logPcurrent(1.0),
    _tolJump(0.1), _iter(0), _normal(0.0, 1.0), _uniform(0.0, 1.0)
  {
    if (eps <= 0)
      throw (std::logic_error("Step size must be > 0"));
    if (targetJumpProb <= 0 || targetJumpProb >= 1)
      throw (std::logic_error("Target jump rate must be between (0, 1"));
    if (gamma <= 0 || gamma >= 1)
      throw (std::logic_error("Decay rate must be between (0, 1)"));
    _eps = eps;
    _targetJumpProb = targetJumpProb;
    _gamma = gamma;
  };


  template< typename S, typename... Args >
  void update(
    S &theta,
    double logPosterior(const S &theta, Args&&...),
    std::mt19937 &rng,
    Args&&... args
  );

  // getters
  double epsilon() const;
  double jumpProbability() const;
  int iteration() const;

  void fixStepSize();
  
};


#include "RandomWalk.inl"

#endif  // _RANDOM_WALK_


