

btglmControl <- function(
  batch.size = 100, burnin = 500, eps = 1e-8,
  iter.max.sgd = 2000, learning.rate = NULL,
  metropolis.target = 0.9, mt.decay = 0.9,
  n.save = 100, rng.seed = NULL, thin = 200, tol = 1e-6, vt.decay = 0.99
  ) {
  structure(
    list(batch.size = batch.size, burnin = burnin, eps = eps,
         iter.max.sgd = iter.max.sgd, learning.rate = learning.rate,
         metropolis.target = metropolis.target, mt.decay = mt.decay,
         n.save = n.save, rng.seed = rng.seed, thin = thin,
         tol = tol, vt.decay = vt.decay),
    class = "btglmControl")
}


