

btglmPostMode <- function(X, y, beta0, lambda = NULL,
                          tau.sq = 1e4, M = NULL, include = 1,
                          batch.size = min(100, nrow(X)),
                          iter.max = 1000, eps = 1e-8, tol = 1e-6,
                          learning.rate = NULL, mt.decay = 0.9, vt.decay = 0.99,
                          family = binomial(),  # make this explicit
                          rng.seed = NULL
                          ) {
  if (is.null(M))  M <- max(abs(beta0[-include]))
  if (is.null(lambda))  lambda <- M * rbeta(1, 1, 49)
  if (is.null(learning.rate))  learning.rate <- M / min(dim(X))
  if (is.null(rng.seed))  rng.seed <- as.integer(Sys.time())
  if (ncol(X) == length(beta0))
    X <- t(X)
  else if (nrow(X) != length(beta0))
    stop ("dim(X) not related to dim(beta0) (", length(beta0), ")")
  structure(
    .Call("btglmPostApprox", X, y, beta0, lambda, tau.sq, M, include - 1,
          batch.size, iter.max, eps, tol, learning.rate, mt.decay, vt.decay,
          rng.seed,
          PACKAGE = "btglm"),
    family = family,
    class = "btglm")
}





btglm <- function(X, y, beta0, lambda = NULL,
                 tau.sq = 1e4, M = NULL, include = 1,
                 batch.size = min(100, nrow(X)),
                 n.save = 100, thin = 200, burnin = 500,
                 iter.max.sgd = 2000, eps = 1e-8, tol = 1e-6,
                 learning.rate = NULL, metropolis.target = 0.9,
                 mt.decay = 0.9, vt.decay = 0.99,
                 family = binomial(),  # make explicit
                 rng.seed = NULL
                 ) {
  if (is.null(M))  M <- max(abs(beta0))
  if (is.null(lambda))  lambda <- M * rbeta(1, 1, 49)
  if (is.null(learning.rate))  learning.rate <- M / min(dim(X))
  if (is.null(rng.seed))  rng.seed <- as.integer(Sys.time())
  if (!any(dim(X) == length(beta0)))
    stop ("dim(X) not related to dim(beta0) (", length(beta0), ")")
  else if (nrow(X) != length(beta))
    X <- t(X)
  n.save <- floor(n.save)
  thin <- floor(thin)
  burnin <- floor(burnin)
  structure(
    .Call("btglm", X, y, beta0, lambda, tau.sq, M, include - 1, batch.size,
          n.save, thin, burnin, iter.max.sgd, eps, tol, learning.rate, mt.decay,
          vt.decay, metropolis.target, rng.seed, PACKAGE = "btglm"),
    class = "btglm")
}

