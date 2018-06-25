

btlmPostMode <- function(X, y, beta0, lambda = NULL,
                         tau.sq = 1e4, M = NULL, batch.size = min(100, nrow(X)),
                         iter.max = 1000, eps = 1e-8, tol = 1e-6,
                         learning.rate = NULL, mt.decay = 0.9, vt.decay = 0.999,
                         rng.seed = NULL
                         ) {
  if (is.null(M))  M <- max(abs(beta0))
  if (is.null(lambda))  lambda <- M * rbeta(1, 1, 49)
  if (is.null(learning.rate))  learning.rate <- M / min(dim(X))
  if (is.null(rng.seed))  rng.seed <- as.integer(Sys.time())
  structure(
    .Call("btlmPostApprox", X, y, beta0, lambda, tau.sq, M, batch.size,
          iter.max, eps, tol, learning.rate, mt.decay, vt.decay, rng.seed,
          PACKAGE = "btglm"),
    class = "btglm")
}



coef.btglm <- function(object, threshold = TRUE) {
  ans <- if (threshold)
    object$coefficients * (abs(object$coefficients) > object$lambda)
  else
    object$coefficients
  ans[1] <- object$coefficients[1]
  ans
}

