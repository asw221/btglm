

btlm <- function(X, y, beta0, lambda = NULL,
                 tau.sq = 1e4, M = NULL, include = 1,
                 batch.size = min(100, nrow(X)),
                 n.save = 100, thin = 200, burnin = 500,
                 iter.max.sgd = 2000, eps = 1e-8, tol = 1e-6,
                 learning.rate = NULL, metropolis.target = 0.9,
                 mt.decay = 0.9, vt.decay = 0.99,
                 rng.seed = NULL
                 ) {
  if (is.null(M))  M <- max(abs(beta0))
  if (is.null(lambda))  lambda <- M * rbeta(1, 1, 49)
  if (is.null(learning.rate))  learning.rate <- M / min(dim(X))
  if (is.null(rng.seed))  rng.seed <- as.integer(Sys.time())
  if (ncol(X) == length(beta0))
    X <- t(X)
  else if (nrow(X) != length(beta0))
    stop ("dim(X) not related to dim(beta0) (", length(beta0), ")")
  n.save <- floor(n.save)
  thin <- floor(thin)
  burnin <- floor(burnin)
  structure(
    .Call("btlm", X, y, beta0, lambda, tau.sq, M, include - 1, batch.size,
          n.save, thin, burnin, iter.max.sgd, eps, tol, learning.rate, mt.decay,
          vt.decay, metropolis.target, rng.seed, PACKAGE = "btglm"),
    class = "btglm")
}




btlmPostMode <- function(X, y, beta0, lambda = NULL,
                         tau.sq = 1e4, M = NULL, include = 1,
                         batch.size = min(100, nrow(X)),
                         iter.max = 1000, eps = 1e-8, tol = 1e-6,
                         learning.rate = NULL, mt.decay = 0.9, vt.decay = 0.99,
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
    .Call("btlmPostApprox", X, y, beta0, lambda, tau.sq, M, include - 1,
          batch.size, iter.max, eps, tol, learning.rate, mt.decay, vt.decay,
          rng.seed,
          PACKAGE = "btglm"),
    class = "btglm")
}




nobs.btglm <- function(object, ...)  object$N

sigma.btglm <- function(object, ...) {
  if (length(object$sigma) > 1)  mean(object$sigma) else object$sigma
}

residuals.btglm <- function(object, X, y, ...) {
  c(y - X %*% coef(object, ...))
}

logLik.btglm <- function(object, X, y, ...) {
  fam <- attr(object, "family")
  if (is.null(fam))
    ll <- sum(dnorm(residuals(object, X, y), sd = sigma(object), log = TRUE))
  else if (fam$family == "binomial")
    ll <- sum(dbinom(y, 1, fam$linkinv(X %*% coef(object)), log = TRUE))
  structure(ll, nall = nobs(object), nobs = nobs(object),
            df = sum(coef(object) != 0) + 1,
            class = "logLik")
}

BIC.btglm <- function(object, X, y, ...) {
  ll <- logLik(object, X, y)
  -2 * c(ll) + (attr(ll, "df") - 1) * log(nobs(object))
}



select.lambda.bic <- function(X, y, beta0, M, nlambda = 50,
                              lambda.min.ratio = 1 / nlambda, ...,
                              .func = btlmPostMode, .parallel = FALSE,
                              .verbose = TRUE, .cost = BIC) {
  `%_%` <- if (.parallel) `%dopar%` else `%do%`
  lambda.seq <- lambda.grid(M, nlambda, lambda.min.ratio, .log = TRUE)
  foreach (lambda = lambda.seq, .combine = "c") %_% {
    fit <- .func(X, y, beta0, M = M, ..., lambda = lambda)
    if (.verbose)
      cat ("lambda =", lambda, "\n")
    ## if (!.parallel)
    ##   beta0 <- coef(out, FALSE)
    .cost(fit, X, y)
  } ->
    objective
  se <- 1 + log(length(y))
  lambda <- max(lambda.seq[approx.minima(objective, se)])
  fit <- .func(X, y, beta0, ..., lambda = lambda)
  structure(fit,
    cv = data.frame(lambda = lambda.seq, cost = objective, se = se),
    n.folds = 1,
    cost = .cost,
    class = c("cv.btglm", "btglm")
    )
}



cv.btlm <- function(X, y, beta0, M, n.folds = 5, nlambda = 50,
                    lambda.min.ratio = 1 / nlambda,
                    .func = btlmPostMode,
                    .cost = NULL, .parallel = FALSE, .verbose = TRUE,
                    ...) {
  if (is.null(.cost))  .cost <- mse
  `%_%` <- if (.parallel) `%dopar%` else `%do%`
  lambda.seq <- lambda.grid(M, nlambda, lambda.min.ratio, .log = TRUE)
  sets <- sample.subsets(1:length(y), n.folds)
  objective <- variances <- numeric(nlambda)
  for (i in 1:length(lambda.seq)) {
    foreach (sub = sets, .combine = "c") %_% {
      fit <- .func(X[-sub, ], y[-sub], beta0, M = M, ...,
                   lambda = lambda.seq[i])
      .cost(y[sub], X[sub, ] %*% coef(fit)) * length(sub) / length(y)
    } ->
      cv.cost
    objective[i] <- sum(cv.cost)
    variances[i] <- var(cv.cost)
    if (.verbose)
      cat ("lambda =", lambda.seq[i], "\n")
  }
  lambda <- max(lambda.seq[approx.minima(objective, sqrt(variances))])
  structure(
    .func(X, y, beta0, ..., lambda = lambda),
    cv = data.frame(lambda = lambda.seq, cost = objective, se = sqrt(variances)),
    n.folds = n.folds,
    cost = .cost,
    class = c("cv.btglm", "btglm")
    )
}



plot.cv.btglm <- function(x, y, ...) {
  lam <- log(c(with(attr(x, "cv"), lambda[which.min(cost)]), x$lambda))
  ggplot2::ggplot(attr(x, "cv"), aes(log(lambda), cost)) +
    ggplot2::geom_linerange(aes(ymin = cost - se, ymax = cost + se), alpha = 0.5,
                            size = rel(0.4)) +
      ggplot2::geom_point(color = "firebrick") +
      ggplot2::geom_vline(xintercept = lam[1], linetype = "dashed", size = rel(0.25)) +
      ggplot2::geom_vline(xintercept = lam[2], linetype = "dashed", size = rel(0.25)) +
      ggplot2::labs(x = expression(ln(lambda)),
                   y = paste0(attr(x, "n.folds"), "-fold CV Error"))
}




coef.btglm <- function(object, threshold = TRUE) {
  ans <- if (is.matrix(object$coefficients))
           colMeans(object$coefficients)
         else object$coefficients
  if (threshold) {
    incl <- object$include + 1
    ans[-incl] <- ans[-incl] * (abs(ans[-incl]) > object$lambda)
  }
  ans
}



pnz <- function(object, ...)  UseMethod("pnz")

pnz.btglm <- function(object, ...) {
  if (!is.matrix(out$coefficients))
    stop ('Object "', deparse(substitute(object)), '" not fit using MCMC')
  ans <- rep(1, ncol(object$coefficients))
  incl <- object$include + 1
  ans[-incl] <- apply(object$coefficients[, -incl], 2,
                      function(x) mean(abs(x) > object$lambda))
  ans
}

