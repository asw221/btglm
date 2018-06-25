

test_that("Medium sized regression problem, independent X",
{

  library (glmnet)
  library (btglm)

  rtnorm <- function(n, q, mean = 0, sd = 1, tail = TRUE) {
    if (tail)
      p <- runif(n, pnorm(abs(q), mean, sd), 1)
    else
      p <- runif(n, 0.5, pnorm(abs(q), mean, sd))
    qnorm(p, mean, sd) * sign(runif(n, -1, 1))
  }

  objective <- function(X, y, beta, tau.sq, lambda) {
    b <- beta * (abs(beta) > lambda)
    -0.5 * c(crossprod(y - X %*% b)  - 0.5 * crossprod(beta) / tau.sq)
  }


  n <- 100
  p <- 20
  q <- 5
  rho <- 0.6
  lambda <- 0.7
  prec0 <- 1 / 1000
  beta <- c(rnorm(1, sd = 2), rtnorm(q - 1, lambda), rep(0, p - q))
  Sigma <- matrix(0, p, p)
  Sigma[1:(2*q), 1:(2*q)] <- rho
  diag (Sigma) <- 1
  X <- MASS::mvrnorm(n, rep(0, p), Sigma)
  sigma.sq <- var(as.vector(X %*% beta)) * (1 - 0.5) / 0.5^2
  y <- rnorm(n, as.vector(X %*% beta), sqrt(sigma.sq))

  tau.sq <- 100
  b0 <- coef(glmnet::glmnet(X[, -1], y, standardize = FALSE, lambda = 1/tau.sq, alpha = 0))@x
  b1 <- c(as.matrix(
    coef(glmnet::cv.glmnet(X[, -1], y, standardize = FALSE, alpha = 1), s = "lambda.min")))
  M <- max(abs(b0))

  out <- btlmPostMode(X, y, b0, 0.5, iter.max = 2000)

  lambda.sq <- seq(0.01, M, length.out = 50)
  obj <- sapply(lambda.sq, function(ll) btlmPostMode(X, y, b0, ll, iter.max = 50)$objective)


  out <- .Call("tlmPostApprox", X, y, b0, lambda, tau.sq, 50, 1000, 1e-8, 1e-6,
               M / max(dim(X)), 0.9, 0.999, 123,
               PACKAGE = "btglm")

  cbind(truth = beta, sgd = out$coefficients, ridge = b0, lasso = b1) %>% round(2)
  c(objective(X, y, beta, tau.sq, lambda),
    objective(X, y, out$coefficients, tau.sq, lambda),
    objective(X, y, b0, tau.sq, lambda),
    objective(X, y, b1, tau.sq, lambda))

  cbind(grad(b0, y, X, lambda, tau.sq),
        .Call("lmGradient", b0, X, y, tau.sq, lambda, PACKAGE = "btglm"),
        grad2(b0, y, X, lambda, tau.sq)) %>%
    round(3)

})




compare.plot <- function(x, y, thresh) {
  qplot(x, y) +
    geom_abline(intercept = 0, slope = 1, color = "gray60") +
    geom_hline(yintercept = thresh, size = rel(0.2), linetype = "dashed") +
    geom_hline(yintercept = -thresh, size = rel(0.2), linetype = "dashed")
}



out <- btlmPostMode(data$X, data$y, b0, 2.99, iter.max = 50000, learning.rate = 0.2, vt.decay = 0.92)

for (i in 1:10) {
out <- btlmPostMode(data$X, data$y, coef(out), 2.5, iter.max = 100, learning.rate = 0.02)
print (out$objective)
print (out$convergance)
compare.plot(data$b, coef(out), out$lambda) +
  labs(x = expression(beta^(truth)), y = expression(hat(beta)))
}





