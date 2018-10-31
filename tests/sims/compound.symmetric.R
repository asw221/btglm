

sim.data <- function(n = 500, p = 1000, rsq = 0.5, q = 6, rho = 0.9,
                     family = gaussian()) {
  Lt <- chol(matrix(rho, p, p) + diag(1 - rho, p, p))
  X <- matrix(rnorm(n * p), n, p) %*% Lt
  b <- c(rnorm(1), rep(5, q-1) * sample(c(-1, 1), q-1, TRUE), rep(0, p - q))
  sigma.sq <- var(c(X %*% b)) %>% (function(x) x * (1 - rsq) / rsq)
  y <- rnorm(n, X %*% b, sqrt(sigma.sq))
  if (family$family == "binomial") {
    y <- as.numeric(y > 0)
  }
  params <- list(n = n, p = p, rsq = rsq, q = q, rho = rho)
  list (X = X, y = y, b = b, sigma.sq = sigma.sq, params = params)
}
