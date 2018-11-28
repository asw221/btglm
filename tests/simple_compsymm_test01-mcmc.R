
library (btglm)

setwd ("~/Dropbox/software/btglm/tests/")
source (file.path("sims", "compound.symmetric.R"))


dat <- sim.data(n = 200, p = 200, rho = 0.6, q = 5)

H <- function(x, lambda, .eps = 1e-4) {
  pcauchy(x, lambda, .eps) + 1 - pcauchy(x, -lambda, .eps)
}

system.time(
  fit0 <- btlm(dat$X[1:100, ], dat$y[1:100],
               tol = 1e-6, lambda = 4, batch.size = 25,
             iter.max = 10000, learning.rate = 0.01, burnin = 10000, n.save = 500,
             lambda.decay = 0.9999, thin = 500, tau.sq = 6 / dat$sigma.sq,
             model.size.prior = log(length(dat$y)) * dat$sigma.sq * 0.9)
)


yhat <- c(dat$X %*% coef(fit0))




B <- fit0$coefficients * H(fit0$coefficients, fit0$lambda, 0.1)
B[, fit0$include + 1] <- fit0$coefficients[, fit0$include + 1]
B <- data.frame(t(apply(B, 2, quantile, probs = c(0.025, 0.1, 0.5, 0.9, 0.975))),
                check.names = FALSE)

ggplot(B, aes(1:nrow(B), `50%`)) +
  geom_linerange(aes(ymin = `2.5%`, ymax = `97.5%`), size = rel(0.4)) +
  geom_linerange(aes(ymin = `10%`, ymax = `90%`), size = rel(0.6)) +
  geom_point(size = rel(0.6)) +
  geom_point(aes(y = dat$b), color = "firebrick", shape = "x", size = rel(2))


fit.enet <- cv.glmnet(dat$X[, -1], dat$y, nfolds = 5)

yhat.enet <- c(as.matrix((dat$X %*% coef(fit.enet, s = "lambda.min"))))



mse(dat$y[101:200], yhat[101:200])
mse(dat$y[101:200], yhat.enet[101:200])


mse(dat$y[101:200], (dat$X %*% (coef(fit0, FALSE) * pnz(fit0)))[101:200])



foreach (i = 1:100) %do% {
  fit0 <- btlm(dat$X[1:100, ], dat$y[1:100], rnorm(ncol(dat$X), sd = 3),
               tol = 1e-5, lambda = 3.5,
               iter.max = 10000, learning.rate = 0.01,
               lambda.decay = 0.9999, thin = 500, n.save = 1,
               model.size.prior = log(length(dat$y)) * dat$sigma.sq)
} %>%
  (function(models) {
    B <- t(sapply(models, "[[", "coefficients"))
    sig <- sapply(models, "[[", "sigma")
    lam <- sapply(models, "[[", "lambda")
    aggregate <- models[[1]]
    aggregate$coefficients <- B
    aggregate$sigma <- sig
    aggregate$lambda <- lam
    aggregate
  }) ->
  fit




foreach (i = 1:100, .combine = "rbind") %do% {
  fit0 <- btlm(dat$X[1:100, ], dat$y[1:100], rnorm(ncol(dat$X), sd = 3),
               tol = 1e-5, lambda = 3.5,
               iter.max = 10000, learning.rate = 0.01,
               lambda.decay = 0.9999, thin = 500, n.save = 1, burnin = 1000,
               model.size.prior = log(length(dat$y)) * dat$sigma.sq)
  c(dat$X %*% coef(fit0))
} %>%
  colMeans() ->
  yhat

## system.time(
## fit0 <- btlmPostMode(dat$X, dat$y, tol = 1e-5, beta0 = rnorm(ncol(dat$X)), lambda = 3.5,
##                      iter.max = 10000, learning.rate = 0.01)
## )


## fit.enet <- cv.glmnet(dat$X[, -1], dat$y, nfolds = 5, alpha = 0.5)
## b.enet <- c(as.matrix(coef(fit.enet, s = "lambda.min")))

## fit0 <- btlmPostMode(dat$X, dat$y, tol = 1e-5, beta0 = b.enet, lambda = 3.5,
##                      iter.max = 10000, learning.rate = 0.01)







## H <- function(x, lambda, eps = 0.1) 1 + pcauchy(x, lambda, eps) - pcauchy(x, -lambda, eps)
## h <- function(x, lambda, eps = 0.1) dcauchy(x, lambda, eps) - dcauchy(x, -lambda, eps)
## h.fd <- function(x, lambda, eps = 0.1, .h = 0.01) (H(x + .h, lambda, eps) - H(x - .h, lambda, eps)) / (2 * .h)

## x0 <- seq(-1, 1, length.out = 200)

## qplot(x0, H(x0, 0.4), geom = "line")
## qplot(x0, h(x0, 0.4), geom = "line")
## qplot(x0, h.fd(x0, 0.4), geom = "line")
## qplot(h(x0, 0.4), h.fd(x0, 0.4), geom = "line") +
##   geom_abline(slope = 1, intercept = 0, linetype = "dotted",
##               color = "red")
