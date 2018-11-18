
library (btglm)

setwd ("~/Dropbox/software/btglm/tests/")
source (file.path("sims", "compound.symmetric.R"))


dat <- sim.data(n = 100, p = 50, rho = 0.5, q = 5)



system.time(
fit0 <- btlm(dat$X, dat$y, tol = 1e-5, lambda = 3.5,
             iter.max = 10000, learning.rate = 0.01,
             lambda.decay = 0.9999,
             model.size.prior = log(length(dat$y)) * dat$sigma.sq)
)




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
