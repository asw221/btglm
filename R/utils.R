

close.match <- function(x, y, tol = 1e-4) {
  abs(x - y) <= tol
}

approx.minima <- function(x, tol = 1e-4, position = TRUE) {
  ndx <- which(close.match(min(x), x, tol))
  if (position) ndx else x[ndx]
}


mse <- function(y, yhat) {
  c(crossprod(y - yhat)) / length(y)
}


bin.dev <- function(y, yhat) {
  -2 * sum(y * log(pmax(yhat, 1e-8)) + (1 - y) * log(pmin(yhat, 1 - 1e-8))) / length(y)
}


lambda.grid <- function(lambda.max, ngrid = 50, min.ratio = 1 / ngrid, .log = TRUE) {
  end.pts <- c(lambda.max * min.ratio, lambda.max)
  if (.log) {
    end.pts <- log(end.pts)
    exp(seq(end.pts[1], end.pts[2], length.out = ngrid))
  }
  else
    seq(end.pts[1], end.pts[2], length.out = ngrid)
}


sample.subsets <- function(x, n.subsets = 5) {
  tapply(sample(x), rep(1:n.subsets, length.out = length(x)), c)
}


