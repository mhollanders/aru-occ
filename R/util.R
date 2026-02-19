# inverse logit
inv_logit <- function(x) 1 / (1 + exp(-x))

# # prepare Stan data from dataframes of deployments and detections
# prep_data <- function(deployments = deployments, 
#                       detections = detections, 
#                       camera_id = "camera_id", 
#                       deployment_date = "deployment_date",
#                       retrieval_date = "retrieval_date",
#                       species = "species") {
#   I <- nlevels(sites[[camera_id]])
#   K <- nlevels(dets[[species]])
#   Delta <- sites |>
#     mutate(date = list(seq.Date(deployment_date, retrieval_date)), 
#            .by = camera_id) |> 
#     unnest(date) |> 
#     mutate(Delta = 1L) |>
#     select(date, camera_id, Delta) |> 
#     arrange(camera_id, date) |> 
#     pivot_wider(names_from = camera_id, values_from = Delta, values_fill = 0) |>
#     arrange(date) |>
#     column_to_rownames("date") |> 
#     t()
#   dates <- ymd(colnames(Delta))
#   J <- length(dates)
#   y <- dets |>
#     count(camera_id, date, species) |> 
#     complete(camera_id = levels(sites$camera_id), 
#              date = dates, 
#              species, 
#              fill = list(n = 0)) |> 
#     arrange(species, date, camera_id) |> 
#     pull(n) |> 
#     array(c(I, J, K))
#   list(I = I, J = J, K = K, Delta = Delta, X = c(0, 0), 
#        x_i = matrix(0, 0, I), x_j = array(0, c(0, J, I)),
#        utm = sites |> select(X, Y) |> as.matrix(),
#        dates = as.integer(dates),
#        y = y, NB = 0, ZI = 0, C = 0)
# }

# correlation matrix from vector of correlations
corr_matrix <- function(rho) {
  D <- (1 + sqrt(1 + 8 * length(rho))) / 2
  O <- diag(D)
  idx <- 1
  for (i in 1:(D - 1)) {
    for (j in (i + 1):D) {
      O[j, i] <- O[i, j] <- rho[idx]
      idx <- idx + 1
    }
  }
  O
}

# squared exponential GP covariance kernel
gp_exp_quad_cov <- function(x, sigma, ell, jitter = TRUE) {
  K <- sigma^2 * exp(-0.5 * (x / ell)^2)
  if (jitter) {
    K <- K + diag(1e-9, nrow(K))
  }
  K
}

# grid of points for prediction surface
prediction_surface <- function(x_obs, res = 1000, buffer = 0.05) {
  n <- round(sqrt(res))
  int <- apply(x_obs, 2, \(x) {
    r <- range(x)
    width <- diff(r)
    seq(r[1] - buffer * width, r[2] + buffer * width, length.out = n)
  })
  expand.grid(x = int[, 1], y = int[, 2])
}

# multivariate GP predictions
multi_gp_predict <- function(x_obs, x_pred, sigma, ell, gp_f, tau, rho) {
  I <- nrow(x_pred)
  D <- ncol(gp_f)
  K_obs <- gp_exp_quad_cov(raster::pointDistance(x_obs, lonlat = F), sigma, ell)
  K_pred <- gp_exp_quad_cov(raster::pointDistance(x_pred, lonlat = F), sigma, ell)
  K_cross <- gp_exp_quad_cov(raster::pointDistance(x_obs, x_pred, lonlat = F), sigma, ell, 
                             jitter = F)
  L_obs <- chol(K_obs)
  mu_pred <- t(K_cross) %*% backsolve(L_obs, forwardsolve(t(L_obs), gp_f))
  v <- forwardsolve(t(L_obs), K_cross)
  cov_pred <- K_pred - t(v) %*% v
  L_cov <- chol(cov_pred + diag(1e-9, nrow(cov_pred)))
  pred <- mu_pred + L_cov %*% matrix(rnorm(I * D), I, D) %*% 
    chol(corr_matrix(rho)) %*% diag(tau)
  as_tibble(x_pred) |> 
    set_names("X", "Y") |> 
    mutate(i = row_number(), .before = 1) |> 
    bind_cols(as_tibble(pred) |> set_names(1:D)) |> 
    pivot_longer(-(1:3), names_to = "s", values_to = "mu") |> 
    mutate(s = as.integer(s))
}

multi_gp_predict2 <- function(x_obs, x_pred, sigma, ell, gp_f, rho) {
  I <- nrow(x_pred)
  D <- ncol(gp_f)
  K_obs <- gp_exp_quad_cov(raster::pointDistance(x_obs, lonlat = F), sigma, ell)
  K_pred <- gp_exp_quad_cov(raster::pointDistance(x_pred, lonlat = F), sigma, ell)
  K_cross <- gp_exp_quad_cov(raster::pointDistance(x_obs, x_pred, lonlat = F), sigma, ell, 
                             jitter = F)
  L_obs <- chol(K_obs)
  mu_pred <- t(K_cross) %*% backsolve(L_obs, forwardsolve(t(L_obs), gp_f))
  v <- forwardsolve(t(L_obs), K_cross)
  cov_pred <- K_pred - t(v) %*% v
  L_cov <- chol(cov_pred + diag(1e-9, nrow(cov_pred)))
  pred <- mu_pred + L_cov %*% matrix(rnorm(I * D), I, D) %*% 
    chol(corr_matrix(rho))
  as_tibble(x_pred) |> 
    set_names("X", "Y") |> 
    mutate(i = row_number(), .before = 1) |> 
    bind_cols(as_tibble(pred) |> set_names(1:D)) |> 
    pivot_longer(-(1:3), names_to = "s", values_to = "mu") |> 
    mutate(s = as.integer(s))
}

multi_gp_predict3 <- function(x_obs, x_pred, sigma, ell, gp_f, Omega) {
  I <- nrow(x_pred)
  D <- ncol(gp_f)
  K_obs <- gp_exp_quad_cov(raster::pointDistance(x_obs, lonlat = F), sigma, ell)
  K_pred <- gp_exp_quad_cov(raster::pointDistance(x_pred, lonlat = F), sigma, ell)
  K_cross <- gp_exp_quad_cov(raster::pointDistance(x_obs, x_pred, lonlat = F), sigma, ell, 
                             jitter = F)
  L_obs <- chol(K_obs)
  mu_pred <- t(K_cross) %*% backsolve(L_obs, forwardsolve(t(L_obs), gp_f))
  v <- forwardsolve(t(L_obs), K_cross)
  cov_pred <- K_pred - t(v) %*% v
  L_cov <- chol(cov_pred + diag(1e-9, nrow(cov_pred)))
  pred <- mu_pred + L_cov %*% matrix(rnorm(I * D), I, D) %*% chol(Omega)
  as_tibble(x_pred) |> 
    set_names("X", "Y") |> 
    mutate(i = row_number(), .before = 1) |> 
    bind_cols(as_tibble(pred) |> set_names(1:D)) |> 
    pivot_longer(-(1:3), names_to = "s", values_to = "mu") |> 
    mutate(s = as.integer(s))
}
