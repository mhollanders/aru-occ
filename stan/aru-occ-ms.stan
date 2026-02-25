functions {
  #include aru-occ.stanfunctions
  #include util.stanfunctions
}

data {
  int<lower=1> I, J, S;  // number of sites, surveys, and species
  matrix<lower=0>[I, J] Delta;  // recording time per survey
  array[I, J, S] int<lower=0> y;  // detection history
  array[3] int<lower=0> P;  // number of site-level and site-by-survey level predictors
  matrix[P[1], I] X1;  // site covariates for occupancy
  matrix[P[2], I] X2;  // site covariates for detection
  array[I] matrix[P[3], J] X3;  // survey covariates
  array[I] vector[2] XY;  // projected site coordinates
  int<lower=0, upper=1> dirichlet,  // logistic-normal (0) or Dirichlet (1) variance decomposition
                        SS;  // species-specific length scales
  real<lower=0> period;  // period for periodic kernel
  int<lower=0> grainsize;  // threading
  int<lower=0> D;  // number of draws of Monte Carlo integration for loo
  int<lower=0, upper=2> OD;  // Poisson without (0) or with OLRE (1), or negative binomial (2)
}

transformed data {
  int P_i = sum(P[1:2]),  // total number of site-level predictors
      P_i_max = max(P[1:2]),  // maximum number of site-level predictors
      SP = sum(XY[1]) != 0,  // spatial GP indicator
      periodic = period > 0,  // periodic GP indicator
      GP = SP + 1 + periodic,  // number of GPs
      OLRE = OD == 1,  // observation-level random effects
      NB = OD == 2,  // negative binomial overdispersion
      G = (P[1] > 0) + (P[2] > 0) + (P[3] > 0) + 2 + OLRE;  // number of species-level correlation matrices
  array[2] int V;  // variance partitions for occupancy and detection
  V[1] = 1 + 2 * P[1];  // random intercept and slopes
  V[2] = 1 + 2 * sum(P[2:3]) // random intercept and slopes
         + (2 + OLRE) * (1 + S);  // site, survey, and OLRE effects
  array[I] int sites = linspaced_int_array(I, 1, I);  // site sequence
  array[J] real surveys = linspaced_array(J, 1, J);  // survey sequence
  matrix[J, I] log_Delta = log(Delta');  // offsets
  array[I, 2] int f_l = first_last_survey(Delta);  // first and last surveys
  array[I] int J_i = zeros_int_array(I);  // number of surveys per site
  array[I, S] int Q = rep_array(0, I, S);  // detections by site and species
  for (i in 1:I) {
    int f = f_l[i, 1], l = f_l[i, 2];
    for (j in f:l) {
      if (!is_inf(log_Delta[j, i])) {
        J_i[i] += 1;
        for (s in 1:S) {
          Q[i, s] += y[i, j, s];
        }
      }
    }
  }
  int N = sum(J_i);  // total surveys
  real log_D = log(D);
}

parameters {
  // intercepts and interspecific correlations
  real<lower=0, upper=1> psi_bar;  // occupancy
  real<lower=0> mu_bar;  // detection
  cholesky_factor_corr[2] alpha_O_L;  // occupancy/detection correlation
  array[2] sum_to_zero_vector[S] alpha_z;  // z-scores
  array[G] cholesky_factor_corr[S] O_L;
  
  // total variances, partition sparsity, and variance partitions
  vector<lower=0>[2] W;  // total variance of logit occupancy and log detection
  vector<lower=0>[2] theta;  // logistic-normal scale or Dirichlet concentration
  vector[sum(V) - 2] W_phi_u;  // unconstrained variance partitions
  
  // coefficients for predictors
  row_vector[P_i] beta_bar_z;  // global site coefficients
  array[2, P_i] sum_to_zero_vector[S] beta_z;  // site coefficient z-scores
  row_vector[P[3]] gamma_bar_z;  // global survey coefficients
  array[P[3]] sum_to_zero_vector[S] gamma_z;  // survey coefficient z-scores
  
  // site and survey effects and GP length scales
  sum_to_zero_vector[I] iota_bar_z;  // global site z-scores
  sum_to_zero_matrix[S, I] iota_z;  // species-level site z-scores
  sum_to_zero_vector[J] kappa_bar_z;  // global survey z-scores
  sum_to_zero_matrix[S, J] kappa_z;  // species-level survey z-scores
  simplex[2] kappa_v;  // variance partitions of periodic and exp. quad. kernels
  vector<lower=0>[GP] ell;  // GP length scales
  
  // species-specific GP length scales
  vector<lower=0>[SS * GP] ell_bar, ell_t;  // GP length scales means and scales for species functions
  array[SS * GP] sum_to_zero_vector[S] ell_z;  // species-level length scale z-scores
  
  // OLRE residuals or 1/sqrt(negative binomial overdispersion)
  array[OLRE] sum_to_zero_vector[N] epsilon_bar_z;  // global residual z-scores
  array[OLRE] sum_to_zero_matrix[S, N] epsilon_z;  // species-level residual z-scores
  vector<lower=0>[NB] inv_sqrt_phi_bar, inv_sqrt_phi_t;  // OD mean and scale
  array[NB] sum_to_zero_vector[S] inv_sqrt_phi_z;  // species-level z-scores
}

transformed parameters {
  // variance partitions (Dirichlet or logistic-normal) and scales
  matrix[V[2], 2] W_phi_z, W_phi = rep_matrix(0, V[2], 2);
  if (dirichlet) {
    W_phi[:V[1], 1] = simplex_jacobian(head(W_phi_u, V[1] - 1));
    W_phi[:, 2] = simplex_jacobian(tail(W_phi_u, V[2] - 1));
  } else {
    W_phi_z[:V[1], 1] = sum_to_zero_jacobian(head(W_phi_u, V[1] - 1));
    W_phi[:V[1], 1] = softmax(theta[1] * W_phi_z[:V[1], 1]);
    W_phi_z[:, 2] = sum_to_zero_jacobian(tail(W_phi_u, V[2] - 1));
    W_phi[:, 2] = softmax(theta[2] * W_phi_z[:, 2]);
  }
  matrix[2, V[2]] tau = sqrt(diag_post_multiply(W_phi, W))';
  
  // bivariate normal intercepts
  matrix[S, 2] alpha = rep_matrix([ logit(psi_bar), log(mu_bar) ], S)
                       + append_col(alpha_z[1], alpha_z[2]) 
                         * diag_post_multiply(alpha_O_L', tau[:, 1]);
  
  // species-level length scales
  matrix[SS * S, GP] ell_s;
  if (SS) {
    for (g in 1:GP) {
      ell_s[:, g] = exp(log(ell_bar[g]) + ell_t[g] * ell_z[g]);
    }
  }
                         
  // species-level coefficients, and site and survey effects
  matrix[2, P_i_max] beta_bar = rep_matrix(0, 2, P_i_max);
  array[2] matrix[S, P_i_max] beta;
  row_vector[P[3]] gamma_bar;
  matrix[S, P[3]] gamma;
  row_vector[I] iota_bar = iota_bar_z';
  matrix[S, I] iota;
  row_vector[J] kappa_bar = kappa_bar_z';
  matrix[S, J] kappa;
  {
    int tau_idx = 2, GP_idx = 1, O_idx = 0;
    
    // site-level predictors
    if (P_i) {
      beta_bar[1, :P[1]] = segment(tau[1], tau_idx, P[1]) 
                           .* head(beta_bar_z, P[1]);
      beta_bar[2, :P[2]] = segment(tau[2], tau_idx, P[2]) 
                           .* tail(beta_bar_z, P[2]);
      for (d in 1:2) {
        O_idx += P[d] ? 1 : 0;
        matrix[S, P[d]] beta_z_mat;
        for (p in 1:P[d]) {
          beta_z_mat[:, p] = beta_z[d, p];
        }
        beta[d, :, :P[d]] = 
          rep_matrix(beta_bar[d, :P[d]], S)
          + O_L[O_idx] 
            * diag_post_multiply(beta_z_mat[:, :P[d]], 
                                 segment(tau[d], tau_idx + P[d], P[d]));
      }
    }
    
    // survey-level predictors
    if (P[3]) {
      O_idx += 1;
      tau_idx += 2 * P[2];
      gamma_bar = segment(tau[2], tau_idx, P[3]) .* gamma_bar_z;
      row_vector[P[3]] gamma_t = segment(tau[2], tau_idx + 1, P[3]);
      matrix[S, P[3]] gamma_z_mat;
      for (p in 1:P[3]) {
        gamma_z_mat[:, p] = gamma_z[p];
      }
      gamma = rep_matrix(gamma_bar, S)
              + O_L[O_idx] * diag_post_multiply(gamma_z_mat, gamma_t);
    }
    
    // site effects
    tau_idx += 2 * P[3];
    O_idx += 1;
    iota_bar *= tau[2, tau_idx];
    matrix[I, I] iota_K, iota_U;
    if (SP) {
      GP_idx += 1;
      iota_K = gp_exp_quad_cov(XY, 1, ell[1]);
      iota_U = cholesky_decompose(add_diag(iota_K, 1e-9))';
      iota_bar *= iota_U;
    }
    iota = rep_matrix(iota_bar, S);
    matrix[S, I] iota_s = diag_pre_multiply(segment(tau[2], tau_idx + 1, S), 
                                            O_L[O_idx]) * iota_z;
    if (SP) {
      if (SS) {
        for (s in 1:S) {
          iota_K = gp_exp_quad_cov(XY, 1, ell_s[s, 1]);
          iota_U = cholesky_decompose(add_diag(iota_K, 1e-9))';
          iota_s[s] *= iota_U;
        }
      } else {
        iota_s *= iota_U;
      }
    }
    iota += iota_s;
    
    // survey effects
    tau_idx += 1 + S;
    O_idx += 1;
    GP_idx += 1;
    matrix[J, J] kappa_K, kappa_U;
    vector[periodic * 2] kappa_t;
    if (periodic) {
      kappa_t = tau[2, tau_idx] * sqrt(kappa_v);
      kappa_K = gp_exp_quad_cov(surveys, kappa_t[1], ell[GP_idx])
                + gp_periodic_cov(surveys, kappa_t[2], ell[GP_idx + 1], period);
    } else {
      kappa_K = gp_exp_quad_cov(surveys, tau[2, tau_idx], ell[2]);
    }
    kappa_U = cholesky_decompose(add_diag(kappa_K, 1e-9))';
    kappa_bar *= kappa_U;
    kappa = rep_matrix(kappa_bar, S);
    matrix[S, J] kappa_s = diag_pre_multiply(segment(tau[2], tau_idx + 1, S), 
                                             O_L[O_idx]) * kappa_z;
    if (SS) {
      kappa_t = sqrt(kappa_v);
      for (s in 1:S) {
        if (periodic) {
          kappa_K = gp_exp_quad_cov(surveys, kappa_t[1], ell_s[s, 2])
                    + gp_periodic_cov(surveys, kappa_t[2], ell_s[s, 3], period);
        } else {
          kappa_K = gp_exp_quad_cov(surveys, 1, ell_s[s, 2]);
        }
        kappa_U = cholesky_decompose(add_diag(kappa_K, 1e-9))';
        kappa_s[s] *= kappa_U;
      }
    } else {
      kappa_s *= kappa_U;
    }
    kappa += kappa_s;
  }
  
  // negative binomial overdispersion
  vector[NB * S] inv_sqrt_phi, phi;
  if (NB) {
    inv_sqrt_phi = log(inv_sqrt_phi_bar[1]) + inv_sqrt_phi_t[1] * inv_sqrt_phi_z[1];
    phi = exp(-2 * inv_sqrt_phi);
  }
  
  // occupancy
  matrix[S, I] logit_psi = rep_matrix(alpha[:, 1], I) + beta[1, :, :P[1]] * X1;
  
  // priors
  real lprior = beta_lpdf(psi_bar | 2, 2)
                + gamma_lpdf(mu_bar | 1, 4)
                + student_t_lpdf(W | 3, 0, 2.5)
                + gamma_lpdf(theta | 1, 1)
                + lkj_corr_cholesky_lpdf(alpha_O_L | 1)
                + inv_gamma_lpdf(ell[1] | 3, 1)
                + inv_gamma_lpdf(ell[2] | 3, 1);
  if (periodic) {
    lprior += inv_gamma_lpdf(ell[3] | 2, 1);
  }
  for (g in 1:G) {
    lprior += lkj_corr_cholesky_lpdf(O_L[g] | 1);
  }
  if (SS) {
    lprior += inv_gamma_lpdf(ell_bar[1] | 3, 1)
              + inv_gamma_lpdf(ell_bar[2] | 3, 1)
              + inv_gamma_lpdf(ell_bar[3] | 2, 1)
              + exponential_lpdf(ell_t | 2);
  }
  if (NB) {
    lprior += exponential_lpdf(inv_sqrt_phi_bar[1] | 2);
    lprior += exponential_lpdf(inv_sqrt_phi_t[1] | 2);
  }
}

model {
  target += lprior
            + std_normal_lupdf(beta_bar_z)
            + std_normal_lupdf(gamma_bar_z)
            + std_normal_lupdf(iota_bar_z)
            + std_normal_lupdf(to_vector(iota_z))
            + std_normal_lupdf(kappa_bar_z)
            + std_normal_lupdf(to_vector(kappa_z));
  for (d in 1:2) {
    target += dirichlet ?
              dirichlet_lupdf(W_phi[:V[d], d] | rep_vector(inv(theta[d]), V[d]))
              : std_normal_lupdf(W_phi_z[:V[d], d]);
    target += std_normal_lupdf(alpha_z[d]);
    for (p in 1:P_i) {
      target += std_normal_lupdf(beta_z[d, p]);
    }
  }
  for (p in 1:P[3]) {
    target += std_normal_lupdf(gamma_z[p]);
  }
  if (SS) {
    for (g in 1:3) {
      target += std_normal_lupdf(ell_z[g]);
    }
  }
  if (NB) {
    target += std_normal_lupdf(inv_sqrt_phi_z[1]);
  } else if (OLRE) {
    target += std_normal_lupdf(epsilon_bar_z[1])
              + std_normal_lupdf(to_vector(epsilon_z[1]));
  }
  
  // multivariate normal residuals for Poisson
  matrix[OLRE * S, N] epsilon;
  if (OLRE) {
    epsilon = rep_matrix(tau[2, V[2] - S - 1] * epsilon_bar_z[1]', S)
              + diag_pre_multiply(tail(tau[2], S), O_L[G]) * epsilon_z[1];
  }
                           
  // likelihood
  target += grainsize ?
            reduce_sum(partial_aru_occ_ms_lupmf, sites, grainsize, y, Q, f_l,
                       log_Delta, J_i, X2, X3, logit_psi, alpha[:, 2], 
                       beta[2, :, :P[2]], gamma, iota, kappa, epsilon, phi)
            : aru_occ_ms_lupmf(y | Q, f_l, log_Delta, X2, X3, logit_psi, 
                               alpha[:, 2], beta[2, :, :P[2]], gamma, iota, 
                               kappa, epsilon, phi);
}

generated quantities {
  // correlations
  real alpha_rho = multiply_lower_tri_self_transpose(alpha_O_L)[1, 2];
  array[G] corr_matrix[S] O;
  for (g in 1:G) {
    O[g] = multiply_lower_tri_self_transpose(O_L[g]);
  }
  
  // log likelihood, latent occupancy, and posterior predictions
  matrix[S, I] log_lik;
  matrix[(D > 0) * S, I] log_lik2;
  array[I, S] int z = rep_array(1, I, S);
  array[I, J, S] int yrep = rep_array(0, I, J, S);
  array[I, S] int zrep, Qrep = rep_array(0, I, S);
  
  {
    // reconstruct log likelihood and latent states
    int SN = S * N;
    row_vector[OLRE * N] epsilon_bar;
    matrix[OLRE * S, N] epsilon, epsilon_rep;
    array[SN] int zeros = zeros_int_array(SN), ones = ones_int_array(SN);
    if (OLRE) {
      epsilon_bar = tau[2, V[2] - S - 1] * epsilon_bar_z[1]';
      epsilon = rep_matrix(epsilon_bar, S)
                + diag_pre_multiply(tail(tau[2], S), O_L[G]) * epsilon_z[1];
    }
    tuple(matrix[S, I], array[I] matrix[S, 2], array[I] matrix[S, J]) lp =
      aru_occ_ms(y, Q, f_l, log_Delta, X2, X3, logit_psi, alpha[:, 2], 
                 beta[2, :, :P[2]], gamma, iota, kappa, epsilon, phi);
    log_lik = lp.1;
    for (i in 1:I) {
      for (s in 1:S) {
        if (!Q[i, s]) {
          z[i, s] = bernoulli_logit_rng(lp.2[i, s, 2] - log_lik[s, i]);
        }
      }
    }
    array[I] matrix[S, J] log_mu = lp.3;
    
    // Monte Carlo for loo
    if (D) {
      int tau_idx = 1 + 2 * sum(P[2:3]) + 2, 
          O_idx = (P[1] > 0) + (P[2] > 0) + (P[3] > 0) + 1;
      matrix[I, I] iota_K, iota_U;
      matrix[S, I] iota_s, iota_rep;
      array[D] matrix[S, I] log_lik_k;
      for (d in 1:D) {
        iota_s = to_matrix(normal_rng(zeros[:S * I], ones[:S * I]), S, I);
        iota_s = diag_pre_multiply(segment(tau[2], tau_idx, S), O_L[O_idx])
                     * iota_s;
        if (SP) {
          if (SS) {
            for (s in 1:S) {
              iota_K = gp_exp_quad_cov(XY, 1, ell_s[s, 1]);
              iota_U = cholesky_decompose(add_diag(iota_K, 1e-9))';
              iota_s[s] *= iota_U;
            }
          } else {
            iota_K = gp_exp_quad_cov(XY, 1, ell[1]);
            iota_U = cholesky_decompose(add_diag(iota_K, 1e-9))';
            iota_s *= iota_U;
          }
        }
        iota_rep = rep_matrix(iota_bar, S) + iota_s;
        if (OLRE) {
          epsilon_rep = to_matrix(normal_rng(zeros, ones), S, N);
          epsilon_rep = rep_matrix(epsilon_bar, S)
                        + diag_pre_multiply(tail(tau[2], S), O_L[G]) 
                          * epsilon_rep;
        }
        lp = aru_occ_ms(y, Q, f_l, log_Delta, X2, X3, logit_psi, alpha[:, 2], 
                        beta[2, :, :P[2]], gamma, iota_rep, kappa, epsilon_rep, 
                        phi);
        log_lik_k[d] = lp.1;
      }
      log_lik2 = rep_matrix(-log_D, S, I);
      for (i in 1:I) {
        for (s in 1:S) {
          log_lik2[s, i] += log_sum_exp(log_lik_k[:, s, i]);
        }
      }
      
      // produce posterior predictive OLREs regardless
    } else if (OLRE) {
      epsilon_rep = to_matrix(normal_rng(zeros, ones), S, N);
      epsilon_rep = rep_matrix(epsilon_bar, S)
                    + diag_pre_multiply(tail(tau[2], S), O_L[G]) 
                      * epsilon_rep;
    }
    
    // posterior predictions
    matrix[S, J] log_mu_i;
    int n = 0;
    for (i in 1:I) {
      int f = f_l[i, 1], l = f_l[i, 2];
      zrep[i] = bernoulli_logit_rng(logit_psi[:, i]);
      log_mu_i = log_mu[i];
      if (OLRE) {
        for (j in f:l) {
          if (!is_inf(log_Delta[j, i])) {
            n += 1;
            log_mu_i[:, j] += epsilon_rep[:, n];
          }
        }
      }
      for (s in 1:S) {
        if (zrep[i, s]) {
          for (j in f:l) {
            if (!is_inf(log_Delta[j, i])) {
              log_mu_i[s, j] = min({ log_mu_i[s, j], 20 });
              yrep[i, j, s] = NB ?
                              neg_binomial_2_log_rng(log_mu_i[s, j], phi[s])
                              : poisson_log_rng(log_mu_i[s, j]);
            }
          }
          Qrep[i, s] = sum(yrep[i, f:l, s]);
        }
      }
    }
  }
}
