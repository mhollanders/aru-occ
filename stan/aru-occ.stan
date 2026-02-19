functions {
  #include aru-occ.stanfunctions
  #include util.stanfunctions
}

data {
  int<lower=1> I, J;  // number of sites and surveys
  matrix<lower=0>[J, I] Delta;  // recording time per survey
  array[I, J] int<lower=0> y;  // detection history
  array[3] int<lower=0> P;  // number of site-level and site-by-survey level predictors
  matrix[P[1], I] X1;  // site covariates for occupancy
  matrix[P[2], I] X2;  // site covariates for detection
  array[I] matrix[P[3], J] X3;  // survey covariates
  array[I] vector[2] XY;  // projected site coordinates
  int<lower=0, upper=1> dirichlet;  // logistic-normal (0) or Dirichlet (1) variance decomposition
  real<lower=0> period;  // period for periodic kernel
  int<lower=0> grainsize;  // threading
  int<lower=0> D;  // number of draws of Monte Carlo integration for loo
  int<lower=0, upper=2> OD;  // Poisson without (0) or with OLRE (1), or negative binomial (2)
}

transformed data {
  int P_i = sum(P[1:2]),  // total number of site-level predictors
      P_i_max = max(P[1:2]),  // maximum number of site-level predictors
      periodic = period > 0,  // periodic GP indicator
      GP = 2 + periodic,  // number of GPs
      OLRE = OD == 1,  // observation-level random effects
      NB = OD == 2;  // negative binomial overdispersion
  array[2] int V;  // variance partitions for occupancy and detection
  V[1] = P[1];  // random slopes
  V[2] = sum(P[2:3]) + 2 + OLRE; // random slopes and site, survey, and OLRE effects
  array[I] int sites = linspaced_int_array(I, 1, I);  // site sequence
  array[J] real surveys = linspaced_array(J, 1, J);  // survey sequence
  matrix[J, I] log_Delta = log(Delta);  // offsets
  array[I, 2] int f_l = first_last_survey(Delta);  // first and last surveys
  array[I] int J_i = zeros_int_array(I),  // number of surveys per site
               Q = zeros_int_array(I);  // detections by site and species
  for (i in 1:I) {
    int f = f_l[i, 1], l = f_l[i, 2];
    for (j in f:l) {
      if (Delta[j, i] > 0) {
        J_i[i] += 1;
        Q[i] += y[i, j];
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
  
  // total variances, partition sparsity, and variance partitions
  vector<lower=0>[2] W;  // total variance of logit occupancy and log detection
  vector<lower=0>[2] theta;  // logistic-normal scale or Dirichlet concentration
  vector[sum(V) - 2] W_phi_u;  // unconstrained variance partitions
  
  // coefficients for predictors
  row_vector[P_i] beta_z;  // site coefficients
  row_vector[P[3]] gamma_z;  // survey coefficients
  
  // site and survey effects and GP length scales
  sum_to_zero_vector[I] iota_z;  // site z-scores
  sum_to_zero_vector[J] kappa_z;  // global survey z-scores
  array[periodic] simplex[2] kappa_v;  // variance partitions of periodic and exp. quad. kernels
  vector<lower=0>[GP] ell;  // GP length scales
  
  // OLRE residuals or 1/sqrt(negative binomial overdispersion)
  array[OLRE] sum_to_zero_vector[N] epsilon_z;
  vector<lower=0>[NB] inv_sqrt_phi;
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

  // coefficients and and site and survey effects
  matrix[2, P_i_max] beta = rep_matrix(0, 2, P_i_max);
  row_vector[P[3]] gamma;
  row_vector[I] iota;
  vector[J] kappa;
  {
    int tau_idx = 1;
    
    // site-level predictors
    if (P_i) {
      beta[1, :P[1]] = segment(tau[1], tau_idx, P[1]) .* head(beta_z, P[1]);
      beta[2, :P[2]] = segment(tau[2], tau_idx, P[2]) .* tail(beta_z, P[2]);
    }
    
    // survey-level predictors
    if (P[3]) {
      tau_idx += P[2];
      gamma = segment(tau[2], tau_idx, P[3]) .* gamma_z;
    }
    
    // site effects
    tau_idx += P[3];
    matrix[I, I] iota_K = gp_exp_quad_cov(XY, tau[2, tau_idx], ell[1]),
                 iota_U = cholesky_decompose(add_diag(iota_K, 1e-9))';
    iota = iota_z' * iota_U;
    
    // survey effects
    tau_idx += 1;
    matrix[J, J] kappa_K, kappa_L;
    if (periodic) {
      vector[2] sqrt_kappa_v = sqrt(kappa_v[1]);
      kappa_K = gp_exp_quad_cov(surveys, sqrt_kappa_v[1], ell[2])
                + gp_periodic_cov(surveys, sqrt_kappa_v[2], ell[3], period);
    } else {
      kappa_K = gp_exp_quad_cov(surveys, 1, ell[2]);
    }
    kappa_L = cholesky_decompose(add_diag(kappa_K, 1e-9));
    kappa = tau[2, tau_idx + 1] * kappa_L * kappa_z;
  }
  
  // negative binomial overdispersion
  vector[NB] phi = square(inv(inv_sqrt_phi));
  
  // occupancy
  row_vector[I] logit_psi = logit(psi_bar) + beta[1, :P[1]] * X1;
  
  // priors
  real lprior = beta_lpdf(psi_bar | 2, 2)
                + gamma_lpdf(mu_bar | 1, 4)
                + student_t_lpdf(W | 3, 0, 2.5)
                + gamma_lpdf(theta | 1, 1)
                + inv_gamma_lpdf(ell[1] | 3, 1)
                + inv_gamma_lpdf(ell[2] | 3, 1);
  if (periodic) {
    lprior += inv_gamma_lpdf(ell[3] | 2, 1);
  }
  if (NB) {
    lprior += exponential_lpdf(inv_sqrt_phi | 2);
  }
}

model {
  target += lprior
            + std_normal_lupdf(beta_z)
            + std_normal_lupdf(gamma_z)
            + std_normal_lupdf(iota_z)
            + std_normal_lupdf(kappa_z);
  for (d in 1:2) {
    target += dirichlet ?
              dirichlet_lupdf(W_phi[:V[d], d] | rep_vector(inv(theta[d]), V[d]))
              : std_normal_lupdf(W_phi_z[:V[d], d]);
  }
  
  // multivariate normal residuals for Poisson
  vector[OLRE * N] epsilon;
  if (OLRE) {
    epsilon = tau[2, V[2]] * epsilon_z[1];
  }
                           
  // likelihood
  target += grainsize ?
            reduce_sum(partial_aru_occ_lupmf, sites, grainsize, y, Q, f_l, 
                       log_Delta, J_i, X2, X3, logit_psi, log(mu_bar), 
                       beta[2, :P[2]], gamma, iota, kappa, epsilon, phi)
            : aru_occ_lupmf(y | Q, f_l, log_Delta, X2, X3, logit_psi, 
                            log(mu_bar), beta[2, :P[2]], gamma, iota, kappa, 
                            epsilon, phi);
}

generated quantities {
  // log likelihood, latent occupancy, and posterior predictions
  vector[I] log_lik;
  vector[(D > 0) * I] log_lik2;
  array[I] int z = ones_int_array(I);
  array[I, J] int yrep = rep_array(0, I, J);
  array[I] int zrep, Qrep = zeros_int_array(I);
  
  {
    // reconstruct log likelihood and latent states
    vector[OLRE * N] epsilon, epsilon_rep;
    if (OLRE) {
      epsilon = tau[2, V[2]] * epsilon_z[1];
    }
    array[N] int zeros = zeros_int_array(N), ones = ones_int_array(N);
    tuple(vector[I], matrix[2, I], matrix[J, I]) lp =
      aru_occ(y, Q, f_l, log_Delta, X2, X3, logit_psi, log(mu_bar), 
              beta[2, :P[2]], gamma, iota, kappa, epsilon, phi);
    log_lik = lp.1;
    for (i in 1:I) {
      if (!Q[i]) {
        z[i] = bernoulli_logit_rng(lp.2[2, i] - log_lik[i]);
      }
    }
    matrix[J, I] log_mu = lp.3;
    
    // Monte Carlo for loo
    if (D) {
      int tau_idx = sum(P[2:3]) + 1;
      matrix[I, I] iota_K, iota_U;
      row_vector[I] iota_rep;
      matrix[I, D] log_lik_k;
      for (d in 1:D) {
        iota_rep = to_row_vector(normal_rng(zeros[:I], ones[:I]));
        iota_K = gp_exp_quad_cov(XY, tau[2, tau_idx], ell[1]);
        iota_U = cholesky_decompose(add_diag(iota_K, 1e-9))';
        iota_rep = iota_rep * iota_U;
        if (OLRE) {
          epsilon_rep = tau[2, V[2]] * to_vector(normal_rng(zeros, ones));
        }
        lp = aru_occ(y, Q, f_l, log_Delta, X2, X3, logit_psi, log(mu_bar), 
                     beta[2, :P[2]], gamma, iota_rep, kappa, epsilon_rep, phi);
        log_lik_k[:, d] = lp.1;
      }
      log_lik2 = rep_vector(-log_D, I);
      for (i in 1:I) {
        log_lik2[i] += log_sum_exp(log_lik_k[i]);
      }
      
      // produce posterior predictive OLREs regardless
    } else if (OLRE) {
      epsilon_rep = tau[2, V[2]] * to_vector(normal_rng(zeros, ones));
    }
    
    // posterior predictions
    int n = 0;
    for (i in 1:I) {
      int f = f_l[i, 1], l = f_l[i, 2];
      zrep[i] = bernoulli_logit_rng(logit_psi[i]);
      if (OLRE) {
        for (j in f:l) {
          if (Delta[j, i] > 0) {
            n += 1;
            log_mu[j, i] += epsilon_rep[n] - epsilon[n];
          }
        }
      }
      if (zrep[i]) {
        for (j in f:l) {
          if (Delta[j, i] > 0) {
            log_mu[j, i] = min({ log_mu[j, i], 20 });
            yrep[i, j] = NB ?
                         neg_binomial_2_log_rng(log_mu[j, i], phi[1])
                         : poisson_log_rng(log_mu[j, i]);
          }
        }
        Qrep[i] = sum(yrep[i, f:l]);
      }
    }
  }
}
