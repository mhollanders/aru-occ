# aggregate counts by number of days
aggregate_by_days <- function(dates, reference, days = 1) {
  assert_that(inherits(dates, "Date"),
              msg = "`dates` must be dates.")
  assert_that(inherits(reference, "Date"),
              msg = "`reference` must be a date.")
  assert_that(days %% 1 == 0,
              msg = "`days` must be an integer.")
  reference + as.integer(dates - reference) %/% days * days
}

# assign detections to a cluster
assign_clusters <- function(times, gap) {
  cluster <- integer(length(times))
  current_cluster <- 1L
  last_time <- times[1]
  for (i in seq_along(times)) {
    if (as.double(difftime(times[i], last_time, units = "mins")) >= gap) {
      current_cluster <- current_cluster + 1L
      last_time <- times[i]
    }
    cluster[i] <- current_cluster
  }
  cluster
}

# prepare data for Stan
prep_data <- function(deployments, detections, 
                      site = site, start = start, end = end, 
                      projected_X, projected_Y, timestamp = timestamp,
                      species = species, count = count, days = 1, reference_date, 
                      day_start = "midday", minutes, failures = NULL, 
                      failure_start = failure_start, failure_end = failure_end, 
                      occupancy_site_predictors = NULL,
                      detection_site_predictors = NULL,
                      survey_predictors = NULL, date = date) {
  # get global reference date from earliest deployment
  if (missing(reference_date)) {
    reference_date <- deployments |>
      pull({{ start }}) |>
      min()
  } else {
    assert_that(inherits(reference_date, "Date"),
                msg = "`reference_date` must be NULL or a date.")
  }
  
  # data checks
  assert_that(identical(deployments |> 
                          pull({{ site }}) |> 
                          levels(),
                        detections |> 
                          pull({{ site }}) |> 
                          levels()),
              msg = "Factor levels for `site` are not identical for `deployments` and `detections`.")
  
  assert_that(deployments |> pull({{ start }}) |> inherits("Date") &&
                deployments |> pull({{ end }}) |> inherits("Date"),
              msg = "`start` and `end` columns in `deployments` must be dates.")
  
  assert_that(missing(minutes) || minutes > 0,
              msg = "`minutes` must be a positive real.")
  
  if (!is.null(failures)) {
    assert_that(identical(deployments |> 
                            pull({{ site }}) |> 
                            levels(),
                          failures |> 
                            pull({{ site }}) |> 
                            levels()),
                msg = "Factor levels for `site` are not identical for `deployments` and `failures`.")
    
    assert_that(failures |> pull({{ failure_start }}) |> inherits("Date") &&
                  failures |> pull({{ failure_end }}) |> inherits("Date"),
                msg = "`failure_start` and `failure_end` columns in `failures` must be dates.")
  }
  
  assert_that(detections |> pull({{ timestamp }}) |> inherits("POSIXt"),
              msg = "`timestamp` must be a date_times.")
  
  assert_that(days %% 1 == 0,
              msg = "`days` must be an integer.")
  
  assert_that(day_start %in% c("midnight", "midday"),
              msg = "`day_start` must be 'midnight' or 'midday'.")
  
  if (!is.null(occupancy_site_predictors)) {
    assert_that(identical(deployments |> 
                            pull({{ site }}) |> 
                            levels(),
                          occupancy_site_predictors |> 
                            pull({{ site }}) |> 
                            levels()),
                msg = "Factor levels for `site` are not identical for `deployments` and `occupancy_site_predictors`.")
  }
  
  if (!is.null(detection_site_predictors) &&
      !all.equal(occupancy_site_predictors, detection_site_predictors)) {
    assert_that(identical(deployments |> 
                            pull({{ site }}) |> 
                            levels(),
                          detection_site_predictors |> 
                            pull({{ site }}) |> 
                            levels()),
                msg = "Factor levels for `site` are not identical for `deployments` and `detection_site_predictors`.")
  }
  
  if (!is.null(survey_predictors)) {
    assert_that(identical(deployments |> 
                            pull({{ site }}) |> 
                            levels(),
                          survey_predictors |> 
                            pull({{ site }}) |> 
                            levels()),
                msg = "Factor levels for `site` are not identical for `deployments` and `survey_predictors`.")
    first_last <- survey_predictors |> 
      summarise(first = min({{ date }}), last = max({{ date }}))
    assert_that(first_last$first <= reference_date &&
                  first_last$last >= max(pull(deployments, {{ end }})),
                msg = "`survey_predictors` don't cover full deployment, or do not start on provided `reference_date`.")
  }
  
  # check for detections outside start and end dates
  outside <- detections |> 
    left_join(deployments, by = join_by({{ site }})) |> 
    mutate(outside = !between(as_date({{ timestamp }} ), 
                              {{ start }} , {{ end }} )) |> 
    filter(outside)
  if (nrow(outside)) {
    message(glue::glue("`detections` contains {nrow(outside)} timestamps outside of `start` and `end` dates for some sites. These detections are filtered out."))
  }
  
  # number of sites and species
  site_lvl <- deployments |>
    pull({{ site }}) |>
    levels()
  I <- length(site_lvl)
  species_lvl <- detections |>
    pull({{ species }}) |>
    levels()
  S <- length(species_lvl)
  
  # create daily deployment grid first
  daily_grid <- deployments |>
    select({{ site }}, {{ start }}, {{ end }}) |>
    mutate(date = list(seq.Date({{ start }}, {{ end }})), .by = {{ site }}) |>
    unnest(c(date))
  
  # add Delta indicator or incorporate failure dates
  if (is.null(failures)) {
    daily_grid <- daily_grid |> 
      mutate(Delta = 1L)
  } else {
    failure_dates <- failures |>
      select({{ site }}, {{ failure_start }}, {{ failure_end }}) |>
      rowwise() |>
      mutate(date = list(seq.Date({{ failure_start }} + days(1), 
                                  {{ failure_end }}))) |>
      unnest(c(date)) |>
      mutate(failure = 1L) |>
      select({{ site }}, date, failure)
    
    daily_grid <- daily_grid |> 
      left_join(failure_dates, by = join_by({{ site }}, date)) |>
      mutate(Delta = if_else(is.na(failure), 1L, 0L)) |>
      select(-failure)
  }
  
  # aggregate to desired time unit and produce fractional Delta
  deployments_aggregated <- daily_grid |> 
    mutate(survey = aggregate_by_days(date, reference_date, days)) |> 
    summarise(Delta = sum(Delta) / days, .by = c({{ site }}, survey))
  
  Delta <- deployments_aggregated |>
    pivot_wider(names_from = {{ site }},
                values_from = Delta,
                values_fill = 0,
                names_sort = TRUE) |>
    arrange(survey) |> 
    column_to_rownames("survey") |> 
    t()
  surveys <- colnames(Delta)
  J <- length(surveys)
  
  # thin detection history (messy)
  if (!missing(minutes)) {
    detections <- detections |> 
      mutate(cluster = assign_clusters({{ timestamp }}, minutes),
             .by = c({{ site }}, {{ species }})) |> 
      slice_max({{ count }}, with_ties = FALSE, 
                by = c({{ site }}, {{ species }}, cluster))
  }
  
  # aggregate detection history
  detections_aggregated <- detections |>
    mutate(date = as_date({{ timestamp }} - 
                            hours(ifelse(day_start == "midnight", 0, 12))),
           survey = aggregate_by_days(date, reference_date, days)) |>
    filter(survey %in% surveys) |>
    summarise(n = sum({{ count }}), .by = c({{ site }}, survey, {{ species }}))
  
  # create detection array
  y <- detections_aggregated |> 
    complete({{ species }} := factor(species_lvl, species_lvl),
             survey := ymd(surveys), 
             {{ site }} := factor(site_lvl, site_lvl),
             fill = list(n = 0)) |> 
    arrange({{ species }}, survey, {{ site }}) |> 
    pull(n) |> 
    array(c(I, J, S), dimnames = list(site_lvl, surveys, species_lvl))
  
  if (missing(projected_X) | missing(projected_X)) {
    XY <- matrix(0, I, 2, dimnames = list(site_lvl, c("X", "Y")))
  } else {
    XY <- deployments |>
      arrange({{ site }}) |>
      select({{ site }}, {{ projected_X }}, {{ projected_Y }}) |>
      column_to_rownames(rlang::as_name(rlang::ensym(site))) |> 
      as.matrix()
  }
  
  # site covariates
  P <- c(0, 0, 0)
  if (is.null(occupancy_site_predictors)) {
    X1 <- matrix(nrow = 0, ncol = I)
  } else {
    X1 <- occupancy_site_predictors |>
      arrange({{ site }}) |>
      column_to_rownames(rlang::as_name(rlang::ensym(site))) |>
      t()
    P[1] <- nrow(X1)
  }
  if (is.null(detection_site_predictors)) {
    X2 <- matrix(nrow = 0, ncol = I)
  } else if (all.equal(occupancy_site_predictors, detection_site_predictors)) {
    P[2] <- P[1]
    X2 <- X1
  } else {
    X2 <- detection_site_predictors |>
      arrange({{ site }}) |>
      column_to_rownames(rlang::as_name(rlang::ensym(site))) |>
      as.matrix() |> 
      t()
    P[2] <- nrow(X2)
  }
  if (is.null(survey_predictors)) {
    X3 <- array(dim = c(I, 0, J))
  } else {
    X3_lvl <- survey_predictors |> 
      select(-c({{ site }}, {{ date }})) |> 
      colnames()
    P[3] <- length(X3_lvl)
    X3 <- survey_predictors |>
      filter({{ date }} >= reference_date) |> 
      mutate(date = aggregate_by_days({{ date }}, reference_date, days)) |>
      summarise(across(where(is.numeric), mean), 
                .by = c({{ site }}, {{ date }})) |>
      pivot_longer(-c({{ site }}, {{ date }}), names_to = "p") |>
      mutate(p = factor(p, levels = X3_lvl)) |> 
      arrange({{ date }}, p, {{ site }}) |> 
      pull(value) |> 
      array(c(I, P[3], J), dimnames = list(site_lvl, X3_lvl, surveys))
  }
  
  # return
  list(I = I, J = J, S = S, Delta = Delta, y = y[, , 1:S], XY = XY, P = P, 
       X1 = X1, X2 = X2, X3 = X3, days = days)
}

# add Stan default to output of dh()
append_defaults <- function(dh, dirichlet = 1, period = 0, grainsize = 0,
                            D = 0, OD = 0, SS = 0) {
  append(dh, list(dirichlet = dirichlet, period = period, grainsize = grainsize, 
                  D = D, OD = OD, SS = SS))
}
