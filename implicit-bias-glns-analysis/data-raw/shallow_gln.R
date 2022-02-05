devtools::load_all()

hparams <- c('context_seed', 'momentum', 'train_size', 'shatter_dims', 'median_init')

shallow_gln_ts <- get_lightning_data(
  '../data/shallow_gln',
  hparams = hparams
) %>%
  dplyr::group_by_at(c(hparams, 'epoch', 'split', 'metric')) %>%
  dplyr::summarise(value = mean(value)) %>%
  dplyr::ungroup() %>%
  dplyr::mutate(
    value = dplyr::if_else(metric == 'acc', 1-value, value),
    metric = dplyr::if_else(metric == 'acc', 'error', metric)
  )

shallow_gln_last <- shallow_gln_ts %>%
  dplyr::filter(epoch == 500/train_size*800-1)

shallow_gln_best <- shallow_gln_ts %>%
  dplyr::group_by_at(c(hparams, 'split', 'metric')) %>%
  dplyr::summarise(value = min(value)) %>%
  dplyr::ungroup()

accs_and_bounds <- get_accs_and_bounds(
  '../data/shallow_gln_cp'
)

shallow_gln_cp <-
  accs_and_bounds[[1]] %>%
  dplyr::mutate(error = 1-acc)

shallow_gln_bounds <-
  accs_and_bounds[[2]]

usethis::use_data(shallow_gln_ts, overwrite = TRUE)
usethis::use_data(shallow_gln_last, overwrite = TRUE)
usethis::use_data(shallow_gln_best, overwrite = TRUE)
usethis::use_data(shallow_gln_cp, overwrite = TRUE)
usethis::use_data(shallow_gln_bounds, overwrite = TRUE)
