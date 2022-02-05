devtools::load_all()

hparams <- c('context_seed', 'momentum', 'latent_dims', 'train_size', 'shatter_dims', 'median_init')

deep_gln_ts <- get_lightning_data(
  '../data/deep_gln',
  hparams=hparams
) %>%
  dplyr::group_by_at(c(hparams, 'epoch', 'split', 'metric')) %>%
  dplyr::summarise(value = mean(value)) %>%
  dplyr::ungroup() %>%
  dplyr::mutate(
    value = dplyr::if_else(metric == 'acc', 1-value, value),
    metric = dplyr::if_else(metric == 'acc', 'error', metric)
  )

deep_gln_last <- deep_gln_ts %>%
  dplyr::filter(epoch == 500/train_size*800-1)

deep_gln_best <- deep_gln_ts %>%
  dplyr::group_by_at(c(hparams, 'split', 'metric')) %>%
  dplyr::summarise(value = min(value)) %>%
  dplyr::ungroup()

deep_gln_cp <- get_accs('../data/deep_gln_cp') %>%
  dplyr::select(type, acc, directory) %>%
  dplyr::mutate(
    directory = stringr::str_split_fixed(directory, stringr::coll('/'), n = 4)[,4]
  ) %>%
  tidyr::separate(directory, into = c(
    NA, 'shatter_dims', NA, 'latent_dims', NA, 'context_seed', NA, 'train_size', NA, 'median_init', NA, 'constraints'
  ), extra = 'merge') %>%
  dplyr::mutate(error = 1-acc) %>%
  dplyr::mutate(
    train_size = as.numeric(train_size), latent_dims = as.numeric(latent_dims), shatter_dims = as.numeric(shatter_dims)
  )


usethis::use_data(deep_gln_ts, overwrite = TRUE)
usethis::use_data(deep_gln_last, overwrite = TRUE)
usethis::use_data(deep_gln_best, overwrite = TRUE)
usethis::use_data(deep_gln_cp, overwrite = TRUE)
