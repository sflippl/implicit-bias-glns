devtools::load_all()

hparams <- c('hidden_dims', 'init_seed', 'momentum', 'train_size')

relu_net_ts <- get_lightning_data(
  '../data/relu_net',
  hparams=hparams
) %>%
  dplyr::group_by_at(c(hparams, 'epoch', 'split', 'metric')) %>%
  dplyr::summarise(value = mean(value)) %>%
  dplyr::ungroup() %>%
  dplyr::mutate(
    value = dplyr::if_else(metric == 'acc', 1-value, value),
    metric = dplyr::if_else(metric == 'acc', 'error', metric)
  )

relu_net_last <- relu_net_ts %>%
  dplyr::filter(epoch == 500/train_size*800-1)

relu_net_best <-
  relu_net_ts %>%
  dplyr::group_by_at(c(hparams, 'split', 'metric')) %>%
  dplyr::summarise(value = min(value)) %>%
  dplyr::ungroup()

relu_net_cp <- get_accs('../data/relu_net') %>%
  dplyr::select(type, acc, model, objective) %>%
  dplyr::mutate(
    model = stringr::str_split_fixed(model, stringr::coll('/'), n = 4)[,3]
  ) %>%
  tidyr::separate(model, into = c(
    NA, 'hidden_dims', NA, 'init_seed', NA, 'momentum', NA, 'train_size'
  ), sep = "[_,-]") %>%
  dplyr::mutate(error = 1-acc) %>%
  dplyr::mutate(
    train_size = as.numeric(train_size), hidden_dims = as.numeric(hidden_dims)
  )


usethis::use_data(relu_net_ts, overwrite = TRUE)
usethis::use_data(relu_net_last, overwrite = TRUE)
usethis::use_data(relu_net_best, overwrite = TRUE)
usethis::use_data(relu_net_cp, overwrite = TRUE)
