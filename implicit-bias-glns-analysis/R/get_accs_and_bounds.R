get_accs_and_bounds <- function(dir){
  accs <- list()
  bounds <- list()
  for(subdir in list.dirs(dir)) {
    if('accs.csv' %in% list.files(subdir)){
      df_accs <- readr::read_csv(paste0(subdir, '/accs.csv')) %>%
        dplyr::select(-...1) %>%
        mutate(median_init = stringr::str_split(subdir, pattern='mi_')[[1]][2])
      df_bounds <- dplyr::bind_cols(
        readr::read_csv(paste0(subdir, '/bounds.csv')) %>%
          dplyr::select(-...1),
        df_accs %>%
          dplyr::select(-type, -acc) %>%
          dplyr::distinct()
      )
      accs <- c(accs, list(df_accs))
      bounds <- c(bounds, list(df_bounds))
    }
  }
  return(list(dplyr::bind_rows(accs), dplyr::bind_rows(bounds)))
}

get_accs <- function(dir){
  accs <- list()
  bounds <- list()
  for(subdir in list.dirs(dir)) {
    if('accs.csv' %in% list.files(subdir)){
      df_accs <- readr::read_csv(paste0(subdir, '/accs.csv')) %>%
        dplyr::select(-...1) %>%
        mutate(directory = subdir)
      accs <- c(accs, list(df_accs))
    }
  }
  return(dplyr::bind_rows(accs))
}
