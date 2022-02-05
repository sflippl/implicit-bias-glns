get_lightning_data <- function(dir, hparams=c()) {
  lst <- list()
  for(subdir in list.dirs(dir)) {
    if('metrics.csv' %in% list.files(subdir)){
      df_metrics <-
        readr::read_csv(paste0(subdir, '/metrics.csv'))
      if(ncol(df_metrics) > 0) {
        df_metrics <-
          df_metrics %>%
          tidyr::pivot_longer(cols = !c(epoch, step)) %>%
          tidyr::separate('name', into=c('split', 'metric')) %>%
          dplyr::filter(!is.na(value))
        df_hparams <-
          yaml::read_yaml(paste0(subdir, '/hparams.yaml')) %>%
          magrittr::extract(hparams)
        for(param in hparams){
          if(length(df_hparams[[param]])>1){
            if(param == 'shatter_dims'){
              df_hparams[[param]] <- df_hparams[[param]][1]
            }
          }
        }
        df_hparams <-
          df_hparams %>%
          dplyr::bind_cols()
        lst = c(lst, list(dplyr::bind_cols(df_metrics, df_hparams)))
      }
    }
  }
  return(dplyr::bind_rows(lst))
}
