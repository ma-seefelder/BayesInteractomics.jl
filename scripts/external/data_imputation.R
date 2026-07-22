library(mice)
library(dplyr)
library(magrittr)
library(readxl)

# Load data
setwd("C:/Users/manue/Proton Drive/manuel.seefelder/My files/HTT_interactome_meta_analysis")

data <- read_excel("dataset.xlsx") %>%
  as.data.frame()

data_matrix <- data[, c(2:ncol(data))] %>% as.matrix()
data <- data[, 1]
# Impute missing values
m <- 5
imputed <- mice(data_matrix, printFlag = FALSE, m = m, maxit = 20)
print("Data imputed")

# export
imp <- complete(imputed, action = "a")

for (i in 1:5) {
  tmp <- cbind(data, imp[[i]])
  # Renamed from dataset_imp_<i>.csv to dataset_imp_mice_<i>.csv to disambiguate from dataset_mnar.xlsx
  # `load_data(...; imputation = :mar)` accepts both old and new filenames in v1.2; old name removed in v1.3
  write.csv(tmp, paste0("dataset_imp_mice_", i, ".csv"))
}



#import CSV, DataFrames, XLSX
#for i in 1:5
#  x = CSV.read("imputed_data/dataset_imp_mice_$i.csv", DataFrames.DataFrame)[:, 2:end]
#  XLSX.writetable("imputed_data/dataset_imp_mice_$i.xlsx", x)
#end