# data analysis #

# packages

library(haven)
library(ggplot2)
library(readxl)
library(dplyr)
library(data.table)
library(readr)
library(stargazer)
library(naniar)
library(tidyverse)
library(corrplot)
library(GGally)
library(ggcorrplot)

# data import
campaigndataclean <- read.csv("campaign_data_clean.csv")
campaigndataclean <- campaigndataclean[-c(1)]
campaigndataclean$category <- as.factor(campaigndataclean$category)
summary(campaigndataclean)
str(campaigndataclean)

# baseline model: OLS ----

# extended models ----
## holdout sample ----

## interaction ----

## latent class ----

# assumptions and robustness ----



