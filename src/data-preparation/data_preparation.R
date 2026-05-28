####################
# data preparation #
####################

# packages ----
library(haven)
library(ggplot2)
library(readxl)
library(dplyr)
library(data.table)
library(readr)
library(janitor)
library(stargazer)

# data import ----
campaigndataraw <- read.csv("gofundmecampaigns.csv")

# variable structuring: ensure each variable has correct type ----
campaigndataraw$category <- as.factor(campaigndataraw$category)
campaigndataraw$goal_amount <- as.integer(campaigndataraw$goal_amount)
campaigndataraw$donor_count <- as.integer(campaigndataraw$donor_count)
campaigndataraw$gain_loss_score <- as.numeric(campaigndataraw$gain_loss_score)
campaigndataraw$emotional_valence <- as.numeric(campaigndataraw$emotional_valence)
campaigndataraw$inclusivity_score <- as.numeric(campaigndataraw$inclusivity_score)
  
# raw inspection of descriptives ----
summary(campaigndataraw)
str(campaigndataraw)
sd(campaigndataraw$goal_amount, na.rm = TRUE)
sd(campaigndataraw$donor_count, na.rm = TRUE)
sd(campaigndataraw$gain_loss_score, na.rm = TRUE)
sd(campaigndataraw$emotional_valence, na.rm = TRUE)
sd(campaigndataraw$inclusivity_score, na.rm = TRUE)

# interpretation:
# Goal amount: min 1, a bit illogical, but at least not 0 or negative; 
# max 15000000, high but still realistic as a campaign run by GoFundMe. 11/5040 NA's, very small, likely bug
# Donor count: min 1, makes sense;
# max 25388, also good.
# IV scores: min -1, max +1 (+0.9687 for emotion), looks good;
# 2/5040 NA's, likely bug
# goal amount and donor count intgers, good.
# scores numeric, also good.

# create dummies for each category
campaigndataraw$catisemergency[campaigndataraw$category=="Emergency"] <- 1
campaigndataraw$catisevent[campaigndataraw$category=="Event"] <- 1
campaigndataraw$catiseducation[campaigndataraw$category=="Education"] <- 1
campaigndataraw$catisanimal[campaigndataraw$category=="Animal"] <- 1
campaigndataraw$catisbusiness[campaigndataraw$category=="Business"] <- 1
campaigndataraw$catischarity[campaigndataraw$category=="Charity"] <- 1
campaigndataraw$catiscommunity[campaigndataraw$category=="Community"] <- 1
campaigndataraw$catiscompetition[campaigndataraw$category=="Competition"] <- 1
campaigndataraw$catiscreative[campaigndataraw$category=="Creative"] <- 1
campaigndataraw$catisenvironment[campaigndataraw$category=="Environment"] <- 1
campaigndataraw$catisfaith[campaigndataraw$category=="Faith"] <- 1
campaigndataraw$catisfamily[campaigndataraw$category=="Family"] <- 1
campaigndataraw$catismedical[campaigndataraw$category=="Medical"] <- 1
campaigndataraw$catismemorial[campaigndataraw$category=="Memorial"] <- 1
campaigndataraw$catissports[campaigndataraw$category=="Sports"] <- 1
campaigndataraw$catistravel[campaigndataraw$category=="Travel"] <- 1
campaigndataraw$catisvolunteer[campaigndataraw$category=="Volunteer"] <- 1
campaigndataraw$catiswishes[campaigndataraw$category=="Wishes"] <- 1

# replacing the NA's in the dummies with 0
campaigndataraw[, 8:25][is.na(campaigndataraw[, 8:25])] <- 0

# save to new csv
write.csv(campaigndataraw, file = "campaign_data_prepped.csv", fileEncoding = "UTF-8")
