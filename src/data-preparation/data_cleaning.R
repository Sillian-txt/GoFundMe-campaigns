# data cleaning #

# packages ----

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

# data import ----

campaigndataprepped <- read.csv("campaign_data_prepped.csv")
campaigndataprepped <- campaigndataprepped[-c(1)]
summary(campaigndataprepped)
str(campaigndataprepped)
campaigndataprepped$category <- as.factor(campaigndataprepped$category)

# handling missing values ----

colSums(is.na(campaigndataprepped))

# all IV scores: 2 NA's
# logically such a small fraction of dataset: 
# these NA's are likely due to a script bug/interrupt -> listwise deletion

campaigndataprepped <- campaigndataprepped %>%
  filter(
    !is.na(gain_loss_score),
    !is.na(emotional_valence),
    !is.na(inclusivity_score)
  )

## MCAR test ----
# goal_amount: 11 NA's
# again, tiny fraction, run little's MCAR test for sanity
mcar_test_result <- mcar_test(
  campaigndataprepped %>%
    select(goal_amount, donor_count, gain_loss_score,
           emotional_valence, inclusivity_score)
)
print(mcar_test_result)

# high p.value, supports MCAR, listwise deletion appropriate
campaigndataprepped <- campaigndataprepped %>%
  filter(!is.na(goal_amount))

# outlier handling ----
# significant outliers in donor_count and goal_amount
# rightskew is expected in donation data
## tukey's fences ----
tukey_fences <- function(x, label) {
  q1 <- quantile(x, 0.25, na.rm = TRUE)
  q3 <- quantile(x, 0.75, na.rm = TRUE)
  iqr <- q3 - q1
  lower <- q1 - 1.5 * iqr
  upper <- q3 + 1.5 * iqr
  n_out <- sum(x < lower | x > upper, na.rm = TRUE)
  invisible(list(lower = lower, upper = upper, n_outliers = n_out))
}

fences_goal  <- tukey_fences(campaigndataprepped$goal_amount,  "goal_amount ")
fences_donor <- tukey_fences(campaigndataprepped$donor_count,  "donor_count ")

# fences_goal: 553 outliers
# fences_donor: 574 outiers

rawgoalplot <- ggplot(campaigndataprepped, aes(x = goal_amount)) +
  geom_histogram(bins = 80, fill = "gold", colour = "black", linewidth = 0.5) +
  scale_x_continuous(labels = scales::comma) +
  labs(title = "Raw distribution: goal_amount",
       x = "Goal amount (£)", y = "Count") +
  theme_minimal()

rawcountplot <- ggplot(campaigndataprepped, aes(x = donor_count)) +
  geom_histogram(bins = 80, fill = "firebrick1", colour = "black", linewidth = 0.5) +
  scale_x_continuous(labels = scales::comma) +
  labs(title = "Raw distribution: donor_count",
       x = "Donor count", y = "Count") +
  theme_minimal()

ggsave("plot_raw_goal.png",  rawgoalplot, width = 7, height = 4)
ggsave("plot_raw_donor.png", rawcountplot, width = 7, height = 4)

print(rawcountplot)
print(rawgoalplot)
# likely no errors in this distribution, so deletion not justified
print(campaigndataprepped %>%
        arrange(desc(goal_amount)) %>%
        select(campaign_id, category, goal_amount, donor_count) %>%
        head(30))

print(campaigndataprepped %>%
        arrange(desc(donor_count)) %>%
        select(campaign_id, category, goal_amount, donor_count) %>%
        head(30))

# Both goal_amount and donor_count show skewed distribution: long upper tail
# logtransform
campaigndataprepped <- campaigndataprepped %>%
  mutate(
    log_goal_amount  = log(goal_amount),
    log_donor_count  = log(donor_count)
  )

print(summary(campaigndataprepped$log_goal_amount))
print(summary(campaigndataprepped$log_donor_count))

#check for remaining outliers
fences_loggoal <- tukey_fences(campaigndataprepped$log_goal_amount, "log_goal_amount")
fences_logdonor <- tukey_fences(campaigndataprepped$log_donor_count, "log_donor_count")

print(fences_loggoal)
print(fences_logdonor)

#log_goal: 80
#log_donor: 48
#not many, but check for influence when modelling

# plots ----
## boxplot distributions of log transformed variables ----

ggplot(campaigndataprepped,
       aes(x = category, y = log_goal_amount)) +
  geom_boxplot(fill = "dodgerblue") +
  labs(
    x = "Category",
    y = "log(goal_amount)",
    title = "log(goal_amount) distribution per campaign category"
  ) +
  theme(
    axis.text.x = element_text(
      angle = 45,
      hjust = 1
    )
  )

ggplot(campaigndataprepped,
       aes(x = category, y = log_donor_count)) +
  geom_boxplot(fill = "firebrick1") +
  labs(
    x = "Category",
    y = "log(donor_count)",
    title = "log(donor_count) distribution per campaign category"
  ) +
  theme(
    axis.text.x = element_text(
      angle = 45,
      hjust = 1
    )
  )

## density plots of final logtransformed variables ----

ggplot(campaigndataprepped, aes(x = log_donor_count)) +
  geom_density(fill = "firebrick1", alpha = 0.3)

ggplot(campaigndataprepped, aes(x = log_goal_amount)) +
  geom_density(fill = "dodgerblue", alpha = 0.4)

## NLP scores distribution ----
# Check proportion at floor (−1) and ceiling (+1): if >10% hit the boundary,
# that indicates a scoring artefact rather than variation
for (var in c("gain_loss_score", "emotional_valence", "inclusivity_score")) {
  n_floor   <- sum(campaigndataprepped[[var]] == -1, na.rm = TRUE)
  n_ceiling <- sum(campaigndataprepped[[var]] ==  1, na.rm = TRUE)
  n_total   <- sum(!is.na(campaigndataprepped[[var]]))
  cat(sprintf(
    "%s: floor(−1) = %d (%.1f%%)  ceiling(+1) = %d (%.1f%%)\n",
    var,
    n_floor,   100 * n_floor   / n_total,
    n_ceiling, 100 * n_ceiling / n_total
  ))
}

# gain/loss has high ceiling rate (19.4%)
# inclusivity has high floor rate (22.3%)

## boxplots of final NLP scores per category ----
ggplot(campaigndataprepped,
       aes(x = category, y = gain_loss_score)) +
  geom_boxplot(fill = "yellow") +
  labs(
    x = "Category",
    y = "Gain_loss_score",
    title = "Gain_loss_score distribution per campaign category"
  ) +
  theme(
    axis.text.x = element_text(
      angle = 45,
      hjust = 1
    )
  )

ggplot(campaigndataprepped,
       aes(x = category, y = emotional_valence)) +
  geom_boxplot(fill = "green") +
  labs(
    x = "Category",
    y = "Emotional_valence",
    title = "Emotional_valence distribution per campaign category"
  ) +
  theme(
    axis.text.x = element_text(
      angle = 45,
      hjust = 1
    )
  )

ggplot(campaigndataprepped,
       aes(x = category, y = inclusivity_score)) +
  geom_boxplot(fill = "purple") +
  labs(
    x = "Category",
    y = "Inclusivity_score",
    title = "Inclusivity_score distribution per campaign category"
  ) +
  theme(
    axis.text.x = element_text(
      angle = 45,
      hjust = 1
    )
  )

## density plots of final NLP variables ----

ggplot(campaigndataprepped, aes(x = gain_loss_score)) +
  geom_density(fill = "yellow", alpha = 0.3)

ggplot(campaigndataprepped, aes(x = emotional_valence)) +
  geom_density(fill = "green", alpha = 0.3)

ggplot(campaigndataprepped, aes(x = inclusivity_score)) +
  geom_density(fill = "purple", alpha = 0.3)

## correlation & scatter plot of variables ----

corr_vars <- campaigndataprepped[, c(
  "gain_loss_score",
  "emotional_valence",
  "inclusivity_score",
  "log_goal_amount",
  "log_donor_count"
)]

corr_mat <- cor(corr_vars, use = "complete.obs")

ggcorrplot(
  corr_mat,
  type = "upper",
  hc.order = TRUE,
  lab = TRUE,
  lab_size = 3,
  colors = c("#D73027", "white", "#4575B4")
)

# descriptives of final dataset ----
summary(campaigndataprepped)
sd(campaigndataprepped$goal_amount, na.rm = TRUE)
sd(campaigndataprepped$donor_count, na.rm = TRUE)
sd(campaigndataprepped$gain_loss_score, na.rm = TRUE)
sd(campaigndataprepped$emotional_valence, na.rm = TRUE)
sd(campaigndataprepped$inclusivity_score, na.rm = TRUE)
sd(campaigndataprepped$log_goal_amount, na.rm = TRUE)
sd(campaigndataprepped$log_donor_count, na.rm = TRUE)

# save to new csv ----
write.csv(campaigndataprepped, file = "campaign_data_clean.csv", fileEncoding = "UTF-8")

