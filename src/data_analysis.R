# data analysis #

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
library(car)
library(lme4)
library(lmtest)
library(sandwich)
library(caret)
library(modelsummary)
library(officer)
library(flextable)

# data import ----
campaigndataclean <- read.csv("campaign_data_clean.csv")
campaigndataclean <- campaigndataclean[-c(1)]
campaigndataclean$category <- as.factor(campaigndataclean$category)
summary(campaigndataclean)
str(campaigndataclean)

#  estimate baseline model for hypotheses 1a,2a,3a
model1 <- lm(log_donor_count ~
               gain_loss_score +
               emotional_valence +
               inclusivity_score +
               log_goal_amount,
             campaigndataclean)
summary(model1)
AIC(model1)
BIC(model1)
# R^2 0.502, p < 0.001
# AIC: 14320.43
# BIC: 14359.56

# estimate category only model
catonlymodel <- lm(log_donor_count ~ 0 +
                     catisemergency +
                     catisevent +
                     catiseducation +
                     catisanimal +
                     catisbusiness +
                     catischarity +
                     catiscommunity +
                     catiscompetition +
                     catiscreative +
                     catisenvironment +
                     catisfaith +
                     catisfamily +
                     catismedical +
                     catismemorial +
                     catissports +
                     catistravel +
                     catisvolunteer +
                     catiswishes,
                   campaigndataclean)
summary(catonlymodel)
# cannot interpret R^2 or F-stat, but coefficients are accurate: 
# significant small differences in category intercepts

catonlymodel <- lm(log_donor_count ~
                     catisemergency +
                     catisevent +
                     catiseducation +
                     catisanimal +
                     catisbusiness +
                     catischarity +
                     catiscommunity +
                     catiscompetition +
                     catiscreative +
                     catisenvironment +
                     catisfaith +
                     catisfamily +
                     catismedical +
                     catismemorial +
                     catissports +
                     catistravel +
                     catisvolunteer +
                     catiswishes,
                   campaigndataclean)
summary(catonlymodel)
# report model fit based on this model with intercept wrt baseline category

# significant differences between categories, most are quite small though

# estimate model with categories: OLS (no intercept) ----
model1withcat <- lm(log_donor_count ~ 0 +
               catisemergency +
               catisevent +
               catiseducation +
               catisanimal +
               catisbusiness +
               catischarity +
               catiscommunity +
               catiscompetition +
               catiscreative +
               catisenvironment +
               catisfaith +
               catisfamily +
               catismedical +
               catismemorial +
               catissports +
               catistravel +
               catisvolunteer +
               catiswishes +
                gain_loss_score +
                emotional_valence +
                inclusivity_score +
                log_goal_amount,
                   campaigndataclean)
summary(model1withcat)

write.csv(as.data.frame(summary(model1withcat)$coefficients), file = "model1withcat.csv", fileEncoding = "UTF-8")

# interpret coefficients from this model

model1withcatalt <- lm(log_donor_count ~ 
               catisemergency +
               catisevent +
               catiseducation +
               catisanimal +
               catisbusiness +
               catischarity +
               catiscommunity +
               catiscompetition +
               catiscreative +
               catisenvironment +
               catisfaith +
               catisfamily +
               catismedical +
               catismemorial +
               catissports +
               catistravel +
               catisvolunteer +
               catiswishes +
               gain_loss_score +
               emotional_valence +
               inclusivity_score +
               log_goal_amount,
             campaigndataclean)
summary(model1withcatalt)
# model fit: R^2 0.5804 on p < 0.001

# fit
AIC(model1withcatalt) #13493.4
BIC(model1withcatalt) #13643.42

# extended model ----

## interaction ----
# full interaction (no intercept)
model2 <- lm(
  log_donor_count ~
    0 +
    (gain_loss_score + emotional_valence + inclusivity_score) * category +
    log_goal_amount,
  data = campaigndataclean)
summary(model2)

write.csv(as.data.frame(summary(model2)$coefficients), file = "model2.csv", fileEncoding = "UTF-8")

# again, coefficients can be interpreted, but cannot interpret model fit stats:
# estimate intercept model for fit
model2alt <- lm(
  log_donor_count ~
    (gain_loss_score + emotional_valence + inclusivity_score) * category +
    log_goal_amount,
  data = campaigndataclean)
summary(model2alt)

# R^2 0.5893, p < 0.001

# compare model performace
anova(model1,model2alt)
# interaction makes significant but small improvement

AIC(model2alt) #13488.3
BIC(model2alt) #13970.97

# information criteria also similar

AIC(model2alt) - AIC(model1) #-832.12, slight improvement
BIC(model2alt) - BIC(model1) #-388.59, slight improvement

# robustness ----
## mixed effects base model ----
baseOLSalt <- lmer(
  log_donor_count ~ gain_loss_score +
    emotional_valence +
    inclusivity_score +
    log_goal_amount +
    (1 | category),
  campaigndataclean)
summary(baseOLSalt)

AIC(baseOLSalt)
BIC(baseOLSalt)

## OOS performance: holdout sample ----
set.seed(123)

ctrl <- trainControl(
  method = "cv",
  number = 10
)

cv_model <- train(
  log_donor_count ~
    0 +
    (gain_loss_score + emotional_valence + inclusivity_score) * category +
    log_goal_amount,
  data = campaigndataclean,
  method = "lm",
  trControl = ctrl
)

cv_model
# RMSE 0.9267666
# R^2 0.5771979
# MAE 0.706614

# assumption checks ----

## linearity ----
ggplot(campaigndataclean, aes(x = emotional_valence, y = log_donor_count)) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "loess", se = TRUE) +
  labs(
    x = "Emotional valence (-1 to 1)",
    y = "Log donor count",
    title = "Emotional valence and donor count"
  )

ggplot(campaigndataclean, aes(x = gain_loss_score, y = log_donor_count)) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "loess", se = TRUE) +
  labs(
    x = "Gain/loss score",
    y = "Log donor count",
    title = "Gain/loss score and donor count"
  )

ggplot(campaigndataclean, aes(x = inclusivity_score, y = log_donor_count)) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "loess", se = TRUE) +
  labs(
    x = "Inclusivity score",
    y = "Log donor count",
    title = "Inclusivity score and donor count"
  )

ggplot(campaigndataclean, aes(x = log_goal_amount, y = log_donor_count)) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "loess", se = TRUE) +
  labs(
    x = "log goal amount",
    y = "Log donor count",
    title = "log goal amount and donor count"
  )

# some curving of fitted lines, but mostly linear where most data points appear.

## error term ----
### homoscedasticity ----
# visual test
plot(fitted(model1), resid(model1),
     xlab = "Fitted values",
     ylab = "Residuals",
     main = "Residuals vs Fitted")
abline(h = 0, col = "firebrick1", lwd = 1)

plot(model1, which = 3)
#fanning pattern down the middle of fitted values: test formally
bptest(model2)
ncvTest(model2)
#significant heteroscedasticity: try using robust errors
coeftest(model2alt, vcov = vcovHC(model2alt, type = "HC1"))

### normality of error ----
qqnorm(residuals(model2))
qqline(residuals(model2))
# looks good, only tail and head of points start to deviate slightly

## multicollinearity of predictors ----
#re-estimate baseline model with intercept, needed for sensible vif scores
model1 <- lm(log_donor_count ~ 
               category +
               gain_loss_score +
               emotional_valence +
               inclusivity_score +
               log_goal_amount,
             campaigndataclean)
summary(model1)
vif(model1)
# no vif scores above 2, no problem
