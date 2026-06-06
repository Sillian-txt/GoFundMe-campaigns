# How message framing influences fundraising success

This repository contains all additional information related to the processes of data collection, parsing, preparation and analysis for my master's thesis [Framing Charity: How Message Framing Influences Donation Decisions on GoFundMe].

## Data description
The data study used a scraped dataset from the public GoFundMe website. Data of 5040 campaigns were processed across GoFundMe's 18 categories. 

## This Repository's structure
```
├── README.md
├── Durkstra (2026).pdf
├── .gitignore
├── data
├── gen
|   ├── temp
|   └── output
└── src
    ├── analysis
    └── data-preparation
```

## Dependencies
- Python and the following packages:

```
pip install bs4
pip install selenium
pip install empath
pip install nltk
```

- NLTK dictionaries
- R and the following packages:

```
install.packages(c(library(haven)
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
library(caret)))
```
