library(imputeTS)
library(caret)
library(dplyr)
library(randomForest)
library(Metrics)
library(rattle)
library(rpart)
library(rpart.plot)
library(MASS)

# Read in data from GSA
# data source can be found at Calc.GSA.gov
dataframe <- read.csv("government_contracting_rates.csv")

# Carry forward prices into next year if there is no pricing data for year +1 or year +2
dataframe$Next.Year.Labor.Price <- na_replace(dataframe$Next.Year.Labor.Price, dataframe$Current.Year.Labor.Price)
dataframe$Second.Year.Labor.Price <- na_replace(dataframe$Second.Year.Labor.Price, dataframe$Next.Year.Labor.Price)

# Convert data format for the dated elements
dataframe$Begin.Date <- as.Date(dataframe$Begin.Date)
dataframe$End.Date <- as.Date(dataframe$End.Date)

# omit rows with NA to simplify modeling
dataframe <- na.exclude(dataframe)

#########################
## Feature Engineering ##
#########################

# convert all text in labor category titles to lowercase to simplify key word matching
dataframe$Labor.Category <- tolower(dataframe$Labor.Category)

# create dummy variables for key words in LCAT titles
dataframe$Technician <- grepl("technician", dataframe$Labor.Category)
dataframe$Admistrative <- grepl("administrative", dataframe$Labor.Category)
dataframe$Clerk <- grepl("clerk", dataframe$Labor.Category)
dataframe$Engineer <- grepl("engineer", dataframe$Labor.Category)
dataframe$Architect <- grepl("architect", dataframe$Labor.Category)
dataframe$Analyst <- grepl("analyst", dataframe$Labor.Category)
dataframe$Program <- grepl("program", dataframe$Labor.Category)
dataframe$Project <- grepl("project", dataframe$Labor.Category)
dataframe$Manager <- grepl("manager", dataframe$Labor.Category)
dataframe$Director <- grepl("director", dataframe$Labor.Category)
dataframe$Expert <- grepl("expert", dataframe$Labor.Category)
dataframe$Executive <- grepl("executive", dataframe$Labor.Category)
dataframe$Principal <- grepl("principal", dataframe$Labor.Category)
dataframe$Lead <- grepl("lead", dataframe$Labor.Category)
dataframe$Senior <- grepl("senior", dataframe$Labor.Category)
dataframe$Junior <- grepl("junior", dataframe$Labor.Category)
dataframe$Journeyman <- grepl("journeyman", dataframe$Labor.Category)
dataframe$Partner <- grepl("partner", dataframe$Labor.Category)
dataframe$Strategy <- grepl("strategy", dataframe$Labor.Category)
dataframe$Actuary <- grepl("actuary", dataframe$Labor.Category)
dataframe$President <- grepl("president", dataframe$Labor.Category)
dataframe$Clevel <- grepl("c-level", dataframe$Labor.Category)
dataframe$Coach <- grepl("coach", dataframe$Labor.Category)
dataframe$ERP <- grepl("erp", dataframe$Labor.Category)
dataframe$Strategic <- grepl("strategic", dataframe$Labor.Category)
dataframe$Producer <- grepl("producer", dataframe$Labor.Category)
dataframe$Sr <- grepl("sr.", dataframe$Labor.Category)
dataframe$Jr <- grepl("jr.", dataframe$Labor.Category)
dataframe$Consultant <- grepl("consultant", dataframe$Labor.Category)
dataframe$Interpreter <- grepl("interpreter", dataframe$Labor.Category)

# subset for just the variables to use in the model
dataframe <- subset(dataframe, select=  -c(Contract.., SIN, Vendor.Name, Labor.Category, Next.Year.Labor.Price, Second.Year.Labor.Price))

# partition data into training and test set
dataframe <- dataframe %>% mutate(row_id = row_number())
set.seed(82)
train <- dataframe %>% sample_frac(.75)
test  <- anti_join(dataframe, train, by = 'row_id')

# remove Row ID so that the model won't attempt to train on these items
train <- subset(train, select= -c(row_id))
test <- subset(test, select= -c(row_id))

####################
#  RANDOM FOREST   #
####################

# Train RF 
RF <- randomForest(Current.Year.Labor.Price ~ ., data = train, ntree = 200, mtry = 12, nodesize = 2, importance = TRUE, proximity = FALSE)

# Run model against test set
RFPred <- predict(RF, test)

# score model using RMSE
rmse(test$Current.Year.Labor.Price, RFPred)

# Plot results versus predictions
plotframe <- data.frame( predicted = RFPred, actual = test$Current.Year.Labor.Price, ed = test$education.Level)
ggplot(data = plotframe, aes( x = actual, y = predicted)) + geom_point() + labs(x = "Actual Bill Rate (current year)", y = "Predicted Bill Rate", title = "Bi-variate analysis, Random Forest Model") + geom_abline(slope = 1, intercept = 0)

# View relative feature importance
importance(RF)
varImpPlot(RF)


########################
#  Linear Regression   #
########################

# Fit the Full Model
LM <- lm(Current.Year.Labor.Price ~ ., data = train)

# Run model against test set
LMPred <- predict(LM, test)

# score model using RMSE
rmse(test$Current.Year.Labor.Price, LMPred)

# Plot results versus predictions
plotframe <- data.frame( predicted = LMPred, actual = test$Current.Year.Labor.Price, ed = test$education.Level)
ggplot(data = plotframe, aes( x = actual, y = predicted)) + geom_point() + labs(x = "Actual Bill Rate (current year)", y = "Predicted Bill Rate", title = "Bi-variate analysis, Linear Regression Model") + geom_abline(slope = 1, intercept = 0)

## Repeat prediction and scoring for Stepwise regression ##

# Stepwise regression model
step.model <- stepAIC(LM, direction = "forward", trace = FALSE) 
# tried forward and backward stepping.  No meaningful difference in model performance
summary(step.model)

# Run model against test set
StepwisePred <- predict(step.model, test)

# score model using RMSE
rmse(test$Current.Year.Labor.Price, StepwisePred)

# Plot results versus predictions
plotframe <- data.frame( predicted = StepwisePred, actual = test$Current.Year.Labor.Price, ed = test$education.Level)
ggplot(data = plotframe, aes( x = actual, y = predicted)) + geom_point() + labs(x = "Actual Bill Rate (current year)", y = "Predicted Bill Rate", title = "Bi-variate analysis, Stepwise Linear Regression Model") + geom_abline(slope = 1, intercept = 0)

## Repeat prediction and scoring for linear regression with 2 way interactions ##

LMinteractions <- lm(Current.Year.Labor.Price ~ . ^2, data = train)

# Stepwise regression model
interaction.model <- stepAIC(LMinteractions, direction = "forward", trace = FALSE) 

# tried forward and backward stepping.  No meaningful difference in model performance
summary(interaction.model)

# Run model against test set
InteractionPred <- predict(interaction.model, test)

# Some very wonky outliers.  Set upper bound at $600 and lower bound @ $11 to match original data set
InteractionPred <- ifelse(InteractionPred > 600, 600, InteractionPred)
InteractionPred <- ifelse(InteractionPred < 11, 11, InteractionPred)

# score model using RMSE
rmse(test$Current.Year.Labor.Price, InteractionPred)

# Plot results versus predictions
plotframe <- data.frame( predicted = InteractionPred, actual = test$Current.Year.Labor.Price, ed = test$education.Level)
ggplot(data = plotframe, aes( x = actual, y = predicted)) + geom_point() + labs(x = "Actual Bill Rate (current year)", y = "Predicted Bill Rate", title = "Bi-variate analysis, Stepwise Linear Regression Model with two way interactions") + geom_abline(slope = 1, intercept = 0)

## Repeat prediction and scoring for linear regression using log transformation for the dependant variable ##

#create log-transformed variable - remove current year labor price from data set.  
train$LogX <- log(train$Current.Year.Labor.Price)
test$LogX <- log(test$Current.Year.Labor.Price)
train <- subset(train, select = -c(Current.Year.Labor.Price))

# Fit the model
LMlog <- lm(LogX ~ ., data = train)

# Run model against test set
LogPred <- predict(LMlog, test)

#unwind the log transformation 
LogPred <- exp(LogPred)

# score model using RMSE
rmse(test$Current.Year.Labor.Price, LogPred)

# Plot results versus predictions
plotframe <- data.frame(predicted = LogPred, actual = test$Current.Year.Labor.Price, ed = test$education.Level)
ggplot(data = plotframe, aes( x = actual, y = predicted)) + geom_point() + labs(x = "Bill Rate (current year)", y = "Predicted Bill Rate", title = "Bi-variate analysis, Linear Regression Model using log transform for DV") + geom_abline(slope = 1, intercept = 0)
