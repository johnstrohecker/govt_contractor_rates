library(imputeTS)
library(caret)
library(dplyr)
library(randomForest)
library(Metrics)
library(rattle)
library(rpart)
library(rpart.plot)
library(MASS)

# Read in data
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

# create dummy variables for key words in LCAT titles
dataframe$Technician <- grepl("Technician", dataframe$Labor.Category)
dataframe$Admistrative <- grepl("Administrative", dataframe$Labor.Category)
dataframe$Clerk <- grepl("Clerk", dataframe$Labor.Category)
dataframe$Engineer <- grepl("Engineer", dataframe$Labor.Category)
dataframe$Architect <- grepl("Architect", dataframe$Labor.Category)
dataframe$Analyst <- grepl("Analyst", dataframe$Labor.Category)
dataframe$Program <- grepl("Program", dataframe$Labor.Category)
dataframe$Project <- grepl("Project", dataframe$Labor.Category)
dataframe$Manager <- grepl("Manager", dataframe$Labor.Category)
dataframe$Director <- grepl("Director", dataframe$Labor.Category)
dataframe$Expert <- grepl("Expert", dataframe$Labor.Category)
dataframe$Executive <- grepl("Executive", dataframe$Labor.Category)
dataframe$Principal <- grepl("Principal", dataframe$Labor.Category)
dataframe$Lead <- grepl("Lead", dataframe$Labor.Category)
dataframe$Senior <- grepl("Senior", dataframe$Labor.Category)
dataframe$Junior <- grepl("Junior", dataframe$Labor.Category)
dataframe$Journeyman <- grepl("Journeyman", dataframe$Labor.Category)
dataframe$Partner <- grepl("Partner", dataframe$Labor.Category)
dataframe$Strategy <- grepl("Strategy", dataframe$Labor.Category)
dataframe$Actuary <- grepl("Actuary", dataframe$Labor.Category)
dataframe$President <- grepl("President", dataframe$Labor.Category)
dataframe$Clevel <- grepl("C-level", dataframe$Labor.Category)
dataframe$Coach <- grepl("Coach", dataframe$Labor.Category)
dataframe$ERP <- grepl("ERP", dataframe$Labor.Category)
dataframe$Strategic <- grepl("Strategic", dataframe$Labor.Category)
dataframe$Producer <- grepl("Producer", dataframe$Labor.Category)
dataframe$Sr <- grepl("Sr.", dataframe$Labor.Category)
dataframe$Jr <- grepl("Jr.", dataframe$Labor.Category)
dataframe$Consultant <- grepl("Consultant", dataframe$Labor.Category)
dataframe$Interpreter <- grepl("Interpreter", dataframe$Labor.Category)

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

####################
#  Decision Tree   #
####################

# Train DT 
DT <- rpart(Current.Year.Labor.Price ~ ., data = train)

# Run model against test set
DTPred <- predict(DT, test)

# score model using RMSE
rmse(test$Current.Year.Labor.Price, DTPred)

# plot the decision tree
rpart.plot(DT, type = 3)

# Plot results versus predictions
plotframe <- data.frame( predicted = DTPred, actual = test$Current.Year.Labor.Price, ed = test$education.Level)
ggplot(data = plotframe, aes( x = actual, y = predicted)) + geom_point() + labs(x = "Actual Bill Rate (current year)", y = "Predicted Bill Rate", title = "Bi-variate analysis, Decision Tree Model") + geom_abline(slope = 1, intercept = 0)

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
