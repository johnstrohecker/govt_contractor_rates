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
dataframe <- dataframe %>%
  mutate(
    Technician = grepl("technician", Labor.Category), 
    Administrative = grepl("administrative", Labor.Category), 
    Clerk = grepl("clerk", Labor.Category), 
    Engineer = grepl("engineer", Labor.Category), 
    Architecut = grepl("architect", Labor.Category), 
    Analyst = grepl("analyst", Labor.Category), 
    Program = grepl("program", Labor.Category), 
    Project = grepl("project", Labor.Category), 
    Manager = grepl("manager", Labor.Category), 
    Director = grepl("director", Labor.Category), 
    Expert = grepl("expert", Labor.Category), 
    Excutive = grepl("executive", Labor.Category), 
    Principal = grepl("principal", Labor.Category), 
    Lead = grepl("lead", Labor.Category), 
    Senior = grepl("senior", Labor.Category), 
    Junior = grepl("junior", Labor.Category), 
    Journeyman = grepl("journeyman", Labor.Category), 
    Partner = grepl("partner", Labor.Category), 
    Strategy = grepl("strategy", Labor.Category), 
    Actuary = grepl("actuary", Labor.Category), 
    President = grepl("president", Labor.Category), 
    CLevel = grepl("c-level", Labor.Category), 
    Coach = grepl("coach", Labor.Category), 
    ERP = grepl("erp", Labor.Category), 
    Strategic = grepl("strategic", Labor.Category), 
    Producer = grepl("producer", Labor.Category), 
    SR = grepl("sr", Labor.Category), 
    JR = grepl("jr", Labor.Category), 
    Consultant = grepl("consultant", Labor.Category), 
    Interpreter = grepl("interpreter", Labor.Category), 
  )

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
                                                                                
########################
#  Neural Network  #
########################

#adapted from https://www.analyticsvidhya.com/blog/2017/09/creating-visualizing-neural-network-in-r/
df <- dataframe

colnames(df)

#pull in categorical variables
install.packages("dummies")
library(dummies)

df.new <- dummy.data.frame(df, names = c("Business.Size", "Schedule","education.Level"), sep = "_")
names(df.new)

keeps <- c("Schedule_36_Office_Imaging_Document", "Schedule_621i_Healthcare", "Schedule_71_Furniture", "Schedule_71_IIK",
           "Schedule_736TAPS", "Schedule_78_SPORTS", "Schedule_AIMS", "Schedule_Consolidated", "Schedule_Environmental", 
           "Schedule_FABS", "Schedule_IT Schedule 70", "Schedule_Language Services", "Schedule_Logistics", "Schedule_PES",
           "Schedule_MOBIS", "education.Level_Associates", "education.Level_Bachelors", "education.Level_Masters", 
           "education.Level_High School", "education.Level_Ph.D.", "Business.Size_other than small business", "Business.Size_small business", "Schedule_03FAC","Minimum.Years.Experience","Current.Year.Labor.Price")

df = df.new[keeps]

colnames(df)
head(df)

# fix column names

names(df)[names(df) == "Schedule_IT Schedule 70"] <- "Schedule_IT_Schedule"
names(df)[names(df) == "Schedule_Language Services"] <- "Schedule_Language_Services"
names(df)[names(df) == "Business.Size_other than small business"] <- "Business.Size_other_than_small_business"
names(df)[names(df) == "Business.Size_small business"] <- "Business.Size_small_business"
names(df)[names(df) == "education.Level_High School"] <- "education.Level_High_School"
colnames(df)

df <- na.exclude(df)

# Random sampling
samplesize = 0.60 * nrow(df)
set.seed(80)
index = sample( seq_len ( nrow ( df ) ), size = samplesize )

# Create training and test set
datatrain = df[ index, ]
datatest = df[ -index, ]

## Scale data for neural network

max = apply(df , 2 , max)
min = apply(df, 2 , min)
scaled = as.data.frame(scale(df, center = min, scale = max - min))

## Fit neural network 

# install library
install.packages("neuralnet ")

# load library
library(neuralnet)

# creating training and test set
trainNN = scaled[index , ]
testNN = scaled[-index , ]

# fit neural network
#increased stepmax to make work https://stackoverflow.com/questions/19360835/neuralnet-overcoming-the-non-convergence-of-algorithm 
set.seed(2)
NN = neuralnet(Current.Year.Labor.Price ~ Schedule_36_Office_Imaging_Document + Schedule_621i_Healthcare + Schedule_71_Furniture + Schedule_71_IIK +   Schedule_736TAPS + Schedule_78_SPORTS + Schedule_AIMS + Schedule_Consolidated + Schedule_Environmental + Schedule_FABS + Schedule_IT_Schedule + Schedule_Language_Services + Schedule_Logistics + Schedule_PES + Schedule_MOBIS + education.Level_Associates + education.Level_Bachelors + education.Level_Masters + education.Level_High_School + education.Level_Ph.D. + Business.Size_other_than_small_business +  Business.Size_small_business + Schedule_03FAC + Minimum.Years.Experience, trainNN, hidden = 3 , linear.output = T, stepmax = 1e6)

 # plot neural network
plot(NN)

# Prediction using neural network
colnames(df)
predict_testNN = compute(NN, testNN[,c(1:25)])
predict_testNN = (predict_testNN$net.result * (max(df$Current.Year.Labor.Price) - min(df$Current.Year.Labor.Price))) + min(df$Current.Year.Labor.Price)

plot(datatest$Current.Year.Labor.Price, predict_testNN, col='blue', pch=16, ylab = "predicted price NN", xlab = "real price")

# Calculate Root Mean Square Error (RMSE) and MSE
RMSE.NN = (sum((datatest$Current.Year.Labor.Price - predict_testNN)^2) / nrow(datatest)) ^ 0.5
MSE.NN <- sum((datatest$Current.Year.Labor.Price - predict_testNN)^2)/nrow(datatest)

#plot NN with LM
plot(datatest$Current.Year.Labor.Price,predict_testNN,col='red',main='Real vs predicted NN',pch=18,cex=0.7)
points(test$Current.Year.Labor.Price, LogPred,col='blue',pch=18,cex=0.7)
abline(0,1,lwd=2)
legend('bottomright',legend=c('NN','LM'),pch=18,col=c('red','blue'))


#Cross validation of neural network model with MSE ***this needs work***
#https://www.r-bloggers.com/fitting-a-neural-network-in-r-neuralnet-package/ (also attempted cross validation example in first link)
set.seed(450)
cv.error <- NULL
k <- 10

library(plyr) 
pbar <- create_progress_bar('text')
pbar$init(k)

for(i in 1:k){
  index <- sample(1:nrow(df),round(0.9*nrow(df)))
  trainNN = scaled[index,]
  testNN = scaled[-index,]
  
  NN <- neuralnet(Current.Year.Labor.Price ~ Schedule_36_Office_Imaging_Document + Schedule_621i_Healthcare + Schedule_71_Furniture + Schedule_71_IIK +   Schedule_736TAPS + Schedule_78_SPORTS + Schedule_AIMS + Schedule_Consolidated + Schedule_Environmental + Schedule_FABS + Schedule_IT_Schedule + Schedule_Language_Services + Schedule_Logistics + Schedule_PES + Schedule_MOBIS + education.Level_Associates + education.Level_Bachelors + education.Level_Masters + education.Level_High_School + education.Level_Ph.D. + Business.Size_other_than_small_business +  Business.Size_small_business + Schedule_03FAC + Minimum.Years.Experience, trainNN, hidden = 3 , linear.output = T,  stepmax = 1e6)

  
  predict_testNN = compute(NN,testNN[,c(1:25)])
  predict_testNN = (predict_testNN$net.result*(max(df$Current.Year.Labor.Price)-min(df$Current.Year.Labor.Price)))+min(df$Current.Year.Labor.Price)
  
  testNN.r <- (testNN$net.result*(max(df$Current.Year.Labor.Price)-min(df$Current.Year.Labor.Price)))+min(df$Current.Year.Labor.Price)
  
  cv.error [i] <- sum((df$Current.Year.Labor.Price - predict_testNN)^2)/nrow(testNN)
  
  pbar$step()
}

mean(cv.error)

cv.error

boxplot(cv.error,xlab='MSE CV',col='cyan',
        border='blue',names='CV error (MSE)',
        main='CV error (MSE) for NN',horizontal=TRUE)
