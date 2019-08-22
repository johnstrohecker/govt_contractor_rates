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

# create dummy variables for key works in LCAT titles
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


# Mirza CNN implementaion
# devtools::install_github("rstudio/tensorflow")
library(tensorflow)
install_tensorflow()

library(keras)
install_tensorflow()

?dataset_cifar10 #to see the help file for details of dataset
cifar<-dataset_cifar10()

model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 16,
                kernel_size = c(3,3),
                activation = 'relu',
                input_shape = input_shape) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_flatten() %>% 
  layer_dense(units = 10,
              activation = 'relu') %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = num_classes,
              activation = 'softmax')
#  2nd try ----------------------------------------------------------------------------------------

#a linear stack of layers
model<-keras_model_sequential()
#configuring the Model
model %>%  
  #defining a 2-D convolution layer
  
  layer_conv_2d(filter=32,kernel_size=c(3,3),padding="same",                input_shape=c(32,32,3) ) %>%  
  layer_activation("relu") %>%  
  #another 2-D convolution layer
  
  layer_conv_2d(filter=32 ,kernel_size=c(3,3))  %>%  layer_activation("relu") %>%
  #Defining a Pooling layer which reduces the dimentions of the #features map and reduces the computational complexity of the model
  layer_max_pooling_2d(pool_size=c(2,2)) %>%  
  #dropout layer to avoid overfitting
  layer_dropout(0.25) %>%
  layer_conv_2d(filter=32 , kernel_size=c(3,3),padding="same") %>% layer_activation("relu") %>%  layer_conv_2d(filter=32,kernel_size=c(3,3) ) %>%  layer_activation("relu") %>%  
  layer_max_pooling_2d(pool_size=c(2,2)) %>%  
  layer_dropout(0.25) %>%
  #flatten the input  
  layer_flatten() %>%  
  layer_dense(512) %>%  
  layer_activation("relu") %>%  
  layer_dropout(0.5) %>%  
  #output layer-10 classes-10 units  
  layer_dense(10) %>%  
  #applying softmax nonlinear activation function to the output layer #to calculate cross-entropy  
  layer_activation("softmax") 
#for computing Probabilities of classes-"logit(log probabilities)



#Model's Optimizer
#defining the type of optimizer-ADAM-Adaptive Momentum Estimation
opt<-optimizer_adam( lr= 0.0001 , decay = 1e-6 )
#lr-learning rate , decay - learning rate decay over each update

model %>%
  compile(loss="categorical_crossentropy",
          optimizer=opt,metrics = "accuracy")
#Summary of the Model and its Architecture
summary(model)


# slipt independecnt variable
train_x <- subset(train, select= -c(Current.Year.Labor.Price))
train_y <- train$Current.Year.Labor.Price

test_x <- subset(test, select= -c(Current.Year.Labor.Price))
test_y <- test$Current.Year.Labor.Price

model %>% fit( train_x,train_y ,batch_size=32,
               epochs=2,validation_data = list(test_x, test_y),
               shuffle=TRUE)



#TRAINING PROCESS OF THE MODEL
data_augmentation <- TRUE  
if(!data_augmentation) {  
  model %>% fit( train_x,train_y ,batch_size=32,
                 epochs=2,validation_data = list(test_x, test_y),
                 shuffle=TRUE)
}
else {  
  #Generating images
  
  gen_images <- image_data_generator(featurewise_center = TRUE,
                                     featurewise_std_normalization = TRUE,
                                     rotation_range = 20,
                                     width_shift_range = 0.30,
                                     height_shift_range = 0.30,
                                     horizontal_flip = TRUE  )
  #Fit image data generator internal statistics to some sample data
  gen_images %>% fit_image_data_generator(train_x)
  #Generates batches of augmented/normalized data from image data and #labels to visually see the generated images by the Model
  model %>% fit_generator(
    flow_images_from_data(train_x, train_y,gen_images,
                          batch_size=32,save_to_dir="F:/PROJECTS/CNNcifarimages/"),
    steps_per_epoch=as.integer(50000/32),epochs = 80,
    validation_data = list(test_x, test_y) )
}
#use save_to_dir argument to specify the directory to save the #images generated by the Model and to visually check the Model's #output and ability to classify images.




# Marghub Mirza Charts

# Libraries
library(ggplot2)
library(dplyr)

# Most basic bubble plot
ggplot(data = dataframe, aes(x=Minimum.Years.Experience, y= education.Level, size = Current.Year.Labor.Price)) +
  geom_point(alpha=0.7)


# Most basic bubble plot
data <- dataframe
data %>%
  arrange(desc(Business.Size)) %>%
 # mutate(Site = factor(Site, Site)) %>%
  ggplot(aes(x=Minimum.Years.Experience, y=education.Level, size=Current.Year.Labor.Price, color=Business.Size)) +
  geom_point(alpha=0.5) +
  scale_size(range = c(.1, 24))



# Quick display of two cabapilities of GGally, to assess the distribution and correlation of variables 
#library(devtools)
# install_github("ggobi/ggally")
library(GGally)
ggpairs(data, columns = 7:8, ggplot2::aes(colour=education.Level)) 
ggpairs(data, columns = 6:8, ggplot2::aes(colour=education.Level)) 
ggpairs(data, columns = 8:10, ggplot2::aes(colour=education.Level)) 

data %>%
  ggplot( aes(x= education.Level, y= Current.Year.Labor.Price, fill= Minimum.Years.Experience)) +
  geom_violin() +
  xlab("education.Level") +
  theme(legend.position="none") +
  xlab("")


# Violin Chart Color is Educaion level 
data %>%
  ggplot( aes(x= Minimum.Years.Experience, y= Current.Year.Labor.Price, fill= education.Level)) +
  geom_violin() +
  theme(
    legend.position="none",
    plot.title = element_text(size=11)
  ) +
  ggtitle("Violin Chart Color is Educaion level ") +
  xlab("Years of Expereince")
