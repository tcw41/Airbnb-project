#Week 3
#Neural Networks

#(Install and) load the following libraries
library(caret) #For confusionMatrix(), training ML models, and more
library(neuralnet) #For neuralnet() function
library(dplyr) #For some data manipulation and ggplot
library(pROC)  #For ROC curve and estimating the area under the ROC curve
library(fastDummies) #To create dummy variables
library(sigmoid)
### Data processing steps

#Download the following dataset and load/save it as 'titanic'
airbnb_df= read.csv("AirbnbListings.csv", sep = ",", stringsAsFactors = TRUE)
airbnb_df= AirbnbListings
str(airbnb_df)
summary(airbnb_df)
View(airbnb_df)

##Want to calculate the number of days between Host since and today's date (which is 09/30/2022 in this assignment)
##First, need to create a column for today's date
airbnb_df$today_date= "09/30/2022"

View(airbnb_df)## checking to see if column was created (can confirm that a column was created correctly)



### Now converting Host since and today_date into dates
airbnb_df$host_since<-as.Date(airbnb_df$host_since,format="%m/%d/%Y")
airbnb_df$today_date<-as.Date(airbnb_df$today_date,format="%m/%d/%Y")

###Now will calculate difference between host since and today_date in days
airbnb_df$Date_difference_in_days <-difftime(airbnb_df$today_date,airbnb_df$host_since,units=c("days"))
### Converting back the dates to numeric 

airbnb_df$host_since= as.numeric(airbnb_df$host_since)
airbnb_df$today_date= as.numeric(airbnb_df$today_date)
airbnb_df$Date_difference_in_days= as.numeric(airbnb_df$Date_difference_in_days)
airbnb_df$years_host = as.numeric(airbnb_df$Date_difference_in_days) %/% 365.25
View(airbnb_df)
#Create dummy variables for other categorical variables
airbnb_dummies = dummy_cols(airbnb_df, select_columns = c('neighborhood','superhost','room_type'))

##Viewing dataset after dummy function
View(airbnb_dummies)
summary(airbnb_dummies)
#Let's remove variables that we will not use for distance calculation
#due to their data type being categorical (such as sex) or not being relevant (such as Id)
final_data = 
  airbnb_dummies %>% select(-c(listing_id, host_since, superhost, superhost_FALSE, neighborhood, neighborhood_Takoma,room_type_NA,room_type, today_date, Date_difference_in_days))

View(final_data)

#1. Data splitting into training and test (70%:30%)
#You can take a random sample (70%) of numbers between 1 and number of rows (891)
set.seed(123)# Set a seed for reproducibility
index = sample(nrow(final_data),0.7*nrow(final_data))
#You can also use createDataPartition() function from caret. 
#index = createDataPartition(titanic$Survived,p=0.7,list=FALSE)

train_data = final_data[index, ]
test_data = final_data[-index, ]

#Cleaning up column name for room type 
colnames(train_data)[18] = "room_type_Entire_home_apt"
colnames(test_data)[18] = "room_type_Entire_home_apt"
colnames(train_data)[19] = "room_type_Private_room"
colnames(test_data)[19] = "room_type_Private_room"
colnames(train_data)[20] = "room_type_Shared_room"
colnames(test_data)[20] = "room_type_Shared_room"


View(train_data)
View(test_data)
#2. Clean and Preprocess Data 

#Check for columns/variables with missing values
sapply(train_data, function(x){sum(is.na(x))})
sapply(test_data, function(x){sum(is.na(x))})

#Replacing Missing Data

train_data$host_acceptance_rate[is.na(train_data$host_acceptance_rate)] = median(train_data$host_acceptance_rate, na.rm = TRUE)
test_data$host_acceptance_rate[is.na(test_data$host_acceptance_rate)] = median(train_data$host_acceptance_rate, na.rm = TRUE)

train_data$total_reviews[is.na(train_data$total_reviews)] = median(train_data$total_reviews, na.rm = TRUE)
test_data$total_reviews[is.na(test_data$total_reviews)] = median(train_data$total_reviews, na.rm = TRUE)

train_data$avg_rating[is.na(train_data$avg_rating)] = median(train_data$avg_rating, na.rm = TRUE)
test_data$avg_rating[is.na(test_data$avg_rating)] = median(train_data$avg_rating, na.rm = TRUE)

train_data$room_type_Entire_home_apt[is.na(train_data$room_type_Entire_home_apt)] = median(train_data$room_type_Entire_home_apt, na.rm = TRUE)
test_data$room_type_Entire_home_apt[is.na(test_data$room_type_Entire_home_apt)] = median(train_data$room_type_Entire_home_apt, na.rm = TRUE) 

train_data$room_type_Private_room[is.na(train_data$room_type_Private_room)] = median(train_data$room_type_Private_room, na.rm = TRUE)
test_data$room_type_Private_room[is.na(test_data$room_type_Private_room)] = median(train_data$room_type_Private_room, na.rm = TRUE) 

train_data$room_type_Shared_room[is.na(train_data$room_type_Shared_room)] = median(train_data$room_type_Shared_room, na.rm = TRUE)
test_data$room_type_Shared_room[is.na(test_data$room_type_Shared_room)] = median(train_data$room_type_Shared_room, na.rm = TRUE) 

#Check to make sure there's no missing age
sapply(train_data, function(x){sum(is.na(x))})
sapply(test_data, function(x){sum(is.na(x))})

View(train_data)

#The most common scaling for neuralnets are min-max normalization (values will be between 0 and 1).
#Since the neuralnet() function doesn't have an option to automatically do that, 
# we will scale the predictors ourselves. 

#We are using preProcess function from "caret" package, using "range" (min-max normalization) method
#Again, we are using train information to scale test data!
#NOTE: Predictors that are not numeric are ignored in the calculations of preProcess function

scale_vals = preProcess(train_data, method="range")
train_data_s = predict(scale_vals, train_data)
test_data_s = predict(scale_vals, test_data)

View(test_data_s)
#Model 1:
NN1 = neuralnet(price~.,
                data=train_data_s,
                linear.output = FALSE,
                stepmax = 1e+06,
                act.fct = relu,
                hidden=2)

#The output model:
NN1

plot(NN1)
#predicted values for test data (these will be between 0 and 1)
pred1 = predict(NN1, test_data_s)

#Scaling back predicted values to the actual scale of price
pred1_acts = pred1*(max(train_data$price)-min(train_data$price))+min(train_data$price)

plot(test_data$price,pred1_acts, xlab="Price",ylab="Predicted Price",main="Model 1")


#Model 2
ctrl = trainControl(method="cv",number=10)
myGrid = expand.grid(size = seq(1,10,1),
                     decay = seq(0.01,0.2,0.04))

set.seed(123)
NN2 = train(
  price ~ ., data = train_data_s,
  linout = TRUE,
  method = "nnet", 
  tuneGrid = myGrid,
  trControl = ctrl,
  trace=FALSE)

#The output model:

NN2
plot(NN2)
#predicted values for test data (these will be between 0 and 1)
pred2 = predict(NN2, test_data_s)

#Scaling back predicted values to the actual scale of price
pred2_acts = pred2*(max(train_data$price)-min(train_data$price))+min(train_data$price)

plot(test_data$price,pred2_acts,xlab="Price",ylab="Predicted Price",main="Model 2")


###Model 3
NN3 = neuralnet(price ~.,
                data=train_data_s,
                hidden = 5,
                linear.output = T,
                lifesign = 'full',
                threshold = 0.1,
                rep = 1,
                act.fct = tanh,
                algorithm = "sag",
                stepmax = 200000)
###The output model
NN3
plot(NN3)

#predicted values for test data (these will be between 0 and 1)
pred3 = predict(NN3, test_data_s)

#Scaling back predicted values to the actual scale of price
pred3_acts = pred3*(max(train_data$price)-min(train_data$price))+min(train_data$price)

plot(test_data$price,pred3_acts,xlab="Price",ylab="Predicted Price",main="Model 3")

#Models comparison
postResample(pred1_acts,test_data$price) #Model 1
postResample(pred2_acts,test_data$price) #Model 2
postResample(pred3_acts,test_data$price) #Model 3


