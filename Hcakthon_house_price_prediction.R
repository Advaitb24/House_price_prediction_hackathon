## 1. Import required libraries

library(missForest)     # Imputing the missing values from dataset
library(doParallel)     # Parallel processing of missForest
library(caret)          # Using Caret package for Cross Validation
library(e1071)          # Explore the Skewness of various features of the dataset.
install.packages("elasticnet")
library(elasticnet)     # Dependency for ridge regression model
library(randomForest)   # Dependeny for random forest regression


## 2. Read train and test datasets

test<-readxl::read_excel(file.choose())
train<-readxl::read_excel(file.choose())
train <- train[-15897,]

train1 <- train
test1 <- test
## 3. Remove the taget variable from train data and combine train and test datasets

SalePrice = train$Price 
train <- train[,-21]

fulldata <- rbind(train,test)
str(fulldata)
fulldata$Date <- as.integer(fulldata$Date)
str(fulldata)
class(fulldata)
fulldata <- as.data.frame(fulldata)

fulldata$`waterfront present` <- as.integer(fulldata$`waterfront present`)
fulldata$`condition of the house` <- as.integer(fulldata$`condition of the house`)
fulldata$`grade of the house`<- as.integer(fulldata$`grade of the house`)
fulldata$`Postal Code` <- as.factor(fulldata$`Postal Code`)

fulldata$`Postal Code` <- NULL
## 4. Take a look at proportion of missing values

cat(sprintf("Percentage of missing values in the overall train dataset: %s%s\n", round(length(which(is.na(train) == TRUE)) * 100 / (nrow(train) * ncol(train)), 2), "%"))

cat(sprintf("Percentage of missing values in the overall test dataset: %s%s\n", round(length(which(is.na(test) == TRUE)) * 100 / (nrow(test) * ncol(test)), 2), "%"))


## 5. Impute missing values

#5.1 Set no. of cores to be used for parellel processing of missforest

registerDoParallel(cores = 3)
set.seed(999)


#5.2 Impute missing values using missforest package which runs a randomForest on each variable using the observed part and predicts the na values.
#Set (parallelize="variables") to compute several forests on multiple variables at the same time 

fulldata.mis <- missForest(xmis = fulldata, maxiter = 10, ntree = 30, variablewise = FALSE,
                           decreasing = FALSE, verbose = TRUE, mtry = floor(sqrt(ncol(fulldata))), replace = TRUE,
                           classwt = NULL, cutoff = NULL, strata = NULL, sampsize = NULL, nodesize = NULL, 
                           maxnodes = NULL, xtrue = NA, parallelize = "variables")

fullnona <- fulldata.mis$ximp
write.csv(fullnona,"fullnona.csv",row.names = F)

trainnum <- fullnona[1 : nrow(train),]
write.csv(trainnum, "trainnum.csv")
testnum <- fullnona[nrow(train) + 1 : nrow(test),]
write.csv(testnum, "testnum.csv")

summary(fullnona)

## 6. Verify the skewness dataset features

classes <- lapply(fullnona,function(x) class(x))
numeric_feats <- names(classes[classes=="integer" | classes=="numeric"])
factor_feats <- names(classes[classes=="factor"| classes=="character"])

skewed_feats <- sapply(numeric_feats, function(x) skewness(fullnona[[x]]))
skewed_feats <- skewed_feats[abs(skewed_feats) > .75]
skewed_feats


#Take log transformation of features for which skewness more than 0.75

for (x in names(skewed_feats)) {fullnona[[x]] <- log(fullnona[[x]]+1)}


## 7. Use dummyvars function in caret package for one-hot encoding of categorical variables

dummies <- dummyVars(~., data = fullnona)
fullnonanum <- data.frame(predict(dummies, newdata = fullnona))


## 8. Split full dataset into train and test sets

trainnumlog <- fullnonanum[1 : nrow(train),]

testnumlog <- fullnonanum[nrow(train) + 1 : nrow(test),]


## 9. Add SalePrice back to train dataset

trainnumlog <- cbind(trainnumlog, SalePrice)


## 10. Ridge Regression Model

### 10.1 Create model

tr.control <- trainControl(method="repeatedcv", number = 10,repeats = 10)

lambdas <- seq(1,0,-.001)

set.seed(123)
ridge_model <- train(SalePrice~., data=trainnumlog,method="glmnet",metric="RMSE",
                     maximize=FALSE,trControl=tr.control,
                     tuneGrid=expand.grid(alpha=0,lambda=lambdas))



### 10.2 Verify accuracy

ridge_model$results


### 10.3 Take a look at contribution of each variable to make prediction

varImp(ridge_model)


### 10.4 Write predictions to file

ridge_preds <- round(predict(ridge_model,newdata = testnum), 2)
write.csv(data.frame(id=test$id,Price=ridge_preds),"Bazinga_File_3.csv",row.names = F)


## 11. Laso Regression Model

### 11.1 Create model
set.seed(123)
lasso_model <- train(SalePrice~., data=trainnumlog,method="glmnet",metric="RMSE",
                     maximize=FALSE,trControl=tr.control,
                     tuneGrid=expand.grid(alpha=1,lambda=c(1,0.1,0.05,0.01,seq(0.009,0.001,-0.001), 0.00075,0.0005,0.0001)))


### 11.2 Verify accuracy
lasso_model$results

### 11.3 Take a look at contribution of each variable to make prediction
varImp(lasso_model)


### 11.4 Write predictions to file
lassopreds <- round(predict(lasso_model,newdata = testnum), 2)

write.csv(data.frame(id=test$id,Price=lassopreds),"Bazinga_File_4.csv",row.names = F)


## 12. Random Forest Regression Model

### 12.1 Create model
rf_model <- train(SalePrice~., data=trainnumlog,method="rf",metric="RMSE",
                  maximize=FALSE,trControl=trainControl(method="repeatedcv",number=5),
                  tuneGrid=expand.grid(mtry = c(5)), importance = T, allowParallel = T, prox = T)


### 12.2 Verify accuracy
rf_model$results


### 12.3 Take a look at contribution of each variable to make prediction

varImp(rf_model)

### 12.4 Write predictions to file

rfpreds <- round(predict(rf_model,newdata = testnum), 2)

write.csv(data.frame(Id=test$Id,SalePrice=rfpreds),"random_forest_preds.csv",row.names = F)


## 13. Linear Regression Model

### 13.1 Create model

lm_model <- train(SalePrice~., data=trainnum, method="lm",metric="RMSE",
                  maximize=FALSE,trControl=trainControl(method = "repeatedcv",number = 10)
)


### 13.2 Verify accuracy

lm_model$results


### 13.3 Take a look at contribution of each variable to make prediction

varImp(lm_model)


### 13.4 Write predictions to file

lmpreds <- round(predict(lm_model,newdata = testnum), 2)

write.csv(data.frame(Id=test$id,SalePrice=lmpreds),"linear_model_preds.csv",row.names = F)

# Gradient Boosting
boostcontrol <- trainControl(method="repeatedcv", number=5, repeats=2)
set.seed(123)
fit.gbm <- train(SalePrice~., data=trainnumlog, method="gbm", metric="RMSE", trControl=boostcontrol, verbose=FALSE)
fit.gbm$results
lmpreds <- predict(fit.gbm, newdata=testnumlog)
write.csv(data.frame(Id=test$id,SalePrice=lmpreds),"Bazinga_File_5.csv",row.names = F)

