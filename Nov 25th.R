# column deduction
#packages
library(datasets)
library(data.table)
library(stringr)
library(glmnet) # for LASSO
library(usdm) # for VIF and multicollinearity
library(caret) # for feature selection purpose
library(ROCR)  # for ROC curve
loan0 <- read.csv("lending-club-loan-data/loan.csv")

# use data as is, then undersample majority/oversample minority and try again.

# row deduction
drops <- c("id","member_id",
           "emp_title",
           "earliest_cr_line",
           "tot_coll_amt",
           "tot_cur_bal",
           "title","grade","issue_d","url","pymnt_plan",
           "zip_code","earlist_cr_line","mths_since_last_delinq",
           "mths_since_last_record","last_pymnt_d","last_pymnt_amnt",
           "next_pymnt_d","last_credit_pull_d","mths_since_last_major_derog",
           "policy_code","annual_inc_joint",
           "dti_joint","verification_status_joint", "open_acc_6m",
           "open_il_6m","open_il_12m","open_il_24m","mths_since_rcnt_il",
           "total_bal_il","il_util","open_rv_12m","open_rv_24m","max_bal_bc",
           "all_util","total_rev_hi_lim","inq_fi","total_cu_tl","inq_last_12m")
loan1 <- loan0[,!(names(loan0) %in% drops)]

# remove instances with joint applicant or dti has a value 9999
# (meaning NA in survey), then remove the applicaiton_type field
loan2 <- subset(loan1,application_type =="INDIVIDUAL")
loan2 <- loan2[,!(names(loan2) == "application_type")]

# move the y column to the last column of data frame
loan3 <- loan2
loan3 <- cbind(loan3[,!(names(loan3) %in% "loan_status")],loan3$loan_status)
colnames(loan3)[35] <- "loan_status"


#Simplify Y lables
loan4<-loan3
YN <- c("Issued")
YT <- c("Current","Fully Paid","In Grace Period","Does not meet the credit policy. Status:Fully Paid")
YF <- c("Charged Off","Default", "Late (16-30 days)","Late (31-120 days)", "Does not meet the credit policy. Status:Charged Off")
loan4 <- loan4[!(loan4$loan_status %in% YN),] # delete all status being "Issued"
loan4$loan_status <- factor(loan4$loan_status %in% YT) # Assigning the status in YT with True, the rest with FALSE

#convert desc to length of desc
loan4$desc <- str_length(loan4$desc)
loan4$desc[is.na(loan4$desc)] <-0

# remove instances with NA 
indx <- apply(loan4, 2, function(x) any(is.na(x))) # check if loan4 still contains NA or inf
colnames(indx) # show the column names of these missing data if there is any, 
loan5 <- na.omit(loan4)

#re-order the data by numeric first, catagorical second, y to the last.
loan6 <- loan5
facvec <- which(sapply(loan6, class) == 'factor')
numvex <- which(sapply(loan6,class) == 'numeric')
loan6 <- cbind(loan6[numvex],loan6[facvec])

# Find the correlation matrix of all numeric columns in loan6
# Follow the method found at: http://machinelearningmastery.com/feature-selection-with-the-caret-r-package/
set.seed(127)
corr <- cor(loan6[,1:sum(table(numvex))]) # calc the correlation matrix between all the numeric values from data
#corr[lower.tri(corr,diag = TRUE)]<-NA # transform the correlation matrix to a upper triangular matrix
# print(corr) # summarize the correlation martrix
highlycorrelated <- findCorrelation(corr, cutoff = 0.95) # find attribute that are highly correlated (ideally >0.75)
#print(colnames(loan6[,highlycorrelated])) # print indexes of highly correlated attributes
##[1] "funded_amnt"     "funded_amnt_inv"      "total_pymnt"    "total_pymnt_inv"
#[5]  "out_prncp_inv"
### We found all the three columns with the suffix inv are almost pure colones of their non-suffix versions.
### According to the dictionary, they are just the non _inv surffix amount invested by the investors on the platform, which has no cross interest with our research goal. Therefore,we decide to delete them all with confidence. 
### funded_amnt and total_amnt are nearly exactly identical to loan_amnt because by dictionary they are still investor exclusively related, so we can delete them for its redundancy. 
as <- c("funded_amnt_inv", "total_pymnt_inv", "out_prncp_inv")
        
loan7 <- loan6[,!(names(loan6) %in% as)]
corr1 <- cor(loan7[,1:(sum(table(numvex))-sum(table(as)))]) # calc the correlation matrix between all the numeric values from data
highlycorrelated1 <- findCorrelation(corr1, cutoff = 0.9) # find attribute that are highly correlated (ideally >0.75)
#print(colnames(loan7[,highlycorrelated1])) # print indexes of highly correlated attributes
## Here we get a NULL, meanning no more redundant attributes after screening by the correlation matrix
## note: the correlation matrix should be corr1

# rescale all the numeric data into (0,1) interval
loan8 <- loan7
for (ic in 1:(sum(table(numvex))-sum(table(as)))) {
  loan8[,ic] <- loan8[,ic] - min(loan8[,ic])
  loan8[,ic] <- loan8[,ic]/max(loan8[,ic])
}
# expand the factor columns as design matrix without the intercept
loan9 <- cbind(loan8[,1:23],model.matrix(~ term + sub_grade + emp_length + home_ownership + verification_status + purpose + addr_state + initial_list_status + loan_status, loan8)[,-1])

# check if loan9 still contains NA or inf
indx <- apply(loan9, 2, function(x) any(is.na(x)| is.infinite(x)))
colnames(indx) # show the column names of these missing data if there is any, 

## generate 10 random partition from the original data to train and test randomly 5-fold CV
Nsimu = 3
Nobserv = nrow(loan9)
train.index <- matrix(0, Nsimu, Nobserv) # each row indicates one set of simulated training indices
set.seed(3)
for(i in 1:Nsimu) train.index[i,]=sample(x=c(rep(1,floor(Nobserv*0.8)),rep(0,ceiling(Nobserv*0.2))), size=Nobserv, replace=F)   # generate random indices of training data

Npara <- ncol(loan9)
result_beta = list()
result_y = list()
result_AUC = list()
result_C_Matrix = list()


for(isimu in 1:Nsimu) {             ### start of loop with "isimu"
  # partition the original data into training and testing datasets
  train <- subset( loan9, train.index[isimu,]==1 )[,1:Npara]
  test  <- subset( loan9, train.index[isimu,]==0 )[,1:Npara] 
  
  ### ridge reggression with 5-fold cross-validation under glmnet package
  
  # generate a customized lambda sequence to choose the penalty function coeff
  lambdasequence = exp(seq(from=0, to = -3e1, by = -0.3))
  
  cv.out=cv.glmnet(x=as.matrix(train[,1:Npara-1]), y=as.numeric(train[,Npara]),lambda = lambdasequence, nfolds=5, alpha=0, standardize=F, family ="binomial")
  #plot(cv.out)
  #abline(h=cv.out$cvup[which.min(cv.out$cvm)])
  ## the best lambda chosen by 5-fold cross-validation
  lambda.5fold=cv.out$lambda.1s
  # apply Lasso with chosen lambda
  fitlasso=glmnet(x=as.matrix(train[,1:Npara-1]), y=as.numeric(train[,Npara]),alpha=1,lambda=lambda.5fold,standardize=F,thresh=1e-12)
  # fitted coefficients, note that they are not the same as Table 3.3 on page 63 due to the chosen labmda and fitting algorithm
  # fitlasso$a0
  result_beta[[isimu]] <- fitlasso$beta
  
  ## estimating mean prediction error based on LASSO
  test.lasso=predict(fitlasso,newx=as.matrix(test[,1:Npara-1]))
  
  result_y[[isimu]] <- test.lasso
  
  wrongmatrix <- table(test$loan_statusTRUE, test.lasso>0.5)
  ## Use ROC to determine the best cutoff point to generate the confusion matrix
  # because 0.5 in this case may not be a good choice for the cut-off
  ROCRpred <- prediction(test.lasso, test$loan_statusTRUE) 
  ROCRperf <- performance(ROCRpred, 'tpr','fpr') # true and false positive ratio
   
  
  # choose the closet point on ROC to (0,1) to be the best cut-off point
  dist_to_01<- sqrt((ROCRperf@x.values[[1]])^2+(ROCRperf@y.values[[1]]-1)^2)
  best_cutoff <- ROCRperf@alpha.values[[1]][which.min(dist_to_01)]
  best_cutoff_x <-ROCRperf@x.values[[1]][which.min(dist_to_01)]
  best_cutoff_y <-ROCRperf@y.values[[1]][which.min(dist_to_01)]
  #Auc value
  AUC_value <- performance(ROCRpred, 'auc')
  result_AUC[[isimu]] <- AUC_value@y.values
  
  ## Generate the confusion matrix
  Conf_Matrix <- table(test$loan_statusTRUE, test.lasso>best_cutoff)
  result_C_Matrix[[isimu]] <- Conf_Matrix 
  }                                                               ### end of loop with "isimu"

plot(ROCRperf, colorize = TRUE, text.adj = c(-0.2,1.7)) ## plot requires too many computation
# highlight the best_cutoff point
points(best_cutoff_x,best_cutoff_y, col = "Purple", pch = 16)

wrongprecision <- 158038/(158038+5181)
wrongrecall <- 158038/(158038+11707)
wrongaccuracy <- (158038+646)/(646+11707+5181+158038)
wrongFscore <- 2*wrongprecision*wrongrecall/(wrongprecision+wrongrecall)
trueaccuracy <- (10066+147810)/(10066+147810+2390+15306)
trueprecision <- 147810/(147810+15306)
truerecall <- 147810/(147810+2390)
trueFscore <- 2*trueprecision*truerecall/(trueprecision+truerecall)


# plot the  25 most important features 
absresult_beta <- apply(result_beta[[1]], 1, abs)
barplot2(tail(sort(absresult_beta, decreasing = F), 25), horiz =T, space = 5, log = "x")


st############################ below here is useless
model <- glm(loan_status ~ ., family = binomial(link = "logit"), data = train)
pred <- predict(model, type ='response', test)
#str(predict)
#summary(model)
#str(pred)
#table(test$loan_status, pred>0.5)  # 0.5 这个阈值不一定合理还是用ROC来确定

ROCRpred <- prediction(test.lasso, test$loan_statusTRUE)   
#ROCpred 原理大致:就是从1到0不断iterate cut off value,然后据此算 confusion matrix
ROCRperf <- performance(ROCRpred, 'tpr','fpr') # true and false positive ratio
#ROCRperf原理大致:从1到0不断iterate cut off value,然后据此算 fpr(X), tpr(Y),
plot(ROCRperf, colorize = TRUE, text.adj = c(-0.2,1.7))
# 颜色表示的是alpha,即cut off阈值. 
# 需要自己控制fpr和tpr来确定阈值.
dist_to_01<- sqrt((ROCRperf@x.values[[1]])^2+(ROCRperf@y.values[[1]]-1)^2)
best_cutoff <- ROCRperf@alpha.values[[1]][which.min(dist_to_01)]
best_cutoff_x <-ROCRperf@x.values[[1]][which.min(dist_to_01)]
best_cutoff_y <-ROCRperf@y.values[[1]][which.min(dist_to_01)]
#Auc value
ROCRauc <- performance(ROCRpred, 'auc')
# AUC value: 0.87755
