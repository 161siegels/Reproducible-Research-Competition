install.packages("rstan")
install.packages("caret")
library(caret)
library(rstan)
setwd("~/GitHub/Reproducible-Research-Competition")
dataset = read.csv("cleaned_data.csv")

#For execution on a local, multicore CPU with excess RAM we recommend calling
options(mc.cores = parallel::detectCores())
#To avoid recompilation of unchanged Stan programs, we recommend calling
rstan_options(auto_write = TRUE)
Sys.setenv(LOCAL_CPPFLAGS = '-march=native')

#Split dataset by each players shots
testing <- split(dataset, dataset$player_name)

#Get unique player names in the dataset to start building dataframe
player_names <- as.data.frame(unique(dataset$player_name))

tnm <- matrix(NA, nrow = length(testing), ncol = 1)
rnm <- matrix(NA, nrow = length(testing), ncol = 1)

for (i in 1:length(testing)) {
  pl <- testing[[i]]
    for (j in i) {
  rnm[[j]] <- nrow(testing[[i]])
  tnm[[i]] <- sum(pl$SHOT_RESULT2)/nrow(pl)
  }
}

fdf <- cbind(player_names, rnm, tnm)

#Manually went through each players shots as it worked better than trying to write a loop that went through each player and extracted/stored all the necessary information correctly
damedollar <- dataset[dataset$player_name == "aaron brooks", ]

#If you predict either all make or miss, how accurate are you?
guess <- sum(damedollar$SHOT_RESULT2)/nrow(damedollar)

#Set Seed for Reproducible result
set.seed(1305)

#Split into Train and Test  
sample <- sample.int(n = nrow(damedollar), size = floor(.70*nrow(damedollar)), replace = F)
train <- damedollar[sample, ]
test  <- damedollar[-sample, ]

train$GAME_CLOCK <- as.numeric(train$GAME_CLOCK)
train = dplyr::select(train,SHOT_RESULT2,LOCATION,GAME_CLOCK2,SHOT_CLOCK,DRIBBLES,TOUCH_TIME,SHOT_DIST,PTS_TYPE,CLOSE_DEF_DIST,FG_Percent,FG3_Percent,SHOT_NUMBER,FT_Percent,ratio,End_SC,End_PC,Quick_S,Clutch_T,Garbage_T,PERIOD,point_diff,jump_shot,putback,layup,bank,dunk,driving,tip,pullup,fadeaway,running,hook,reverse,turnaround,fingerroll)

#Grab the predictors so that STAN model is efficient 
X <- model.matrix(train$SHOT_RESULT2 ~ ., data = train[, c(1:7, 9, 12, 14)])

m2 <- '
data {                          
int<lower=0> N;                # number of observations
int<lower=0> K;                # number of variables
int<lower=0,upper=1> SHOT_RESULT2[N];  # setting the dependent variable (Shot_Result) as binary
matrix[N,K]  X;
}

parameters {
real alpha;                    # intercept
vector[K] beta;                       # coefficients
}

model {
SHOT_RESULT2 ~ bernoulli_logit(alpha + X * beta); # model
}
'
#Convert Data to list form for required input in STAN
d2.list <- list(N = nrow(train), SHOT_RESULT2 = train$SHOT_RESULT2, X=X[,-1], K=ncol(X[,-1]))

sim <- stan(model_code = m2, data = d2.list, 
            warmup = 500, thin = 10, iter = 5000, chains = 1,
            #Parameters that assist in computational efficiency/convergence
            control = list(adapt_delta = 0.99, max_treedepth = 20))

#Extract coefficient values
GQ.alpha=extract(sim,pars=c("alpha","beta"))[[1]]
GQ.beta=extract(sim,pars=c("alpha","beta"))[[2]]

GQ=cbind(GQ.alpha,GQ.beta)

#First column is intercept, rest is beta for predictions
cm = colMeans(GQ)
cm

test$GAME_CLOCK <-as.numeric(test$GAME_CLOCK)
test = dplyr::select(test,SHOT_RESULT2,LOCATION,GAME_CLOCK2,SHOT_CLOCK,DRIBBLES,TOUCH_TIME,SHOT_DIST,PTS_TYPE,CLOSE_DEF_DIST,FG_Percent,FG3_Percent,SHOT_NUMBER,FT_Percent,ratio,End_SC,End_PC,Quick_S,Clutch_T,Garbage_T,PERIOD,point_diff,jump_shot,putback,layup,bank,dunk,driving,tip,pullup,fadeaway,running,hook,reverse,turnaround,fingerroll)
tm <- model.matrix(test$SHOT_RESULT2 ~ ., data = test[,c(1:7, 9, 12, 14)])

#Initialize empty matrices for the loops
y <- matrix(NA, nrow = nrow(tm), ncol = 1)
z <- matrix(NA, nrow = nrow(y), ncol = 1)

#FOR LOOP TO GET PREDICITONS LINEAR ALGEBRA 
for (i in 1:nrow(tm)) {
  y[[i]] = cm %*% tm[i, ]
  z[[i]] = exp(y[i])/(1+exp(y[i]))
  z[[i]] = round(z[[i]], 0)
}

#Check the accuracy of the predictions
verify = cbind(z, test$SHOT_RESULT2)
verify <- as.data.frame(verify)
colnames(verify) <- c("Pred", "Actual")
verify$test <- ifelse(verify$Pred == verify$Actual, 1, 0)
predacc <-(sum(verify$test)/nrow(verify))

#Prediction Accuracy 
predacc

verify$Pred <- as.factor(verify$Pred)
verify$test <- as.factor(verify$test)

sen <- caret::sensitivity(verify$Pred, verify$test)
#Sensitivity
sen

spe <- caret::specificity(verify$Pred, verify$test)
#Specificity
spe

#All results were transcribed and put in an Excel sheet (rstanresults.csv) for easy reading



