# AUTOMATING REVIEWS IN MEDICINE

rm(list=ls())
trials = read.csv("clinical_trial.csv", stringsAsFactors = FALSE)
str(trials)
summary(trials)
max(nchar(trials$abstract))
table(nchar(trials$abstract) == 0)

# different way to count the number of missing abstracts
sum(nchar(trials$abstract) == 0)

# Find the observation with the minimum number of characters in the title.
which.min(nchar(trials$title)) # 1258
trials$title[which.min(nchar(trials$title))]

# Preparing the corpus
library(tm)
library(SnowballC)
corpusTitle = Corpus(VectorSource(trials$title))
corpusTitle
corpusTitle[[1]]$content
corpusAbstract = Corpus(VectorSource(trials$abstract))
corpusAbstract
corpusAbstract[[2]]$content

# Convert corpus to lowercase
corpusTitle = tm_map(corpusTitle, tolower)
corpusTitle[[1]]
corpusAbstract = tm_map(corpusAbstract, tolower)
corpusAbstract[[2]]

corpusTitle = tm_map(corpusTitle, PlainTextDocument)
corpusAbstract = tm_map(corpusAbstract, PlainTextDocument)

# remove punctuation
corpusTitle = tm_map(corpusTitle, removePunctuation)
corpusTitle[[1]]$content
corpusAbstract = tm_map(corpusAbstract, removePunctuation)
corpusAbstract[[2]]$content

# Remove the English language stop words 
stopwords("english")
length(stopwords("english"))   # should be 174

corpusTitle = tm_map(corpusTitle, removeWords, stopwords("english"))
corpusTitle[[1]]$content
corpusAbstract = tm_map(corpusAbstract, removeWords, stopwords("english"))

# Stem the words
corpusTitle = tm_map(corpusTitle, stemDocument)
corpusTitle[[1]]$content
corpusAbstract = tm_map(corpusAbstract, stemDocument)
corpusAbstract[[2]]$content

# Build a document term matrix 
dtmTitle = DocumentTermMatrix(corpusTitle)
dtmTitle
dtmAbstract = DocumentTermMatrix(corpusAbstract)
dtmAbstract

# limit sparsness of at most 95% (terms that appear in at least 5% of documents).
dtmTitle = removeSparseTerms(dtmTitle, 0.95)
dtmTitle
dtmAbstract = removeSparseTerms(dtmAbstract, 0.95)
dtmAbstract

# convert to data frames
# rows are documents, columns are words
dtmTitle = as.data.frame(as.matrix(dtmTitle))
dtmAbstract = as.data.frame(as.matrix(dtmAbstract))

# How many terms remain in dtmTitle after removing sparse terms (aka how many columns does it have)?
str(dtmTitle)
# or use:
dim(dtmTitle)
ncol(dtmTitle)

# How many terms remain in dtmAbstract?
str(dtmAbstract)
dim(dtmAbstract)
ncol(dtmAbstract)

# What is the most frequent word stem across all the abstracts? 
which.max(colSums(dtmAbstract))

?paste0
colnames(dtmTitle) = paste0("T", colnames(dtmTitle))
colnames(dtmAbstract) = paste0("A", colnames(dtmAbstract))

dtm = cbind(dtmTitle, dtmAbstract)

# Add the dependent variable "trial" to dtm, copying it from the original data frame. 
# Each search result is labeled with whether the paper is a clinical trial testing a drug therapy for cancer.

# How many columns are in this combined data frame?
dtm$trial = trials$trial
ncol(dtm)

# Building a model
library(caTools)
set.seed(144)
split = sample.split(dtm$trial, SplitRatio = 0.7)
train = subset(dtm, split == TRUE)
test = subset(dtm, split == FALSE)

# baseline of predicting the most frequent outcome in the training set for all observations.
# trial = 0 (no clinical trial testing a drug therapy for cancer)
summary(dtm)
table(dtm$trial)

# accuracy - 0.5607527
#  0     1 
# 1043  817 
1043/nrow(dtm)   

# Build a CART model
library(rpart)
library(rpart.plot)
trialCART = rpart(trial ~ ., data=train, method="class")
prp(trialCART)

# Obtain the training set predictions for the model.
# Extract the predicted probability of a result being a trial (trail = 1) - 0.87189

predictTrain = predict(trialCART, newdata = train )[,2]
summary(predictTrain)
str(predictTrain)

predictTest = predict(trialCART, newdata = test)
summary(predictTest)

# Use a threshold probability of 0.5 to predict that an observation is a clinical trial.
# What is the training set accuracy of the CART model?
table(train$trial, predictTrain > 0.5)
#    FALSE TRUE
# 0   631   99
# 1   131  441
(631+441)/nrow(train) # 0.8233487

# Confusion matrix
#                   | predicted = 0   |  predicted = 1 
#                   _______________________________________________
# Actual = 0  |   True negatives (TN)  |  False positives (FP)
# Actual = 1  |  False negatives (FN)  |   True positives (TP)
# 
# Sensitivity = TP/(TP + FN) - 0.770979
441/(441+131)
# Specificity = TN/(TN + FP) - 0.8643836
631/(631+99)

# Evaluate the CART model on the testing set 
predTest = predict(trialCART, newdata = test )[,2]
summary(predTest)
str(predTest)

# What is the testing set accuracy, assuming a probability threshold of 0.5.
table(test$trial, predTest > 0.5)
#   FALSE TRUE
# 0   261   52
# 1    83  162
(261+162)/nrow(test) # 0.7580645

# Using the ROCR package, what is the testing set AUC of the prediction model?
# (area under ROC curve).
library(ROCR)
ROCRpredTest = prediction(predTest, test$trial)
ROCRperfTest = performance(ROCRpredTest, "tpr", "fpr")
plot(ROCRperfTest, main="Test Data ROC", colorize=TRUE, print.cutoffs.at = seq(0, 1 , 0.1), text.adj = c(-0.4, 1) )
auc = as.numeric(performance(ROCRpredTest,"auc")@y.values)
auc

# False negatives (FN) - predict 0 (set B no trial) actual 1 (set A trial) -> results in missing papers in set A.'
# FP - predict 1 (set A trial), actual 0 (set B no trial), include papers in Set A that have no trials

# FN is more costly, reduce FN rate -> lower threshold - higher sensitivity & lower specificity
# we prefer a lower threshold in cases where false negatives are more costly than false positives, 
# since we will make fewer negative predictions.