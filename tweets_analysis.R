library(mlr)
library(OOBCurve)
library(tidyverse)
library(lubridate)
library(dplyr)
library(purrr)
library(ggplot2)
library(scales)
library(stringr)
library(tidytext)
library(glmnet)
library(tm)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(data.table)
library(ROCR)
library(leaps)
library(xgboost)

#################
#Analysis can begin here
#################
#read in data
trump.tweets<-read.csv("trump_tweets.csv", stringsAsFactors=FALSE)

### Definition and restriction of corpus of words
# Definition of corpus of words
corpus = Corpus(VectorSource(trump.tweets$text))

# Start processing the text in the corpus! Here is a 
# summary of how we shall process the text.  
# 1. Change all the text to lower case.  
# 2. Transform "https://link" to ensure https is a word
# 3. Remove punctuation
# 4. Remove stop words and particular words. 
# 5. "Stem" the documents. 
# 6. Remove infrequent words. 

# 1. Everything in lower case
corpus = tm_map(corpus, tolower)

# 2. Transform "https://link" into "https link" to make sure https is a word
f <- content_transformer(function(x, oldtext,newtext) gsub(oldtext, newtext, x))
corpus <- tm_map(corpus, f, "https://", "http ")

# 3. Remove punctuation
corpus <- tm_map(corpus, removePunctuation)

# 4. Remove stop words and other particular words manually
corpus = tm_map(corpus, removeWords, stopwords("english"))
corpus = tm_map(corpus, removeWords, c("realdonaldtrump", "donaldtrump"))

# 5. Stemming
corpus = tm_map(corpus, stemDocument)

# 6. Find high-frequency words and remove uncommon words
frequencies = DocumentTermMatrix(corpus)
findFreqTerms(frequencies, lowfreq=200)
sparse = removeSparseTerms(frequencies, 0.99)

# Fraction of dataset used in the analysis
pct.text <- sum(as.matrix(sparse)) / sum(as.matrix(frequencies))

strwrap(corpus[[4]])
strwrap(corpus[[111]])

### Add dependent variable into dataset

# Creation dataset and importation of dependent variable
documentterms = as.data.frame(as.matrix(sparse))
documentterms$TrumpWrote = trump.tweets$TrumpWrote

# Putting the dependent variable in first place
col_idx <- grep("TrumpWrote", names(documentterms))
documentterms <- documentterms[, c(col_idx, (1:ncol(documentterms))[-col_idx])]

# Changing two header names to avoid interference with built-in functions
# Problematic to have numbers as headers (2016)
# Problematic to have function names as headers (next)
colnames(documentterms)[colnames(documentterms)=='2016']='y2016'
colnames(documentterms)[colnames(documentterms)=='next']='NEXT'

### Splitting into training and test set

split1 = (trump.tweets$created_at < "2016-06-01")
split2 = (tweets.trump$created_at >= "2016-06-01")
train = documentterms[split1,]
test = documentterms[split2,]

##### Baseline: Predicting all tweets coming from Trump himself

accuracy.baseline = sum(test$TrumpWrote)/nrow(test)
TPR.baseline = 1
FPR.baseline = 1
AUC.baseline = .5

##### BAG OF WORDS

### Logistic regression

logreg = glm(TrumpWrote ~., data=train, family="binomial")
summary(logreg)

predictions.logreg <- predict(logreg, newdata=test, type="response")
matrix.logreg = table(test$TrumpWrote, predictions.logreg > 0.5)
accuracy.logreg = (matrix.logreg[1,1]+matrix.logreg[2,2])/nrow(test)
TPR.logreg = (matrix.logreg[2,2])/sum(matrix.logreg[2,])
FPR.logreg = (matrix.logreg[1,2])/sum(matrix.logreg[1,])

ROC.logreg <- prediction(predictions.logreg, test$TrumpWrote)
ROC.logreg.df <- data.frame(fpr=slot(performance(ROC.logreg, "tpr", "fpr"),"x.values")[[1]],tpr=slot(performance(ROC.logreg, "tpr", "fpr"),"y.values")[[1]])
AUC.logreg <- performance(ROC.logreg, "auc")@y.values[[1]]

### CART

cart = rpart(TrumpWrote ~ ., data=train, method="class", cp = .003)


prp(cart, digits=3, split.font=1, varlen = 0, faclen = 0)

predictions.cart <- predict(cart, newdata=test, type="class")
matrix.cart = table(test$TrumpWrote, predictions.cart)
accuracy.cart = (matrix.cart[1,1]+matrix.cart[2,2])/nrow(test)
TPR.cart = (matrix.cart[2,2])/sum(matrix.cart[2,])
FPR.cart = (matrix.cart[1,2])/sum(matrix.cart[1,])

ROC.cart<- prediction(predict(cart, newdata=test)[,2], test$TrumpWrote)
ROC.cart.df <- data.frame(fpr=slot(performance(ROC.cart, "tpr", "fpr"),"x.values")[[1]],tpr=slot(performance(ROC.cart, "tpr", "fpr"),"y.values")[[1]])
AUC.cart <- as.numeric(performance(ROC.cart, "auc")@y.values)

### Random Forest

train$TrumpWrote = as.factor(train$TrumpWrote)
test$TrumpWrote = as.factor(test$TrumpWrote)

set.seed(123)
rfmodel = randomForest(TrumpWrote ~., data=train)

importance.rf <- data.frame(imp=round(importance(rfmodel)[order(-importance(rfmodel)),],2))

predictions.RF = predict(rfmodel, newdata=test)
matrix.RF = table(test$TrumpWrote, predictions.RF)
accuracy.RF = (matrix.RF[1,1]+matrix.RF[2,2])/nrow(test)
TPR.RF = (matrix.RF[2,2])/sum(matrix.RF[2,])
FPR.RF = (matrix.RF[1,2])/sum(matrix.RF[1,])

ROC.RF<- prediction(predict(rfmodel,newdata=test,type="prob")[,2], test$TrumpWrote)
ROC.RF.df <- data.frame(fpr=slot(performance(ROC.RF, "tpr", "fpr"),"x.values")[[1]],tpr=slot(performance(ROC.RF, "tpr", "fpr"),"y.values")[[1]])
AUC.RF <- as.numeric(performance(ROC.RF, "auc")@y.values)

### Summary

summary.performance <- data.frame (
  accuracy=round(c(accuracy.baseline,accuracy.logreg,accuracy.cart,accuracy.RF),3),
  TPR=round(c(TPR.baseline,TPR.logreg,TPR.cart,TPR.RF),3),
  FPR=round(c(FPR.baseline,FPR.logreg,FPR.cart,FPR.RF),3),
  AUC=round(c(AUC.baseline,AUC.logreg,AUC.cart,AUC.RF),3)
)

print(ggplot() +
        geom_line(data=ROC.logreg.df,aes(x=fpr,y=tpr,colour="a"),lwd=1,lty=1) +
        geom_line(data=ROC.cart.df,aes(x=fpr,y=tpr,colour="b"),lwd=1,lty=2) +
        geom_line(data=ROC.RF.df,aes(x=fpr,y=tpr,colour="c"),lwd=1,lty=3) +
        xlab("False Positive Rate") +
        ylab("True Positive Rate") +
        theme_bw() +
        xlim(0, 1) +
        ylim(0, 1) +
        scale_color_manual(name="Method", labels=c("a"="Logistic regression", "b"="CART", "c"="Random forest"), values=c("a"="gray", "b"="blue", "c"="red")) +
        theme(axis.title=element_text(size=18), axis.text=element_text(size=18), legend.text=element_text(size=18), legend.title=element_text(size=18)))


##### ADD METADATA

### Update to dataset

# Add for presence of picture  
tweets.trump = tweets.trump %>%
  mutate(pic_link = ifelse(str_detect(text, "t.co"),1,0))
# Add tweet length
tweets.trump = tweets.trump %>%
  mutate(length = nchar(text))
# Add hour of day
tweets.trump = tweets.trump %>%
  mutate(hour = hour(with_tz(created_at, "EST")))
# Add number of hashtags
tweets.trump = tweets.trump %>%
  mutate(hashtag = str_count(text, "#"))
# Add number of @mentions
tweets.trump = tweets.trump %>%
  mutate(mentions = str_count(text, "@"))

# Complete dataset
documentterms.metadata <- documentterms
documentterms.metadata$pic_link = tweets.trump$pic_link
documentterms.metadata$length = tweets.trump$length
documentterms.metadata$hour = as.factor(tweets.trump$hour)
documentterms.metadata$hashtag = tweets.trump$hashtag
documentterms.metadata$mentions = tweets.trump$mentions

### Split into training and test set

split1 = (tweets.trump$created_at < "2016-06-01")
split2 = (tweets.trump$created_at >= "2016-06-01")
train = documentterms.metadata[split1,]
test = documentterms.metadata[split2,]

### Logistic regression

logreg.metadata = glm(TrumpWrote ~., data=train, family="binomial")
summary(logreg.metadata)

predictions.metadata.logreg <- predict(logreg.metadata, newdata=test, type="response")
matrix.metadata.logreg = table(test$TrumpWrote, predictions.metadata.logreg > 0.5)
accuracy.metadata.logreg = (matrix.metadata.logreg[1,1]+matrix.metadata.logreg[2,2])/nrow(test)
TPR.metadata.logreg = (matrix.metadata.logreg[2,2])/sum(matrix.metadata.logreg[2,])
FPR.metadata.logreg = (matrix.metadata.logreg[1,2])/sum(matrix.metadata.logreg[1,])

ROC.metadata.logreg <- prediction(predictions.metadata.logreg, test$TrumpWrote)
ROC.metadata.logreg.df <- data.frame(fpr=slot(performance(ROC.metadata.logreg, "tpr", "fpr"),"x.values")[[1]],tpr=slot(performance(ROC.metadata.logreg, "tpr", "fpr"),"y.values")[[1]])
AUC.metadata.logreg <- performance(ROC.metadata.logreg, "auc")@y.values[[1]]

### CART

cart.metadata = rpart(factor(TrumpWrote) ~ ., data=train, method="class", cp = .003)

prp(cart.metadata, digits=3, split.font=1, varlen = 0, faclen = 0)

predictions.metadata.cart <- predict(cart.metadata, newdata=test, type="class")
matrix.metadata.cart = table(test$TrumpWrote, predictions.metadata.cart)
accuracy.metadata.cart = (matrix.metadata.cart[1,1]+matrix.metadata.cart[2,2])/nrow(test)
TPR.metadata.cart = (matrix.metadata.cart[2,2])/sum(matrix.metadata.cart[2,])
FPR.metadata.cart = (matrix.metadata.cart[1,2])/sum(matrix.metadata.cart[1,])

ROC.metadata.cart<- prediction(predict(cart.metadata, newdata=test)[,2], test$TrumpWrote)
ROC.metadata.cart.df <- data.frame(fpr=slot(performance(ROC.metadata.cart, "tpr", "fpr"),"x.values")[[1]],tpr=slot(performance(ROC.metadata.cart, "tpr", "fpr"),"y.values")[[1]])
AUC.metadata.cart <- as.numeric(performance(ROC.metadata.cart, "auc")@y.values)

### Random Forest

train$TrumpWrote = as.factor(train$TrumpWrote)
test$TrumpWrote = as.factor(test$TrumpWrote)

set.seed(123)
rfmodel.metadata = randomForest(TrumpWrote ~., data=train)

importance.rf.metadata <- data.frame(imp=round(importance(rfmodel.metadata)[order(-importance(rfmodel.metadata)),],2))

predictions.metadata.RF = predict(rfmodel.metadata, newdata=test)
matrix.metadata.RF = table(test$TrumpWrote, predictions.metadata.RF)
accuracy.metadata.RF = (matrix.metadata.RF[1,1]+matrix.metadata.RF[2,2])/nrow(test)
TPR.metadata.RF = (matrix.metadata.RF[2,2])/sum(matrix.metadata.RF[2,])
FPR.metadata.RF = (matrix.metadata.RF[1,2])/sum(matrix.metadata.RF[1,])

ROC.metadata.RF<- prediction(predict(rfmodel.metadata,newdata=test,type="prob")[,2], test$TrumpWrote)
ROC.metadata.RF.df <- data.frame(fpr=slot(performance(ROC.metadata.RF, "tpr", "fpr"),"x.values")[[1]],tpr=slot(performance(ROC.metadata.RF, "tpr", "fpr"),"y.values")[[1]])
AUC.metadata.RF <- as.numeric(performance(ROC.metadata.RF, "auc")@y.values)

### Summary

summary.performance.metadata <- data.frame (
  accuracy=round(c(accuracy.baseline,accuracy.metadata.logreg,accuracy.metadata.cart,accuracy.metadata.RF),3),
  TPR=round(c(TPR.baseline,TPR.metadata.logreg,TPR.metadata.cart,TPR.metadata.RF),3),
  FPR=round(c(FPR.baseline,FPR.metadata.logreg,FPR.metadata.cart,FPR.metadata.RF),3),
  AUC=round(c(AUC.baseline,AUC.metadata.logreg,AUC.metadata.cart,AUC.metadata.RF),3)
)


print(ggplot() +
        geom_line(data=ROC.metadata.logreg.df,aes(x=fpr,y=tpr,colour="a"),lwd=1,lty=1) +
        geom_line(data=ROC.metadata.cart.df,aes(x=fpr,y=tpr,colour="b"),lwd=1,lty=2) +
        geom_line(data=ROC.metadata.RF.df,aes(x=fpr,y=tpr,colour="c"),lwd=1,lty=3) +
        xlab("False Positive Rate") +
        ylab("True Positive Rate") +
        theme_bw() +
        xlim(0, 1) +
        ylim(0, 1) +
        scale_color_manual(name="Method", labels=c("a"="Logistic regression", "b"="CART", "c"="Random forest"), values=c("a"="gray", "b"="blue", "c"="red")) +
        theme(axis.title=element_text(size=18), axis.text=element_text(size=18), legend.text=element_text(size=18), legend.title=element_text(size=18)))

##### Sentiment analysis

### Update of dataset
# downloading NRC dataset
# https://saifmohammad.com/WebPages/lexicons.html

nrc <- read.csv('NRC-Emotion-Lexicon-Wordlevel-v0.92.csv', header=FALSE)
colnames(nrc) <- c('word', 'sentiment', 'indicator')
nrc <- nrc %>%
  filter(indicator==1) %>%
  dplyr::select(word, sentiment)

nrc_wide <- dcast(nrc, word ~ sentiment)

# convert nrc wide to be 0 and 1 values
nrc_wide$anger = ifelse(is.na(nrc_wide$anger),0,1)
nrc_wide$anticipation = ifelse(is.na(nrc_wide$anticipation),0,1)
nrc_wide$disgust = ifelse(is.na(nrc_wide$disgust),0,1)
nrc_wide$fear = ifelse(is.na(nrc_wide$fear),0,1)
nrc_wide$joy = ifelse(is.na(nrc_wide$joy),0,1)
nrc_wide$negative = ifelse(is.na(nrc_wide$negative),0,1)
nrc_wide$positive = ifelse(is.na(nrc_wide$positive),0,1)
nrc_wide$sadness = ifelse(is.na(nrc_wide$sadness),0,1)
nrc_wide$surprise = ifelse(is.na(nrc_wide$surprise),0,1)
nrc_wide$trust = ifelse(is.na(nrc_wide$trust),0,1)
nrc_wide

nrc %>% filter(word=="lawful")
nrc_wide %>% filter(word=="lawful")

# take tweets, break into one word per line, so tweetid - word
reg <- "([^A-Za-z\\d#@']|'(?![A-Za-z\\d#@]))"
tweet_words <- tweets.trump %>%
  filter(!str_detect(text, '^"')) %>%
  mutate(text = str_replace_all(text, "https://t.co/[A-Za-z\\d]+|&amp;", "")) %>%
  mutate(text = str_replace_all(text, "'", "")) %>%
  unnest_tokens(word, text, token = "regex", pattern = reg) %>%
  filter(!word %in% stop_words$word,
         str_detect(word, "[a-z]")) %>%
  count(word, TrumpWrote, created_at)

# inner join
# sum over rows so (max) one row per tweetid with totals of sentiments
joined = inner_join(tweet_words, nrc_wide) %>%
  group_by(created_at) %>%
  summarise(anger = sum(anger), anticipation = sum(anticipation), disgust = sum(disgust), fear = sum(fear), joy = sum(joy),
            negative = sum(negative), positive = sum(positive), sadness = sum(sadness), surprise = sum(surprise), trust = sum(trust)) 

joined.full = full_join(tweets.trump,joined, by="created_at")

joined.full$anger = ifelse(is.na(joined.full$anger),0,joined.full$anger)
joined.full$anticipation = ifelse(is.na(joined.full$anticipation),0,joined.full$anticipation)
joined.full$disgust = ifelse(is.na(joined.full$disgust),0,joined.full$disgust)
joined.full$fear = ifelse(is.na(joined.full$fear),0,joined.full$fear)
joined.full$joy = ifelse(is.na(joined.full$joy),0,joined.full$joy)
joined.full$negative = ifelse(is.na(joined.full$negative),0,joined.full$negative)
joined.full$positive = ifelse(is.na(joined.full$positive),0,joined.full$positive)
joined.full$sadness = ifelse(is.na(joined.full$sadness),0,joined.full$sadness)
joined.full$surprise = ifelse(is.na(joined.full$surprise),0,joined.full$surprise)
joined.full$trust = ifelse(is.na(joined.full$trust),0,joined.full$trust)

# New models with sentiment
documentterms.metadata.sentiment <- documentterms.metadata
documentterms.metadata.sentiment$anger = joined.full$anger
documentterms.metadata.sentiment$anticipation = joined.full$anticipation
documentterms.metadata.sentiment$disgust = joined.full$disgust
documentterms.metadata.sentiment$fear = joined.full$fear
documentterms.metadata.sentiment$joy = joined.full$joy
documentterms.metadata.sentiment$negative = joined.full$negative
documentterms.metadata.sentiment$positive = joined.full$positive
documentterms.metadata.sentiment$sadness = joined.full$sadness
documentterms.metadata.sentiment$surprise = joined.full$surprise
documentterms.metadata.sentiment$trust = joined.full$trust

### Splitting into training and test set
split1 = (tweets.trump$created_at < "2016-06-01")
split2 = (tweets.trump$created_at >= "2016-06-01")
train = documentterms.metadata.sentiment[split1,]
test = documentterms.metadata.sentiment[split2,]

### Logistic regression

logreg.metadata.sentiment = glm(TrumpWrote ~., data=train, family="binomial")
summary(logreg.metadata.sentiment)

predictions.metadata.sentiment.logreg <- predict(logreg.metadata.sentiment, newdata=test, type="response")
matrix.metadata.sentiment.logreg = table(test$TrumpWrote, predictions.metadata.sentiment.logreg > 0.5)
accuracy.metadata.sentiment.logreg = (matrix.metadata.sentiment.logreg[1,1]+matrix.metadata.sentiment.logreg[2,2])/nrow(test)
TPR.metadata.sentiment.logreg = (matrix.metadata.sentiment.logreg[2,2])/sum(matrix.metadata.sentiment.logreg[2,])
FPR.metadata.sentiment.logreg = (matrix.metadata.sentiment.logreg[1,2])/sum(matrix.metadata.sentiment.logreg[1,])

ROC.metadata.sentiment.logreg <- prediction(predictions.metadata.sentiment.logreg, test$TrumpWrote)
ROC.metadata.sentiment.logreg.df <- data.frame(fpr=slot(performance(ROC.metadata.sentiment.logreg, "tpr", "fpr"),"x.values")[[1]],tpr=slot(performance(ROC.metadata.sentiment.logreg, "tpr", "fpr"),"y.values")[[1]])
AUC.metadata.sentiment.logreg <- performance(ROC.metadata.sentiment.logreg, "auc")@y.values[[1]]

### CART

cart.metadata.sentiment = rpart(factor(TrumpWrote) ~ ., data=train, method="class", cp = .003)


prp(cart.metadata.sentiment, digits=3, split.font=1, varlen = 0, faclen = 0)

predictions.metadata.sentiment.cart <- predict(cart.metadata.sentiment, newdata=test, type="class")
matrix.metadata.sentiment.cart = table(test$TrumpWrote, predictions.metadata.sentiment.cart)
accuracy.metadata.sentiment.cart = (matrix.metadata.sentiment.cart[1,1]+matrix.metadata.sentiment.cart[2,2])/nrow(test)
TPR.metadata.sentiment.cart = (matrix.metadata.sentiment.cart[2,2])/sum(matrix.metadata.sentiment.cart[2,])
FPR.metadata.sentiment.cart = (matrix.metadata.sentiment.cart[1,2])/sum(matrix.metadata.sentiment.cart[1,])

ROC.metadata.sentiment.cart<- prediction(predict(cart.metadata.sentiment, newdata=test)[,2], test$TrumpWrote)
ROC.metadata.sentiment.cart.df <- data.frame(fpr=slot(performance(ROC.metadata.sentiment.cart, "tpr", "fpr"),"x.values")[[1]],tpr=slot(performance(ROC.metadata.sentiment.cart, "tpr", "fpr"),"y.values")[[1]])
AUC.metadata.sentiment.cart <- as.numeric(performance(ROC.metadata.sentiment.cart, "auc")@y.values)

### Random Forest

train$TrumpWrote = as.factor(train$TrumpWrote)
test$TrumpWrote = as.factor(test$TrumpWrote)

set.seed(123)
rfmodel.metadata.sentiment = randomForest(TrumpWrote ~., data=train)

importance.rf.metadata.sentiment <- data.frame(imp=round(importance(rfmodel.metadata.sentiment)[order(-importance(rfmodel.metadata.sentiment)),],2))

predictions.metadata.sentiment.RF = predict(rfmodel.metadata.sentiment, newdata=test)
matrix.metadata.sentiment.RF = table(test$TrumpWrote, predictions.metadata.sentiment.RF)
accuracy.metadata.sentiment.RF = (matrix.metadata.sentiment.RF[1,1]+matrix.metadata.sentiment.RF[2,2])/nrow(test)
TPR.metadata.sentiment.RF = (matrix.metadata.sentiment.RF[2,2])/sum(matrix.metadata.sentiment.RF[2,])
FPR.metadata.sentiment.RF = (matrix.metadata.sentiment.RF[1,2])/sum(matrix.metadata.sentiment.RF[1,])

ROC.metadata.sentiment.RF<- prediction(predict(rfmodel.metadata.sentiment,newdata=test,type="prob")[,2], test$TrumpWrote)
ROC.metadata.sentiment.RF.df <- data.frame(fpr=slot(performance(ROC.metadata.sentiment.RF, "tpr", "fpr"),"x.values")[[1]],tpr=slot(performance(ROC.metadata.sentiment.RF, "tpr", "fpr"),"y.values")[[1]])
AUC.metadata.sentiment.RF <- as.numeric(performance(ROC.metadata.sentiment.RF, "auc")@y.values)
