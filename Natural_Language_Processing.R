library(data.table)
library(tidyverse)
library(text2vec)
library(caTools)
library(glmnet)
library(stopwords)

setwd("C:/Users/99470/Downloads")
df <- read_csv('nlpdata.csv')

# Split data
set.seed(123)
split <- df$Liked %>% sample.split(SplitRatio = 0.8)
train <- df %>% subset(split == T)
test <- df %>% subset(split == F)

#preprocessing
it_train <- train$Review %>% 
  itoken(preprocessor = tolower, 
         tokenizer = word_tokenizer,
         ids = train$...1,
         progressbar = F) 

#create vocabluary
vocab <- it_train %>% create_vocabulary()
vocab %>% 
  arrange(desc(term_count)) %>% 
  head(110) %>% 
  tail(10)

#create vector and matrix
vectorizer <- vocab %>% vocab_vectorizer()
dtm_train <- it_train %>% create_dtm(vectorizer)

#checking dimention(1st:number of rows(reviews),2nd:number of words(columns created))
dtm_train %>% dim()
#checking if words splited by order while spliting sentence to words
identical(as.numeric(rownames(dtm_train)), (train$...1))

# Modeling ----
glmnet_classifier <- dtm_train %>% 
  cv.glmnet(y = train[['Liked']],
            family = 'binomial', 
            type.measure = "auc",
            nfolds = 10,
            thresh = 0.001,# high value is less accurate, but has faster training
            maxit = 1000)# again lower number of iterations for faster training

glmnet_classifier$cvm %>% max() %>% round(3) %>% paste("-> Max AUC")

#applying model on test data
it_test <- test$Review %>% tolower() %>% word_tokenizer()

it_test <- it_test %>% 
  itoken(ids = test$...1,
         progressbar = F)

dtm_test <- it_test %>% create_dtm(vectorizer)

preds <- predict(glmnet_classifier, dtm_test, type = 'response')[,1] #response-un yerine class yazanda prediction 0/1 kimi verir, reponse-da ise probability kimi
glmnet:::auc(test$Liked, preds) %>% round(2)

#create vocabluary (stopwords and ngram applied)
stopwords=stopwords("en",source="stopwords-iso")
vocab <- it_train %>% create_vocabulary(stopwords = stopwords, ngram = c(1L,2L))
vocab %>% 
  arrange(desc(term_count)) %>% 
  head(110) %>% 
  tail(10)

#create vector and matrix
vectorizer <- vocab %>% vocab_vectorizer()
dtm_train <- it_train %>% create_dtm(vectorizer)

#checking dimention(1st:number of rows(reviews),2nd:number of words(columns created))
dtm_train %>% dim()
#checking if words splited by order while spliting sentence to words
identical(as.numeric(rownames(dtm_train)), train$...1)

# Modeling ----
glmnet_classifier <- dtm_train %>% 
  cv.glmnet(y = train[['Liked']],
            family = 'binomial', 
            type.measure = "auc",
            nfolds = 10,
            thresh = 0.001,# high value is less accurate, but has faster training
            maxit = 1000)# again lower number of iterations for faster training

glmnet_classifier$cvm %>% max() %>% round(3) %>% paste("-> Max AUC")

#applying model on test data
it_test <- test$Review %>% tolower() %>% word_tokenizer()

it_test <- it_test %>% 
  itoken(ids = test$...1,
         progressbar = F)

dtm_test <- it_test %>% create_dtm(vectorizer)

preds <- predict(glmnet_classifier, dtm_test, type = 'response')[,1] #response-un yerine class yazanda prediction 0/1 kimi verir, reponse-da ise probability kimi
glmnet:::auc(test$Liked, preds) %>% round(2)

#interpretation: i would say model performs quite good. there is no overfitting or underfitting. but after applying the stopwords and ngram, the performance of the model decreased.