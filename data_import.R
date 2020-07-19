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


##### Importation of data and descriptive statistics

### Data importation
url <- 'http://www.trumptwitterarchive.com/data/realdonaldtrump/%s.json'
all_tweets <- map(2009:2017, ~sprintf(url, .x)) %>%
  map_df(jsonlite::fromJSON, simplifyDataFrame = TRUE) %>%
  mutate(created_at = parse_date_time(created_at, "a b! d! H!:M!:S! z!* Y!")) %>%
  tbl_df()

### Restriction to Twitter data and definition of iPhone/Android fields
tweets <- all_tweets %>%
  select(id_str, source, text, created_at) %>%
  filter(source %in% c("Twitter for iPhone", "Twitter for Android")) %>%
  mutate(source = ifelse(source=="Twitter for iPhone", "iPhone", source)) %>%
  mutate(source = ifelse(source=="Twitter for Android", "Android", source))

### Descriptive plots at the aggregate level
tweets %>% filter(year(with_tz(created_at, "EST"))>2014, year(with_tz(created_at, "EST"))<2017) %>%
  count(source, hour = hour(with_tz(created_at, "EST"))) %>%
  mutate(percent = n / sum(n)) %>%
  ggplot(aes(hour, percent, color = source)) +
  geom_line(lwd=2) +
  scale_y_continuous(labels = percent_format()) +
  theme(title=element_text(size=18),axis.title=element_text(size=18), axis.text=element_text(size=18),legend.text=element_text(size=18)) +
  labs(title="Proportion of tweets by time of day, per source",
       x = "Hour of day (EST)",
       y = "% of tweets",
       color = "") +
  scale_color_brewer(palette="Set1")

tweets %>%
  count(source,
        quoted = ifelse(str_detect(text, '^"'), "Quoted", "Not quoted")) %>%
  ggplot(aes(source, n, fill = quoted)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme(title=element_text(size=18),axis.title=element_text(size=18), axis.text=element_text(size=18),legend.text=element_text(size=18)) +
  labs(x = "", y = "Number of tweets", fill = "") +
  ggtitle('Whether tweets start with a quotation mark (")') +
  scale_fill_brewer(palette="Dark2")


### Restriction to the period of interest

tweets <- tweets %>%
  filter(created_at < "2017-03-01" & created_at > "2015-06-01")

### Descriptive plots at the word level

# Preparation of the dataset
# L notation ensures that the number is stored as an integer not a double
tweets.source <- tweets %>%
  mutate(fromiPhone = ifelse(source=="iPhone", 1L, 0L)) %>%
  select(source, fromiPhone, text, created_at)

 # Break down tweets into one word per line
reg <- "([^A-Za-z\\d#@']|'(?![A-Za-z\\d#@]))"

tweet_words <- tweets.source %>%
  filter(!str_detect(text, '^"')) %>%
  mutate(text = str_replace_all(text, "https://t.co/[A-Za-z\\d]+|&amp;", "http")) %>% # replaciong links by "http"
  mutate(text = str_replace_all(text, "'", "")) %>%
  mutate(text = str_replace_all(text, "badly", "bad")) %>% #manual stemming
  unnest_tokens(word, text, token = "regex", pattern = reg) %>%
  filter(!word %in% stop_words$word,
         str_detect(word, "[a-z]")) %>%
  count(word, fromiPhone, created_at, source)

tweet_words = data.frame(tweet_words)

Android_iPhone_ratios <- tweet_words %>%
  group_by(word) %>%
  filter(sum(n) >= 40) %>%
  spread(source, n, fill = 0) %>%
  ungroup() %>%
  mutate(ID.iPhone = ifelse(is.na(iPhone/sum(iPhone)),0,iPhone/sum(iPhone))) %>%
  mutate(ID.Android = ifelse(is.na(Android/sum(Android)),0,Android/sum(Android))) %>%
  group_by(word) %>%
  summarise(ID.iPhone = sum(ID.iPhone), ID.Android = sum(ID.Android)) %>%
  ungroup() %>%
  mutate(logratio = ifelse(ID.iPhone==0,10,ifelse(ID.Android==0,-10,log2( ID.Android / ID.iPhone)))) %>%
  arrange(desc(logratio))


Android_iPhone_ratios  %>%
  filter(logratio > 0) %>%
  top_n(20, logratio) %>%
  ungroup() %>%
  mutate(word = reorder(word, logratio)) %>%
  ggplot(aes(word, logratio)) +
  geom_bar(stat = "identity", fill='red', show.legend=FALSE) +
  coord_flip() +
  ylim(0,10) +
  ylab("Android/iPhone log ratio") +
  xlab("") +
  theme(axis.text.x=element_text(size=18), axis.text.y=element_text(size=18))


Android_iPhone_ratios  %>%
  filter(logratio < 0) %>%
  top_n(20, -logratio) %>%
  ungroup() %>%
  mutate(word = reorder(word, logratio)) %>%
  ggplot(aes(word, logratio)) +
  geom_bar(stat = "identity", fill='lightblue', show.legend=FALSE) +
  coord_flip() +
  ylim(-10,0) +
  ylab("Android/iPhone log ratio") +
  xlab("") +
  theme(axis.text.x=element_text(size=18), axis.text.y=element_text(size=18))


##### Preparation of structured dataset
### Defining dependent variable and restriction of data sample

tweets.trump <- tweets %>%
  mutate(TrumpWrote = ifelse(source=="iPhone", 0L, 1L)) %>%
  select(TrumpWrote, text, created_at)

# Writing data file
write.csv(tweets.trump, "trump_tweets.csv")
