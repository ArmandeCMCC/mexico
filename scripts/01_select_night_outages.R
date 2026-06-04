library(readr)
library(readxl)
library(tidyverse)
library(ggplot2)
library(sf)
library(stringi)
library(lubridate)

dist_outage_2017_2021_causa_simple <- read_csv("data/dist_outage_2017_2021_causa_simple.csv")
dist_outage_2017_2021_causa_simple$`RED AFECTADA`<-NULL
dist_outage_2017_2021_causa_simple$CAUSA_SIMPLE<-NULL
columns<-c("year","date","time","length_min","state","municipality","substation","node","reason")
colnames(dist_outage_2017_2021_causa_simple)<-columns

# define “night” as 22:00 to 06:00
night_start <- hms::as_hms("22:00:00")
night_end   <- hms::as_hms("06:00:00")
         
# compute start and end datetimes of each outage
df <- dist_outage_2017_2021_causa_simple %>%
  mutate(
    datetime_start = date + time,
    datetime_end = datetime_start + minutes(length_min)
  )
#Identify outages that overlap with the night period
df2 <- df %>%
  mutate(
    # define the night intervals for each outage date
    night_same_day_start = floor_date(datetime_start, "day") + hours(22),
    night_same_day_end   = floor_date(datetime_start, "day") + days(1),
    
    night_next_day_start = floor_date(datetime_start, "day") + days(1),
    night_next_day_end   = floor_date(datetime_start, "day") + days(1) + hours(6),
    
    overlaps_night =
      (datetime_start < night_same_day_end &
         datetime_end   > night_same_day_start) |
      (datetime_start < night_next_day_end &
         datetime_end   > night_next_day_start)
  )

night_outages<-df2%>%filter(overlaps_night==TRUE)
#Subset 2: Night outages lasting more than 3 hours
night_outages_3hrs <- night_outages %>%
  filter(length_min > 180)

night_outages_3hrs<-night_outages_3hrs[,c(1:6,10,11)]

write_csv(night_outages_3hrs, file="data/night_outages_3hrs.csv")
