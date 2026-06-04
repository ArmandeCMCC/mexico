library(readr)
library(readxl)
library(tidyverse)
library(ggplot2)
library(sf)
library(stringi)

# load night outages lasting more than 3 hours
night_outages_3hrs <- read_csv("data/night_outages_3hrs.csv")

# load the Mexico municipal boundaries from GADM
mex_mun <- st_read("data/gadm41_MEX.gpkg", layer = "ADM_ADM_2")

## match names
mex_mun_clean <- mex_mun %>%
  mutate(
    state_clean = stri_trans_general(toupper(NAME_1), "Latin-ASCII"),
    municipality_clean = stri_trans_general(toupper(NAME_2), "Latin-ASCII")
  )

night_long_clean <- night_outages_3hrs %>%
  mutate(
    state_clean = stri_trans_general(toupper(state), "Latin-ASCII"),
    municipality_clean = stri_trans_general(toupper(municipality), "Latin-ASCII")
  )

setdiff(unique(night_long_clean$municipality_clean),
        unique(mex_mun_clean$municipality_clean))

night_long_clean <- night_long_clean %>%
  mutate(
    municipality_clean = municipality_clean %>%
      str_replace_all("GRAL\\.", "GENERAL") %>%
      str_replace_all("FCO\\.", "FRANCISCO") %>%
      str_replace_all("R\\.", "RIO") %>%
      str_replace_all("[|\\-]", "") %>%
      str_squish() %>%
      stri_trans_general("Latin-ASCII")
  )

outage_count <- night_long_clean %>%
  group_by(state_clean, municipality_clean) %>%
  summarise(total_outages = n(), .groups = "drop")

mex_joined <- mex_mun_clean %>%
  left_join(outage_count, 
            by = c("state_clean", "municipality_clean"))

mex_joined$total_outages[is.na(mex_joined$total_outages)] <- 0

ggplot(mex_joined) +
  geom_sf(aes(fill = total_outages), color = NA) +
  scale_fill_viridis_c(option = "plasma") +
  labs(
    title = "Total Night-time Outages >3h by Municipality",
    fill = "Outages"
  ) +
  theme_minimal()


## merge full data with shapefile
## save the data
night_long_clean <- night_long_clean %>%
  left_join(mex_mun_clean, 
            by = c("state_clean", "municipality_clean"))


night_long_clean <- st_drop_geometry(night_long_clean)

write_csv(night_long_clean, file="data/night_outages_3hrs_with_locations.csv")
