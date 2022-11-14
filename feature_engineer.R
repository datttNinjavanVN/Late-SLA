library(bizdays)
library(fastDummies)
library(arrow)
library(tidyverse)

# Import raw data from SQL query
df_raw = read_parquet("sla_v2.pq")

# Preprocess the raw data
df_preprocess = df_raw %>% filter(prior_flag == 1) %>%
  filter(
  fm_duration_minutes > 0, 
  mm1_duration_minutes > 0, 
  mm2_duration_minutes > 0, 
  mm3_duration_minutes > 0)  %>% 
  mutate(not_late = 1 - act_late,
         sla_days = bizdays(as.Date(sla_date), as.Date(start_clock_date)),
         fm_duration_hours = round(fm_duration_minutes/60,0), 
         mm1_duration_hours = round(mm1_duration_minutes/60,0), 
         mm2_duration_hours = round(mm2_duration_minutes/60,0), 
         mm3_duration_hours = round(mm3_duration_minutes/60,0),
         mm_duration_hours = round(total_mm_duration_minutes/60,0))

# Store the selected variables obtained from WOE and IV
variables_selected = c("total_duration_days", "mm_duration_hours", "mm2_duration_hours", "dest_sector", "dest_area", "rts_flag", "mm3_duration_hours", "sla_days", "origin_sector", "fm_duration_hours", "sales_channel", "no_of_hub_movement", "route_type", "bkk_tag")

# Engineer feature
df_processed = df_preprocess %>%
  filter(total_duration_days != 1,
         !sales_channel %in% c("Cross Border","Shipper Support")) %>%
  mutate(total_duration_days = case_when(total_duration_days == 1 ~ "1",
                                         total_duration_days == 2 ~ "2",
                                         total_duration_days == 3 ~ "3",
                                         TRUE ~ "4 - 23"),
         mm_duration_hours = case_when(between(mm_duration_hours,4,10) ~ "4 - 10",
                                       between(mm_duration_hours,13,15) ~ "13 - 15",
                                       between(mm_duration_hours,16,25) ~ "16 - 25",
                                       between(mm_duration_hours,26,36) ~ "26 - 36",
                                       between(mm_duration_hours,37,43) ~ "37 - 43",
                                       between(mm_duration_hours,44,58) ~ "44 - 58",
                                       between(mm_duration_hours,59,62) ~ "59 - 62",
                                       between(mm_duration_hours,63,81) ~ "63 - 81",
                                       TRUE ~ "82 - 368"),
         mm2_duration_hours = case_when(between(mm2_duration_hours,0 ,7 ) ~ "0 - 7",
                                        between(mm2_duration_hours,8 ,9 ) ~ "8 - 9",
                                        between(mm2_duration_hours,10,12) ~ "10 - 12",
                                        between(mm2_duration_hours,13,19) ~ "13 - 19",
                                        between(mm2_duration_hours,20,31) ~ "20 - 31",
                                        between(mm2_duration_hours,32,39) ~ "32 - 39",
                                        between(mm2_duration_hours,40,54) ~ "40 - 54",
                                        between(mm2_duration_hours,55,59) ~ "55 - 59",
                                        between(mm2_duration_hours,60,69) ~ "60 - 69",
                                        TRUE ~ "70 - 347"),
         dest_area = ifelse(dest_area %in% c("island","urban"), "island & urban", "sub-urban"),
         mm3_duration_hours = case_when(mm3_duration_hours == 0 ~ "0",
                                        mm3_duration_hours == 1 ~ "1",
                                        between(mm3_duration_hours,2,4) ~ "2 - 4",
                                        TRUE ~ "5 - 144"),
         sla_days = ifelse(between(sla_days,5,8), "5 - 8", as.character(sla_days)),
         origin_sector = ifelse(origin_sector %in% c("HCM","DNB"), "HCM & DNB", origin_sector),
         fm_duration_hours = case_when(between(fm_duration_hours,5,6) ~ "5 - 6",
                                       between(fm_duration_hours,7,10) ~ "7 - 10",
                                       between(fm_duration_hours,11,340) ~ "11 - 340",
                                       TRUE ~ as.character(fm_duration_hours)),
         no_of_hub_movement = ifelse(between(no_of_hub_movement,4,8), "4 - 8", as.character(no_of_hub_movement)),
         rts_flag = as.character(rts_flag),
         bkk_tag = as.character(bkk_tag),
         not_late = as.integer(not_late)
  ) %>%
  select(not_late,variables_selected) %>% 
  dummy_cols(remove_first_dummy = TRUE,
  remove_selected_columns = TRUE)

df_processed %>% write_parquet("processed_prior.pq")






