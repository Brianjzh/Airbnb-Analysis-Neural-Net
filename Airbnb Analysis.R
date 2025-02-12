
# load libraries
library(tidyverse)
library(GGally) # for data exploration
library(caret) #For confusionMatrix(), training ML models, and more
library(neuralnet) #For neuralnet() function
library(dplyr) #For some data manipulation and ggplot
library(fastDummies) #To create dummy variable (one hot encoding)
library(sigmoid) #For the relu activation function

# set a random seed for reproducibility
set.seed(123)
#Load datasets for Airbnb Listings and Reviews
complete_data = read.csv("AirbnbListings.csv")


#1 Report your exploratory analysis of the data. This can include data visualization,
#summary tables, changes made to the data, or any other insightful findings about the data.

# Merge the listing and review data on the 'id' and 'listing_id' columns, respectively
# 512 Missing values in total.
total_missing <- sum(is.na(complete_data))

# Calculate the mean of the host acceptance rate, excluding NA values
mean_acceptance_rate <- mean(complete_data$host_acceptance_rate, na.rm = TRUE)

# Replace NA values in the host acceptance rate with the calculated mean
complete_data_draft <- complete_data %>%
  mutate(host_acceptance_rate = ifelse(is.na(host_acceptance_rate), mean_acceptance_rate, host_acceptance_rate))

# Replace NA values in room_type with appropriate values based on the number of bathrooms
complete_data_draft <- complete_data_draft %>%
  mutate(room_type = case_when(
    is.na(room_type) & grepl("private", bathrooms, ignore.case = TRUE) ~ "Private room",
    is.na(room_type) & grepl("shared", bathrooms, ignore.case = TRUE) ~ "Shared room",
    is.na(room_type) ~ "Entire home/apt",
    TRUE ~ room_type
  ))

# Calculate the mean number of bedrooms, rounding up to the nearest integer, excluding NA values
mean_bedrooms <- ceiling(mean(complete_data_draft$bedrooms, na.rm = TRUE))

# Remove non-numeric characters from the 'bath' column and convert it to numeric
# Creates a new column 'bath_n' for the number of bathrooms
complete_data_draft$bath_n <- as.numeric(gsub("[^0-9.]", "", complete_data_draft$bath))

# Replace NA or empty values in the bedrooms column based on the number of beds
complete_data_draft <- complete_data_draft %>%
  mutate(bedrooms = case_when(
    is.na(bedrooms) & beds > 1 ~ mean_bedrooms,
    is.na(bedrooms) & beds == 1 ~ 1,
    bedrooms == "" & beds > 1 ~ mean_bedrooms,
    bedrooms == "" & beds == 1 ~ 1,
    TRUE ~ bedrooms
  ))

# Print the updated dataframe to the console
print(complete_data_draft)


# Check if there are any missing values in each column of the updated dataframe
apply(complete_data_draft, 2, anyNA)

# Identify the rows with missing values in the 'avg_rating' column
missing_row_avg_rating = which(is.na(complete_data_draft$avg_rating))

# Replace missing values in the 'avg_rating' column with the mean of the column values
complete_data_draft$avg_rating[missing_row_avg_rating] = mean(complete_data_draft$avg_rating, na.rm = TRUE)

# Check again for any missing values in each column
apply(complete_data_draft, 2, anyNA)

# Print the names of the columns in the updated dataframe
names(complete_data_draft)



str(complete_data_draft)
summary(complete_data_draft)
colSums(is.na(complete_data_draft))


#-------------------------------------------------------------------------------------------
final_data <- complete_data_draft |>
  # Convert `host_since` to numeric (days since host joined)
  mutate(host_since = as.numeric(Sys.Date() - as.Date(host_since, format = "%m/%d/%Y"))) |>
  
  # Convert `superhost` (logical) to numeric (1 for TRUE, 0 for FALSE)
  mutate(superhost = as.numeric(superhost)) |>
  
  # Create dummy variables for categorical features
  dummy_cols(
    select_columns = c('neighborhood','room_type'),
    remove_selected_columns = TRUE,
    remove_first_dummy = TRUE
  )



str(final_data)
### Step 1: Create a train/test split ----
test_idx <- createDataPartition(
  final_data$price,
  p = 0.2,
  list = FALSE
)

train_data <- final_data[-test_idx, ]

test_data <- final_data[test_idx, ]

colnames(train_data) <- make.names(colnames(train_data))
colnames(test_data) <- make.names(colnames(test_data))

# I am dropping bathrooms because I have the bath_n variable that I set up
train_data <- train_data %>% select(-listing_id)
test_data <- test_data %>% select(-listing_id)

train_data <- train_data %>% select(-bathrooms)
test_data <- test_data %>% select(-bathrooms)


# normalize the data. Do not normalize price.
normalizer <- preProcess(
  train_data |>
    select(where(is.numeric)),  # Selects all numeric (including integer) columns
  method = "range"
)


train_data <- predict(normalizer, train_data)

test_data <- predict(normalizer, test_data)

# Verify structure

str(train_data)

### Step 5: Feature & Model Selection ----

# Train a neural net with one hidden layer and 5 units using the neuralnet package
# Use a ReLu activation function
# if you don't converge change the stepmax parameter to a larger value or change the learning rate, learning rate, activation function, re-scaling the data. Standardization or normalization


nn1 <- neuralnet(
  price ~ .,
  data = train_data,
  linear.output = TRUE,
  act.fct = tanh,
  hidden = 5,
  stepmax = 1e7,
  algorithm = "rprop+"
)


#str(p1_vec)
#summary(p1_vec)

plot(nn1)
summary(nn1)



nn2 <- train(
  price ~.,
  data = train_data,
  method = "nnet",
  linout = TRUE,
  trControl = trainControl( # store since we will reuse
    method = "cv", number = 10
  ),
  tuneGrid = expand.grid(
    size = 1:10, 
    decay = c(0.01,0.05,0.1,0.15, 0.2)
  ),
  metric = "RMSE"
)

plot(nn2)

nn2$bestTune

### Step 7: Predictions and Conclusions ----


# get predictions from both models
p1 <- predict(nn1, test_data)

p2 <- predict(nn2, test_data)

# p1_vec <- as.numeric(p1)


# R2_value_nn <- R2(p1_vec, test_data$price)



# if you normalized price, convert predictions back to dollars

# calculate r-squared using either mvrsquared::calc_rsquared or caret::R2

# calculate RMSE and MAE (check caret for these functions)

results <- tibble(
  model = c("neuralnet", "caret-nnet"),
  r2 = c(R2(test_data$price, p1[,1]), R2(test_data$price, p2)),
  rmse = c(RMSE(test_data$price, p1[,1]), RMSE(test_data$price, p2)),
  mae = c(MAE(test_data$price, p1[,1]), MAE(test_data$price, p2)),
)

results
### -------------------------------------------------------------------------------- RSME value here 129, 86,4-------------------------####
# plot the original price versus the predicted price for both models
predictions <- 
  tibble(
    neuralnet = p1[,1],
    caret = p2,
    actual = test_data$price
  ) |>
  pivot_longer(
    all_of(c("neuralnet", "caret")),
    names_to = "model",
    values_to = "prediction"
  )

predictions |>
  ggplot(aes(x = prediction, y = actual, color = model)) + 
  geom_point(alpha = 0.5) +
  facet_wrap(~model, nrow = 2) +
  theme(legend.position = "none")

