## Table of Contents
1. [Installation](#installation)
2. [Introducing a Dataset](#dataset-introduction)
3. [Project Motivation](#project-motivation)
4. [Data Preparation](#data-preparation)
5. [File Descriptions](#files)
4. [Results](#results)

### Installation <a name="installation"></a>
For running this project, the most important library is Python version of Anaconda Distribution. It installs all necessary packages for analysis and building models. 
 
### Introducing a Dataset <a name="dataset-introduction"></a>
This data set contains simulated data that mimics customer behavior on the Starbucks rewards mobile app. Once every few days, Starbucks sends out an offer to users of the mobile app. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). Some users might not receive any offer during certain weeks.

Not all users receive the same offer, and that is the challenge to solve with this data set.

### Project Motivation <a name="project-motivation"></a>
I chose this project to understand the success rate of offers being sent and analysis is done through addressing the following questions.
1. How many customers were provided with a specific offer?
2. What's the performance level of an offer?

### Data Preparation <a name="data-preparation"></a>
There are three datasets provided and each dataset is cleaned and preprocessed for further analysis. The target features for analysis are offer_success, percent_success.

1. Portfolio - renaming id column name to offer_id, one-hot encoding of channels and offer_type columns
2. Profile - profile: renaming id column name to customer_id, replacing age value 118 to nan, creating readable date format in became_member_on column, dropping rows with no gender, income, age data, converting gender values to numeric 0s and 1s, adding start year and start month columns (for further analysis)
3. Transcript - renaming person column name to customer_id, creating separate columns for amount and offer_id from value col, dropping transaction rows whose customer_id is not in profile:customer_id, converting time in hours to time in days, segregating offer and transaction data, finally dropping duplicates if any

### File Descriptions <a name="files"></a>
There is a notebook available here to showcase work related to the above questions and wrangling process. There are 3 data files used to address the above qustions
1. portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
2. profile.json - demographic data for each customer
3. transcript.json - records for transactions, offers received, offers viewed, and offers completed

## Results<a name="results"></a>
The main observations of the code are published [here](https://medium.com/@omkarbhad66/starbucks-challenge-f131242398c9).
