'''
Move this code into your OWN SF_DAT_15_WORK repo

Please complete each question using 100% python code

If you have any questions, ask a peer or one of the instructors!

When you are done, add, commit, and push up to your repo

This is due 7/1/2015
'''


import pandas as pd
# pd.set_option('max_colwidth', 50)
# set this if you need to

killings = pd.read_csv('hw/data/police-killings.csv')
killings.head()
killings.describe()
killings.shape

# 1. Make the following changed to column names:
# lawenforcementagency -> agency
# raceethnicity        -> race
killings.rename(columns={'lawenforcementagency':'agency', 'raceethnicity':'race'}, inplace=True)
killings.describe()

# 2. Show the count of missing values in each column
killings.isnull().sum()

# 3. replace each null value in the dataframe with the string "Unknown"
killings.streetaddress.fillna(value='Unknown', inplace=True)
killings.isnull().sum()

# 4. How many killings were there so far in 2015?
killings.groupby('year').count().plot(kind='bar')
killings[(killings.year == 2015)].shape[0]


# 5. Of all killings, how many were male and how many female?
killings[(killings.gender == 'Male')].shape[0]
killings[(killings.gender == 'Female')].shape[0]

# 6. How many killings were of unarmed people?
killings[(killings.armed == 'No')].shape[0]


# 7. What percentage of all killings were unarmed?
killings[(killings.armed == 'No')].shape[0] / (killings.shape[0] + 0.00)


# 8. What are the 5 states with the most killings?
killings_state = killings.groupby('state').count()
killings_state.sort_index(by='city', ascending=False).head(5)


# 9. Show a value counts of deaths for each race
killings.groupby('race').city.count()

# 10. Display a histogram of ages of all killings
killings.age.hist(bins=20)

# 11. Show 6 histograms of ages by race
killings.age.hist(by=killings.race, bins=20, sharey=True, sharex=True)


# 12. What is the average age of death by race?
killings.groupby('race').age.mean()


# 13. Show a bar chart with counts of deaths every month
killings.groupby('month').city.count().plot(kind='bar',color='b')
#**how to sort by month

###################
### Less Morbid ###
###################

majors = pd.read_csv('hw/data/college-majors.csv')
majors.head()

# 1. Delete the columns (employed_full_time_year_round, major_code)
del majors['Employed_full_time_year_round']
del majors['Major_code']
majors.head()

# 2. Show the cout of missing values in each column
majors.isnull().sum()

# 3. What are the top 10 highest paying majors?
majors_salary = majors.groupby('Major').mean()
majors_salary = majors_salary.sort_index(by='P75th', ascending=False)
majors_salary.head(10)

# 4. Plot the data from the last question in a bar chart, include proper title, and labels!
majors_salary.P75th.head(10).plot(kind='bar', color='b', title='Top 10 Highest Paying Majors')

# 5. What is the average median salary for each major category?
majors_salary.sort_index(by='Median', ascending=False)

# 6. Show only the top 5 paying major categories
majors_salary = majors_salary.sort_index(by='P75th', ascending=False)
majors_salary.head(5)

# 7. Plot a histogram of the distribution of median salaries
majors.Median.hist(bins=20)

# 8. Plot a histogram of the distribution of median salaries by major category
majors.groupby('Major_category').count()
majors.Median.hist(by=majors.Major_category, bins=20, sharey=True, sharex=True)

#**Error message... ValueError: x has only one data point. bins or range kwarg must be given

# 9. What are the top 10 most UNemployed majors?
# What are the unemployment rates?
majors_salary = majors.groupby('Major').mean()
majors_salary = majors_salary.sort_index(by='Unemployed', ascending=False)
majors_salary.head(10)

#** Pandas hygiene - how to only pull requested column?

# 10. What are the top 10 most UNemployed majors CATEGORIES? Use the mean for each category
# What are the unemployment rates?
majors_unemployed = majors.groupby('Major_category').mean()
majors_unemployed = majors_unemployed.sort_index(by='Unemployed', ascending=False)
majors_unemployed.head(10)

#** Pandas hygiene - how to only pull requested column?

# 11. the total and employed column refer to the people that were surveyed.
# Create a new column showing the emlpoyment rate of the people surveyed for each major
# call it "sample_employment_rate"
# Example the first row has total: 128148 and employed: 90245. it's 
# sample_employment_rate should be 90245.0 / 128148.0 = .7042
majors['sample_employment_rate'] = majors['Employed'] / majors['Total']
majors

# 12. Create a "sample_unemployment_rate" colun
# this column should be 1 - "sample_employment_rate"
majors['sample_unemployment_rate'] = 1 - majors['sample_employment_rate']
majors
