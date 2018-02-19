# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 18:29:19 2017

@author: lippa2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV Files intp Python
allstar = pd.read_csv('AllstarFull.csv')
HallOfFame = pd.read_csv('HallOfFame.csv')
manager = pd.read_csv('Managers.csv')
team = pd.read_csv('Teams.csv')
salary = pd.read_csv('Salaries.csv')
schools = pd.read_csv('Schools.csv')
college_playing = pd.read_csv('CollegePlaying.csv')
#print salary.head()
#print allstar.head()
#print HallOfFame.head()
#print manager.head()
#print team.head()
#print schools.head()
#print college_playing.head()

"""Clean Data"""
# Change yearid to YearID
HallOfFame.rename(columns={'yearid': 'yearID'}, inplace=True)
#print HallOfFame.head()

# schools and collage_playing can be merged to one Dataframe
def combine_dataframes(schools, college_playing):
    
    merge = college_playing.merge(schools, on =  ['schoolID'], how = 'left')
    return merge

print combine_dataframes(schools, college_playing)

# merged Dataframes save in a new CSV file
school_playing = pd.DataFrame(combine_dataframes(schools, college_playing))
school_playing.to_csv('schools_playing.csv', index = False)

school_playing = pd.read_csv('schools_playing.csv')
print school_playing.head()

print len(allstar)
print len(allstar['playerID'].unique())
#5148
#1774

print len(HallOfFame)
print len(HallOfFame['playerID'].unique())
#4156
#1260

print len(manager)
print len(manager['playerID'].unique())
#3436
#698

print len(team)
print len(team['teamID'].unique())
#2835
#149

print len(salary)
print len(salary['playerID'].unique())
#26428
#5155


"""Questions:
   1. How is the Correlation of some Attributes?
   2. How is the average of salary changes? How often will the salary change (avg)?
   3. Compare the count of salary change with the change of salary.
   4. What is the change average of salary over the years?
   5. Which players start her Carrier in school and can be found in the Hall of Fame?
   6. Which states and cities have the most baseball colleges? (Show only Top 10)
   7. Reject or retain the null-hypothesis for win and losses of Seriespost data?
"""
"""Question 1____________________________________________________________"""
def correlation(x, y):
    x_std = (x - x.mean()) / x.std(ddof = 0)
    y_std = (y - y.mean()) / y.std(ddof = 0)
    
    return (x_std * y_std).mean()

"""Attributes to compare!"""

salary_c = salary['salary']
year_c = salary['yearID']
ballots_c = HallOfFame['ballots']
rank_c = team['Rank']

print correlation(ballots_c, rank_c)
# -0.134767987156
print correlation(salary_c, year_c)
# 0.35173999336
print correlation(year_c, rank_c)
#-0.0415697039104
print correlation(salary_c, rank_c)
#-0.000443072942508


"""Question 2____________________________________________________________"""

salary_avg =  salary.groupby('playerID').count()['salary'].mean()
print salary_avg 
# 5.12667313288 - the salary of baseball players change five times (avg)


"""Question 3____________________________________________________________"""
""" Unterfrage: Verteilung Anzahl durschnittliche Gehaltsveränderunge zu Gehaltsveränderung pro Spieler?"""
salary_data =  salary.groupby('playerID').count()['salary']
print salary_data
#playerID
#aardsda01     7
#aasedo01      4
#abadan01      1
#abadfe01      5
#abbotje01     4
# --> Count of Salary changes per player

# salary change per player:
# 1. Step get maximum of salary
salary_max = salary.groupby('playerID').max()['salary']
#print salary_max
#playerID
#aardsda01     4500000
#aasedo01       675000
#abadan01       327000

# 2. Step get minimum of salary
salary_min = salary.groupby('playerID').min()['salary']
#print salary_min
#playerID
#aardsda01     300000
#aasedo01      400000
#abadan01      327000

# 3. Step calculate delta (change)
delta = (salary_max-salary_min) / 1000000
print delta
#playerID before dividing 
#aardsda01     4200000
#aasedo01       275000
#abadan01            0

#playerID after dividing
#aardsda01    4.200
#aasedo01     0.275
#abadan01     0.000

plt.scatter(delta, salary_data, s = salary_data*8, c = salary_data, cmap='YlGnBu')
plt.ylabel('Count of Salary Changes per player')
plt.xlabel('Delta of Salary Change per player in $Mio.')


"""Question 4____________________________________________________________"""

salary_mean = salary.groupby('yearID').mean()['salary']
print salary_mean

year = salary['yearID'].unique()

plt.bar(year, salary_mean, color = 'turquoise')
plt.xlabel('Years')
plt.ylabel('Avg. Salary Change')


"""Question 5____________________________________________________________"""

#Compare playerID from College playern with playerID from HallofFame
compare = school_playing['playerID'].equals(HallOfFame['playerID'])
print compare

# False - means that not the exact players from school are in the hall of fame
players = school_playing.merge(HallOfFame, how = 'inner', on ='playerID')
print players['playerID'].unique()
print len(players['playerID'].unique())
# 386 players they starts carrere in school are later in the Hall of Fame


# Calculate Percentage
career_percentage = (float(len(players['playerID'].unique()))/float(len(HallOfFame['playerID'].unique())))*100
print career_percentage
#30.6349206349

labels = 'Career from School', 'Other'
data = [career_percentage, 100-career_percentage]
color = ['turquoise', 'darkgrey']
plt.pie(data, labels=labels, explode=[0.1,0], colors=color,autopct='%1.1f%%', shadow=True, startangle=140)



"""Question 6____________________________________________________________"""

college_state = school_playing.groupby('state').count()['city']
print college_state

college_city = school_playing.groupby('city').count()['state']
print college_city


def sort_columns(column):
    sorted_column = column.sort_values(ascending = False)
    return sorted_column.iloc[0:10]


# Top 10 States
state_count = pd.Series(college_state)
top10state =  sort_columns(state_count)
print top10state
# Top10 of States
#CA    2948
#TX    1281
#FL    1056
#NC     749
#PA     718
#NY     635
#IL     618
#OH     546
#AZ     524
#MA     500


top10state.plot(color='PaleGreen', linewidth = 8)



# Top10 cities
city_count = pd.Series(college_city)
top10cities = sort_columns(city_count)
print top10cities

# Top 10 of Cities
#Los Angeles    583
#Austin         288
#Palo Alto      248
#Tempe          236
#New York       227
#Columbia       194
#Ann Arbor      191
#Baton Rouge    182
#San Diego      173
#Tucson         173

top10cities.plot(color='Royalblue', linewidth = 8)
plt.xticks(rotation=45)



"""Question 7: Statistical Test: t-Test_________________________________________________"""

# Create dataframe only with needed columns
from pandasql import sqldf
seriespost = pd.read_csv('SeriesPost.csv')
print seriespost.head()
    
pysqldf = lambda q: sqldf(q, globals())

# Needed Columns are the yearID, round, wins and losses
# relevant data are only the results of the world series (WS)
ws = pysqldf("SELECT yearID, round, wins, losses FROM seriespost WHERE round == 'WS';")
print ws.head()


# Histograms for win and loss
plt.hist(ws['wins'])
plt.title('Histograms for wins')

plt.hist(ws['losses'])
plt.title('Histograms for losses')

# boxplots for win and loss
sns.boxplot(ws['wins'], orient='h', color='purple')
sns.boxplot(ws['losses'], orient='h', color='lightgreen')

# Calculate some statistical values:
mean_win = np.mean(ws['wins'])
mean_loss = np.mean(ws['losses'])

print 'Mean win / loss: '
print (mean_win, mean_loss)
#(4.092436974789916, 1.8403361344537814)

median_win = np.median(ws['wins'])
median_loss = np.median(ws['losses'])

print 'Median win / loss: '
print (median_win, median_loss)
#(4.0, 2.0)

w_var = np.var(ws['wins'])
l_var = np.var(ws['losses'])

print 'Variance win / loss: '
print(w_var, l_var)
#(0.4200268342631171, 1.4114822399548062)

w_std = np.std(ws['wins'])
l_std = np.std(ws['losses'])

print 'Standard deviation win / loss: '
print(w_std, l_std)
#(0.6480947725935745, 1.188058180374516)

# degrees of freedom
df = ws['yearID'].count() - 1
print 'Degrees of freedom: '
print df
# 118

"""Calculating of depending paried t-Test_________________________________"""

from scipy import stats as st
print st.ttest_rel(ws['wins'], ws['losses'])
# statistic=21.06929892563307, pvalue=8.4287409022462103e-42)


