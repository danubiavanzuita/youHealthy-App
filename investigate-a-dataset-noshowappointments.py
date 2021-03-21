#!/usr/bin/env python
# coding: utf-8

# # Project: Investigate a Dataset (noshowappointments DataSet)
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# ### Overview
# 
# >To perform the data analysis, the noshowappointments data set will be used
# 
# >The data set contains information on more than 110,000 medical appointments. The information is structured in fourteen columns.
# >
# >Total Appointments vs Gender
# >
# >Attendance and Non-Attendance vs Gender
# >
# >Patients who did not attend and are considered to be hypertensive
# >
# >Patients who did not attend and are considered diabetic
# >
# >Patients who did not attend and are considered alcoholics
# >
# >Patients who did not attend and are enrolled in social programs
# >
# >Date with the highest number of non-attendance to consultations
# >
# >Day of Week with the highest number of non-attendance to consultations
# >
# 

# In[33]:


# Use this cell to set up import statements for all of the packages that you
#   plan to use.

# Remember to include a 'magic word' so that your visualizations are plotted
#   inline with the notebook. See this page for more:
#   http://ipython.readthedocs.io/en/stable/interactive/magics.html

import pandas as pd
import csv
import numpy as np
import datetime
import calendar
import matplotlib.pyplot as plt
import seaborn as sns



# <a id='wrangling'></a>
# ## Data Wrangling
# 
# > **Tip**: In this section of the report, you will load in the data, check for cleanliness, and then trim and clean your dataset for analysis. Make sure that you document your steps carefully and justify your cleaning decisions.
# 
# ### General Properties
# 

# <a id='wrangling'></a>
# ### Load Data
# 
# > Loading data from the file noshowappointments-kagglev2-may-2016.csv
# 

# In[34]:


# Load your data and print out a few lines. Perform operations to inspect data
#   types and look for instances of missing or possibly errant data.

#loading the csv file and storing it a the variable
appointments_data = pd.read_csv(r"C:\DataSet\noshowappointments-kagglev2-may-2016.csv")

#File structure
appointments_data.head()


# <a id='wrangling'></a>
# > Printing file structure, line and columns number
# 

# In[35]:


appointments_data.shape


# 
# ### Data Cleaning 
# 
# 

# ### Removing Columns
# 
# >For analysis that will be performed some of the columns contained in the file will not be used. To facilitate the reading process the columns SMS_received, Neighbourhood and Handcap will be excluded.

# In[36]:


#List of columns to be deleted
#The columns SMS_received and Handcap will not be used
delete_collumns=['SMS_received', 'Neighbourhood', 'Handcap']

appointments_data = appointments_data.drop(delete_collumns, 1)

appointments_data.head()


# ### Changing data format
# 
# >Changing data format for the column ScheduledDay, AppointmentDay
# 

# In[37]:


#Changing the data format for the column ScheduledDay and AppointmentDay
appointments_data.ScheduledDay = pd.to_datetime(appointments_data['ScheduledDay'])
appointments_data.AppointmentDay = pd.to_datetime(appointments_data['AppointmentDay'])


# >Changing data format for the column PatientId
# 

# In[38]:


#Changing the format of the PatientID column 
change_columns= ['PatientId']

#Changing the data type in to column list 
appointments_data[change_columns]= appointments_data[change_columns].applymap(np.int64)


# >Printing the dataset format after type changes
# 

# In[39]:


appointments_data.dtypes


# ### Removing rows with zero values
# 
# >Removing records that have no value for the columns no-show, scheduledDay, AppointmentDay, PatientId', 'AppointmentID, Gender, Age
# 

# In[40]:


#List of columns to be evaluated
null_columns= ['No-show', 'ScheduledDay', 'AppointmentDay', 'PatientId', 'AppointmentID', 'Gender', 'Age']

#Replacing the value '0' of the column list with NAN
appointments_data[null_columns] = appointments_data[null_columns].replace(0, np.NAN)

#Removing all rows containing the NAN value in the column list defined in the null_columns list
appointments_data.dropna(subset= null_columns, inplace= True)

#Printing the number of rows and columns contained in the dataset after eliminating the rows with zero values
appointments_data.shape


# >Some rows have been deleted from the dataset as they have no value for the evaluated columns 
# 

# ### Removing Duplicate Rows (if any)
# 
# >Removing duplicate lines in data set, if any.
# 

# In[9]:


#Removing duplicate lines in data set
appointments_data.drop_duplicates(keep= 'first', inplace= True)

#Printing the number of rows and columns contained in the dataset after eliminating duplicate rows
appointments_data.shape


# ### Adding new columns
# 
# >Added two new columns DayOfWeek_ScheduledDay and DayOfWeek_AppointmentDay that contains the name of the day of the week associated with the ScheduledDay and AppointmentDay dates. DOWNumber_AppointmentDay contains the day number associated with the AppointmentDay
# 

# In[42]:


#Added the description of the day of the week associated with the ScheduledDay date
appointments_data['DayOfWeek_ScheduledDay'] = appointments_data['ScheduledDay'].dt.day_name()

#Added the number of the day of the week associated with the AppointmentDay date
appointments_data['DOWNumber_AppointmentDay'] = appointments_data['AppointmentDay'].dt.dayofweek
#Added the description of the day of the week associated with the AppointmentDay date
appointments_data['DayOfWeek_AppointmentDay'] = appointments_data['AppointmentDay'].dt.day_name()
appointments_data.head(5)


# >Added two new columns SDDate and ADDate that contain dates in YYY-MM-DD format, with no time indication
# 

# In[43]:


#Added two new columns based on the ScheduledDay and AppointmentDay columns. 
#The new columns will contain only the date, no time
appointments_data['SDDate'] = pd.to_datetime(appointments_data['ScheduledDay']).dt.date
appointments_data['ADDate'] = pd.to_datetime(appointments_data['AppointmentDay']).dt.date

appointments_data.head()


# >Added the NoShowValue column. The column will indicate whether the patient attended the consultation or not.
# The column will be based on the information contained in the column No-show, if the value is 1 it indicates that the patient did not attend the consultation. Created the ShowStatus column that contains the description of the patient's status in the consultation. This column will be used to generate the graphs.
# 

# In[44]:


#Added the NoShowValue column.
#The column will indicate whether the patient attended the consultation or not. 
#The column will be based on the information contained in the column No-show, if the value is 1, 
#it indicates that the patient did not attend the consultation.
appointments_data['NoShowValue'] = [1 if x =='Yes' else 0 for x in appointments_data['No-show']]
appointments_data['ShowStatus'] = ['No-Show' if x =='Yes' else 'Show Up' for x in appointments_data['No-show']]

#Printing dataset
appointments_data.head()


# >Adding columns containing the description based on the Hypertension, Diabetes, Alcoholism and Scholarship columns. These columns will be used for the generation of graphs

# In[45]:


appointments_data['HipertensionDesc'] = ['Hypertensive' if x == 1 else 'Non-hypertensive' for x in appointments_data['Hipertension']]
appointments_data['DiabeticDesc'] = ['Diabetic' if x == 1 else 'Non-Diabetic' for x in appointments_data['Diabetes']]
appointments_data['AlcoholicDesc'] = ['Alcoholic' if x == 1 else 'Non-Alcoholic' for x in appointments_data['Alcoholism']]
appointments_data['ScholarshipDesc'] = ['Enrolled' if x == 1 else 'Non-Enrolled' for x in appointments_data['Scholarship']]

#Printing dataset
appointments_data.head()


# ### Data Integrety
# 
# >Register line in the dataset represents a query. Thus, the values contained in the AppointmentID column cannot be repeated.

# In[48]:


appointmentID_count=len(appointments_data.AppointmentID.unique())
print('Total unique records:',appointmentID_count)


# >Checking the number of notes recorded according to the patient's gender

# In[49]:


appointments_data.Gender.value_counts()


# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# > **Tip**: Now that you've trimmed and cleaned your data, you're ready to move on to exploration. Compute statistics and create visualizations with the goal of addressing the research questions that you posed in the Introduction section. It is recommended that you be systematic with your approach. Look at one variable at a time, and then follow it up by looking at relationships between variables.
# 
# 

# ## Question 1: Total Appointments vs Gender

# <a id='eda'></a>
# >The graph below shows the distribution of the total number of notes by gender of the patient.
# 
# 

# In[50]:


#Labels that will be displayed on the chart
labels = ['Female', 'Male']
#Returns the number of notes by patient's gender
sizes = appointments_data.Gender.value_counts()
explode = (0, 0)

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# >The function below will be used to return the number of summarized records according to the columns informed in the parameters

# In[51]:


#Returns the number of records to be separated by the given columns 
def countField(column1, column2):

    count = appointments_data.groupby(column1)[column2].value_counts()
    
    return count


# ## Question 2: Attendance and Non-Attendance vs Gender

# >The function will return the number of patients, separated by gender, who attended the appointments or not

# In[52]:


#Using the function to return the number of patients, separated by gender, who attended the appointments or not
countNoShow = countField('No-show','Gender')
countNoShow.head()


# In[54]:


figure = plt.figure(figsize=(18,8))
ax = figure.add_subplot(1,2,1)
ax.set_title("Attendance and Non-Attendance vs Gender")
sns.countplot(x="Gender",hue="ShowStatus",data=appointments_data,ax=ax)
ax.set_xlabel("Gender")
ax.set_ylabel("Schedule Quantity")


plt.tight_layout
plt.show()


# >Graphical representation of the amount of attendance and no-attendance separated by patient's gender
# >
# 

# ## Question 3: Patients who did not attend and are considered to be hypertensive

# In[57]:


#Using the function to return the number of patients, considered hypertensive, who attended or not attended the consultations
countHipertension = countField('No-show','Hipertension')
countHipertension.head()


# In[59]:


figure = plt.figure(figsize=(18,8))
ax = figure.add_subplot(1,2,1)
ax.set_title("Attendance and Non-Attendance vs Hypertensive")
sns.countplot(x="HipertensionDesc",hue="ShowStatus",data=appointments_data,ax=ax)
ax.set_xlabel("Condition")
ax.set_ylabel("Schedule Quantity")


plt.tight_layout
plt.show()


# >Graphical representation of the amount of attendance and non-attendance of patients with indication of hypertension
# >
# 

# ## Question 4: Patients who did not attend and are considered diabetic

# In[22]:


#Using the function to return the number of patients, considered diabetic, who attended or not attended the consultations
countDiabetic = countField('No-show','Diabetes')
countDiabetic.head()


# >Based on the entire data set, the No-Show = Yes non-attendance rate for patients considered to be diabetic is quite low, around 1400 patients. More than 6.000 diabetic patients regularly attended the scheduled appointment.
# 

# In[60]:


figure = plt.figure(figsize=(18,8))
ax = figure.add_subplot(1,2,1)
ax.set_title("Attendance and Non-Attendance vs Diabetic")
sns.countplot(x="DiabeticDesc",hue="ShowStatus",data=appointments_data,ax=ax)
ax.set_xlabel("Condition")
ax.set_ylabel("Schedule Quantity")


plt.tight_layout
plt.show()


# >Graphical representation of the amount of attendance and non-attendance of patients with indication of diabetic
# >
# 

# ## Question 5: Patients who did not attend and are considered alcoholics

# In[61]:


#Using the function to return the number of patients, considered alcoholics, who attended or not attended the consultations
countAlcoholism = countField('No-show','Alcoholism')
countAlcoholism.head()


# >Based on the entire data set, the No-Show = Yes non-attendance rate for patients considered to be alcoholics is quite low, around 600 patients. More than 2.000 alcoholics patients regularly attended the scheduled appointment.
# 

# In[62]:


figure = plt.figure(figsize=(18,8))
ax = figure.add_subplot(1,2,1)
ax.set_title("Attendance and Non-Attendance vs Alcoholism")
sns.countplot(x="AlcoholicDesc",hue="ShowStatus",data=appointments_data,ax=ax)
ax.set_xlabel("Condition")
ax.set_ylabel("Schedule Quantity")


plt.tight_layout
plt.show()


# >Graphical representation of the amount of attendance and non-attendance of patients with indication of alcoholism
# >
# 

# ## Question 6: Patients who did not attend and are enrolled in social programs

# In[26]:


#Using the function to return the number of patients, considered alcoholics, who attended or not attended the consultations
countScholarship = countField('No-show','Scholarship')
countScholarship.head()


# >Based on the entire data set, the non-attendance rate of No-Attendance = Yes for patients enrolled in government social programs is low, at around 2500 patients. Most patients, more than 99 thousand patients, do not belong to the social program
# 

# In[27]:


figure = plt.figure(figsize=(18,8))
ax = figure.add_subplot(1,2,1)
ax.set_title("Attendance and Non-Attendance vs Social Programs")
sns.countplot(x="ScholarshipDesc",hue="ShowStatus",data=appointments_data,ax=ax)
ax.set_xlabel("Condition")
ax.set_ylabel("Schedule Quantity")


plt.tight_layout
plt.show()


# >Graphical representation of the number of visits and non-visits of patients with an indication of participation in a social program
# >
# 
# 
# 

# ## Question 7: Date with the highest number of non-attendance to consultations

# In[28]:


groupDay = appointments_data.groupby('ADDate')['NoShowValue'].sum()

groupDay.sort_values(ascending = True, inplace = True)

#ploting
lt = groupDay.plot.barh(color = '#0000ff', fontsize = 13)

#Graph Title
lt.set(title = 'Non-Attendence vs Date')

#Graph x-axis
lt.set_xlabel('Amount of Non-Attendence', color = 'black', fontsize = '13')
lt.set_ylabel('Appointment Day', color = 'black', fontsize = '13')

#figure size(width, height)
lt.figure.set_size_inches(12, 9)

#Displaying the plot
plt.show()


# >The graph shows in decreasing form the days that had the highest amount of non-attendance at consultations. May 16, 2016 was the date with the highest number of no-shows, exceeding 1000 abstentions
# 

# ## Question 8: Day of Week with the highest number of non-attendance to consultations

# In[32]:


figure = plt.figure(figsize=(18,8))
ax = figure.add_subplot(1,2,1)
ax.set_title("Days of the Week Attendance vs Non-attendance")
sns.countplot(x="DayOfWeek_AppointmentDay",hue="ShowStatus",data=appointments_data,ax=ax)
ax.set_xlabel("Days of Weekday")
ax.set_ylabel("Schedule Quantity")


plt.tight_layout
plt.show()


# >The graph shows the days of the week and their quantities of non-attendance at the consultation. According to the chart, Tuesday is the day with the highest non-attendance rate.

# <a id='conclusions'></a>
# ## Conclusions
# 
# >After removing records with missing and duplicate information from the dataset, 106988 records were available for analysis.
# Each record contained in the dataset represents a medical consultation. Making the distribution by gender of the patient, 65.5% of the consultations are associated with female patients and 34.5% with male patients.
# >
# >Based on the available records, it can be observed that, in addition to the greater number of consultations being associated with female patients, the higher rate of abstention is also associated with female patients. Representing almost twice the number of abstentions when compared to male patients.
# >
# >Patients who have some kind of previously identified condition have a relatively low rate of abstention. As noted in the graphics for indication of hypertensive, diabetic and alcoholic patients. Indicating that these patients seek to constantly monitor their condition to avoid future problems.
# >
# >Analyzing the patients who have the indication of participation in social programs, it is observed that the abstention rate is also low.
# >
# >According to the dataset, May 16, 2016 was the day with the highest number of abstentions, that is, patients do not attend their appointments.
# Analyzing the number of attendances and no-shows per day of the week, it is observed that the greatest number of abstentions occurred on Tuesday.

# <a id='conclusions'></a>
# ## Limitations 
# 
# >The dataset provided could contain some data that could facilitate the creation of new analyzes:
# >
# >The AppointmentDay column, which contains the date on which the consultation took place, does not have the time of the consultation. If this column contained the time of the consultation, it would be possible to identify the times when the greatest number of no-shows occur. This information could even be crossed with external data, such as traffic information. This information could also be used to determine a pattern by appointment times according to the patient's age.
# >
# >Another point, instead of containing the patient's age, would be better if the patient's date of birth was provided. This information would also allow for complementary analyzes.
