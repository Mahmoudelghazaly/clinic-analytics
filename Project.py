import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

appointments = pd.read_csv('Appointments.csv')
doctors = pd.read_csv('doctors.csv')
expenses = pd.read_csv('expenses.csv')
patients = pd.read_csv('patients.csv')
revenues = pd.read_csv('revenues.csv')

dfs = [appointments, doctors, expenses, patients, revenues]
#بيعمل لوب بيمشي على اسم الجدول و البيانات في نفس الوقت
#بدل ما اعمل اكتر من لوب منفصل
for name, d in zip(['Appointments', 'Doctors', 'Expenses', 'Patients', 'Revenues'], dfs):
    d.dropna(how='all', inplace=True)
    d.drop_duplicates(inplace=True)

    print(f"--- {name} ---")
    print(d.isnull().sum())
    print("Duplicates:", d.duplicated().sum())
    print()
#علشان اتاكد انه بيقرأها ك تاريخ 
appointments['Date'] = pd.to_datetime(appointments['Date'])
revenues['Date'] = pd.to_datetime(revenues['Date'])
#علشان اتأكد انهم متشابهين من غير اي اختلافات
appointments['Specialty'] = appointments['Specialty'].str.strip().str.title()
doctors['Specialty'] = doctors['Specialty'].str.strip().str.title()
patients['City'] = patients['City'].str.strip().str.title()
#دلوقتي عاوز اربطهم مع بعض
merged_df = pd.merge(appointments, doctors, on='DoctorID', how='left')
merged_df = pd.merge(merged_df, patients, on='PatientID', how='left')
#دلوقتي هجمع الشهور و السنين و الايام في عمود جديد
expenses['Date'] = pd.to_datetime(expenses[['Year', 'Month', 'Day']])
merged_df = pd.merge(merged_df, revenues, on='Date', how='left')
merged_df = pd.merge(merged_df, expenses, on='Date', how='left')
# حساب عدد المواعيد التي تم حضورها
attended_count = appointments[appointments['Status'].str.lower() == 'Show'].shape[0]
print(appointments['Status'].value_counts())

#عاوز اعرف عدد زيارات لكل تخصص
visits_per_specialty = appointments['Specialty'].value_counts()
print(visits_per_specialty)
#عاوز اعرف عدد زيارات لكل طبيب
visits_per_doctor = merged_df.groupby(['DoctorID', 'DoctorName']).size().reset_index(name='VisitCount')
print(visits_per_doctor)
#عاوز اعرف عدد زيارات لكل مريض
visits_per_patient = merged_df.groupby(['PatientID', 'Name']).size().reset_index(name='VisitCount')
print(visits_per_patient)
#عدد المرضي لكل مرض
visits_per_type = expenses['Type'].value_counts()
print(visits_per_type)
#اكتر شهر فيه مرضي
visits_per_month = merged_df['Month'].value_counts()
print(visits_per_month)
#عدد كل فئه عمريه من مرضي المكان
bins = [0, 20, 40, 60, 80, 100]
labels = ['<20', '20-40', '40-60', '60-80', '80+']
merged_df['AgeGroup'] = pd.cut(merged_df['Age'], bins=bins, labels=labels, right=False)
age_group_counts = merged_df['AgeGroup'].value_counts()
print(age_group_counts)
#متوسط الإيراد لكل تخصص
avg_rev_specialty = merged_df.groupby('Specialty')['Revenue'].mean().sort_values(ascending=False)
print(avg_rev_specialty)
#متوسط الإيراد لكل شهر
merged_df['Month'] = merged_df['Date'].dt.month
avg_rev_month = merged_df.groupby('Month')['Revenue'].mean().sort_values(ascending=False)
print(avg_rev_month)
#عدد الزيارات لكل تخصص
plt.figure(figsize=(8,5))
sns.barplot(x=visits_per_specialty.index, y=visits_per_specialty.values, palette="viridis")
plt.xticks(rotation=45)
plt.title("Visits per Specialty")
plt.xlabel("Specialty")
plt.ylabel("Number of Visits")
plt.show()
#عدد الزيارات لكل شهر
plt.figure(figsize=(8,5))
sns.barplot(x=visits_per_month.index, y=visits_per_month.values, palette="viridis")
plt.xticks(rotation=45)
plt.title("Visits per Month")
plt.xlabel("Month")
plt.ylabel("Number of Visits")
plt.show()
#عدد الزيارات لكل عمر
plt.figure(figsize=(8,5))
sns.barplot(x=age_group_counts.index, y=age_group_counts.values, palette="viridis")
plt.xticks(rotation=45)
plt.title("Visits per Age Group")
plt.xlabel("Age Group")
plt.ylabel("Number of Visits")
plt.show()
#توزيع الفئات العمرية للمرضى
top_doctors = visits_per_doctor.sort_values(by="VisitCount", ascending=False).head(10)
plt.figure(figsize=(8,5))
sns.barplot(x="VisitCount", y="DoctorName", data=top_doctors, palette="mako")
plt.title("Top 10 Doctors by Visits")
plt.xlabel("Number of Visits")
plt.ylabel("Doctor Name")
plt.show()
#متوسط الإيراد لكل شهر
plt.figure(figsize=(8,5))
sns.barplot(x=avg_rev_month.index, y=avg_rev_month.values, palette="viridis")
plt.xticks(rotation=45)
plt.title("Average Revenue per Month")
plt.xlabel("Month")
plt.ylabel("Average Revenue")
plt.show()
#متوسط الإيراد لكل تخصص
plt.figure(figsize=(8,5))
sns.barplot(x=avg_rev_specialty.index, y=avg_rev_specialty.values, palette="viridis")
plt.xticks(rotation=45)
plt.title("Average Revenue per Specialty")
plt.xlabel("Specialty")
plt.ylabel("Average Revenue")
plt.show()
# اكثر المدن من حيث عدد المرضى
patients_per_city = merged_df['City'].value_counts()
plt.figure(figsize=(8,5))
sns.barplot(x=patients_per_city.index, y=patients_per_city.values, palette="viridis")
plt.xticks(rotation=45)
plt.title("Patients per City")
plt.xlabel("City")
plt.ylabel("Number of Patients")
plt.show()
#الترابط بين العمر والإيراد
#corr() بيحسب معامل الارتباط بين العمودين
correlation = merged_df[['Age', 'Revenue']].corr()
sns.scatterplot(x='Age', y='Revenue', data=merged_df, hue='Specialty')
#هنعمل خط زمني لعدد الزيارات والإيرادات الشهرية.
visits_over_time = merged_df.groupby('Date').size()
revenue_over_time = merged_df.groupby('Date')['Revenue'].sum()
plt.figure(figsize=(12, 6))
plt.plot(visits_over_time.index, visits_over_time.values, label='Visits')
plt.plot(revenue_over_time.index, revenue_over_time.values, label='Revenue')
plt.title('Visits and Revenue Over Time')
plt.xlabel('Date')
plt.ylabel('Count')
plt.legend()
plt.show()
# نتنبأ بالإيرادات المستقبلية بناءً على البيانات الزمنية
daily_revenue = merged_df.groupby('Date')['Revenue'].sum().reset_index()
daily_revenue['Days'] = (daily_revenue['Date'] - daily_revenue['Date'].min()).dt.days
X = daily_revenue[['Days']]
y = daily_revenue['Revenue']
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
future_days = np.arange(daily_revenue['Days'].max() + 1, daily_revenue['Days'].max() + 31).reshape(-1, 1)
future_dates = pd.date_range(start=daily_revenue['Date'].max() + pd.Timedelta(days=1), periods=30)
future_preds = model.predict(future_days)
plt.figure(figsize=(12,6))
plt.plot(daily_revenue['Date'], y, label='Actual Revenue', marker='o')
plt.plot(daily_revenue['Date'], y_pred, label='Predicted (Train)', linestyle='--')
plt.plot(future_dates, future_preds, label='Forecast (Future)', linestyle='dotted')
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.title('Revenue Forecasting using Linear Regression')
plt.legend()
plt.show()
#تقسيم المرضى لمجموعات
visits_per_patient = merged_df.groupby(['PatientID', 'Name', 'City', 'Age']).size().reset_index(name='VisitCount')

le = LabelEncoder()
visits_per_patient['CityEncoded'] = le.fit_transform(visits_per_patient['City'])

features = visits_per_patient[['Age', 'CityEncoded', 'VisitCount']]

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=4, random_state=42)
visits_per_patient['Cluster'] = kmeans.fit_predict(features_scaled)

plt.figure(figsize=(8,6))
sns.scatterplot(
    x=visits_per_patient['Age'],
    y=visits_per_patient['VisitCount'],
    hue=visits_per_patient['Cluster'],
    palette='Set2',
    s=100
)
plt.title('Patient Clusters (Age vs VisitCount)')
plt.xlabel('Age')
plt.ylabel('Visit Count')
plt.legend(title='Cluster')
plt.show()

print(visits_per_patient.head(10))
