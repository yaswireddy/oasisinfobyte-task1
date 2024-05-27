import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('car data.csv')


cars = data[~data['Car_Name'].str.contains('Royal Enfield|UM Renegade|KTM|Bajaj|Hyosung|Hero|TVS|Honda|Yamaha|Suzuki')]
bikes = data[data['Car_Name'].str.contains('Royal Enfield|UM Renegade|KTM|Bajaj|Hyosung|Hero|TVS|Honda|Yamaha|Suzuki')]


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(cars['Selling_Price'], kde=True)
plt.title('Distribution of Car Selling Prices')

plt.subplot(1, 2, 2)
sns.histplot(bikes['Selling_Price'], kde=True)
plt.title('Distribution of Bike Selling Prices')
plt.show()


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(data=cars, x='Year', y='Selling_Price')
plt.title('Car Selling Price vs. Year')

plt.subplot(1, 2, 2)
sns.scatterplot(data=bikes, x='Year', y='Selling_Price')
plt.title('Bike Selling Price vs. Year')
plt.show()


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(data=cars, x='Fuel_Type', y='Selling_Price')
plt.title('Car Selling Price by Fuel Type')

plt.subplot(1, 2, 2)
sns.boxplot(data=bikes, x='Fuel_Type', y='Selling_Price')
plt.title('Bike Selling Price by Fuel Type')
plt.show()


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(data=cars, x='Transmission', y='Selling_Price')
plt.title('Car Selling Price by Transmission Type')

plt.subplot(1, 2, 2)
sns.boxplot(data=bikes, x='Transmission', y='Selling_Price')
plt.title('Bike Selling Price by Transmission Type')
plt.show()


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(data=cars, x='Owner', y='Selling_Price')
plt.title('Car Selling Price by Number of Owners')

plt.subplot(1, 2, 2)
sns.boxplot(data=bikes, x='Owner', y='Selling_Price')
plt.title('Bike Selling Price by Number of Owners')
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(cars[['Selling_Price', 'Present_Price', 'Driven_kms', 'Year']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap for Cars')
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(bikes[['Selling_Price', 'Present_Price', 'Driven_kms', 'Year']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap for Bikes')
plt.show()
