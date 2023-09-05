import pandas as pd
import matplotlib.pyplot as plt

# Step 2: Read data from the CSV file into a Pandas DataFrame
df = pd.read_csv('soccer_players.csv')

# Step 3: Find the top 5 players with the highest number of goals scored
top_goals_players = df.nlargest(5, 'Goals')

# Step 4: Find the top 5 players with the highest salaries
top_salary_players = df.nlargest(5, 'Salary')

# Step 5: Calculate the average age of players
average_age = df['Age'].mean()

# Filter players above the average age
above_average_age_players = df[df['Age'] > average_age]

# Step 6: Visualize the distribution of players based on their positions using a bar chart
position_counts = df['Position'].value_counts()

# Create a bar chart
position_counts.plot(kind='bar', xlabel='Position', ylabel='Number of Players')
plt.title('Distribution of Players by Position')
plt.show()

# Print the results
print("Top 5 Players with Highest Goals:")
print(top_goals_players[['Name', 'Goals']])

print("\nTop 5 Players with Highest Salaries:")
print(top_salary_players[['Name', 'Salary']])

print(f"\nAverage Age of Players: {average_age:.2f}")

print("\nPlayers Above Average Age:")
print(above_average_age_players[['Name', 'Age']])
