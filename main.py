import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import random
import os

# Get the path of tasks.csv in the same folder as this script
file_path = os.path.join(os.path.dirname(__file__), "tasks.csv")

# Load tasks from CSV
try:
    tasks = pd.read_csv(file_path)
except FileNotFoundError:
    tasks = pd.DataFrame(columns=['description', 'priority'])

# Function to save tasks
def save_tasks():
    tasks.to_csv(file_path, index=False)

# Train the ML model only if tasks exist
vectorizer = CountVectorizer()
clf = MultinomialNB()
model = make_pipeline(vectorizer, clf)

if not tasks.empty:
    model.fit(tasks['description'].astype(str), tasks['priority'].astype(str))

# Function to add a task
def add_task(description, priority):
    global tasks

    new_task = pd.DataFrame({
        'description': [description],
        'priority': [priority]
    })

    tasks = pd.concat([tasks, new_task], ignore_index=True)
    save_tasks()

    # Retrain model
    if not tasks.empty:
        model.fit(tasks['description'].astype(str), tasks['priority'].astype(str))

# Function to remove a task
def remove_task(description):
    global tasks

    tasks = tasks[tasks['description'] != description]
    save_tasks()

    # Retrain model
    if not tasks.empty:
        model.fit(tasks['description'].astype(str), tasks['priority'].astype(str))

# Function to list tasks
def list_tasks():
    if tasks.empty:
        print("No tasks available.")
    else:
        print("\nCurrent Tasks:")
        print(tasks)

# Function to recommend a task
def recommend_task():
    if tasks.empty:
        print("No tasks available for recommendations.")
        return

    high_priority_tasks = tasks[tasks['priority'].str.lower() == 'high']

    if not high_priority_tasks.empty:
        random_task = random.choice(high_priority_tasks['description'].tolist())
        print(f"\nRecommended Task: {random_task} (Priority: High)")
    else:
        print("No high-priority tasks available.")

# Main menu
while True:
    print("\nTask Management App")
    print("1. Add Task")
    print("2. Remove Task")
    print("3. List Tasks")
    print("4. Recommend Task")
    print("5. Exit")

    choice = input("Select an option: ")

    if choice == "1":
        description = input("Enter task description: ")
        priority = input("Enter task priority (Low/Medium/High): ").capitalize()

        if priority not in ["Low", "Medium", "High"]:
            print("Invalid priority! Use Low, Medium, or High.")
            continue

        add_task(description, priority)
        print("Task added successfully.")

    elif choice == "2":
        description = input("Enter task description to remove: ")
        remove_task(description)
        print("Task removed successfully.")

    elif choice == "3":
        list_tasks()

    elif choice == "4":
        recommend_task()

    elif choice == "5":
        print("Goodbye!")
        break

    else:
        print("Invalid option. Please select a valid option.")