import pandas as pd

def save_dict_to_csv(data: dict, filename: str):
    try:
        df = pd.DataFrame(data)
        df = df.fillna(0)
        df.to_csv(filename, index=False)
    except Exception as e:
        print(f"An error occurred: {str(e)}")

data = [
    {"Uname": "Mubaris", "Study": 1, "Exam": 1, "Motivation": 1},
    {"Uname": "Albin",  "Study": 1, "Exam": 1, "Motivation": 1},
    {"Uname": "Jen",  "Study": 1, "Exam": 1, "Motivation": 1},
    {"Uname": "Niketh",  "Study": 1, "Exam": 1, "Motivation": 1}
]

save_dict_to_csv(data, "output.csv")
print("Data saved to output.csv")