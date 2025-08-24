import pandas as pd

def save_dict_to_csv(data: dict, filename: str):
    try:
        df = pd.DataFrame(data)
        df = df.fillna(0)
        df.to_csv(filename, index=False)
    except Exception as e:
        print(f"An error occurred: {str(e)}")

data = [
    {"Name": "Alice", "Age": 25, "City": "New York"},
    {"Name": "Bob", "Age": None, "City": "London"},
    {"Name": None, "Age": 22, "City": None}
]

save_dict_to_csv(data, "output.csv")
print("Data saved to output.csv")