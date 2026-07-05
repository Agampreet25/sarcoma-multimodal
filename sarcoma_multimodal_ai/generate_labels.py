import pandas as pd

# Load clinical metadata
df = pd.read_excel("INFOclinical_STS.xlsx")

labels = {}

for _, row in df.iterrows():
    pid = row["Patient ID"]
    outcome = str(row["Outcome (recurrence, mets)"]).lower()

    # Lung metastasis detection
    if "lung" in outcome:
        labels[pid] = 1
    else:
        labels[pid] = 0

# Save labels.csv
labels_df = pd.DataFrame(
    labels.items(),
    columns=["patient_id", "label"]
)

labels_df.to_csv("labels.csv", index=False)

print("labels.csv generated")
print(labels_df["label"].value_counts())
