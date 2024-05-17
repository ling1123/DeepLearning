import pickle
f = open("D:/EA/SIMS/Processed/unaligned_39.pkl", "rb")
content = pickle.load(f)
print(content)
print(content.keys())
print(content["train"].values())