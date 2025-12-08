import pickle, json

with open("models/mos_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

data = {
    "mean": scaler.mean_.tolist(),
    "scale": scaler.scale_.tolist(),
}

with open("models/mos_scaler.json", "w") as f:
    json.dump(data, f, indent=2)
print("Saved MOS scaler → mos_scaler.json")

with open("models/impairment_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

data = {
    "mean": scaler.mean_.tolist(),
    "scale": scaler.scale_.tolist(),
}

with open("models/impairment_scaler.json", "w") as f:
    json.dump(data, f, indent=2)

print("Saved Impairment scaler → impairment_scaler.json")

