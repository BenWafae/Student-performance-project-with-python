import pickle

from analyse_model import model

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)