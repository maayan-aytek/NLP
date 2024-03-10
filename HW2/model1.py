from data_load import data_loader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
import torch


TRAIN_PATH = "data/train.tagged"
TEST_PATH = "data/dev.tagged"
EMBEDDING_PATH = ['word2vec-google-news-300']

X_train, y_train = data_loader(TRAIN_PATH, EMBEDDING_PATH)
print("finished load train")
X_test, y_test = data_loader(TEST_PATH, EMBEDDING_PATH)
print("finished load test")

# Convert all lists to tensors.
X_train_tensor = torch.stack(X_train)
y_train_tensor = torch.tensor(y_train)
X_test_tensor = torch.stack(X_test)
y_test_tensor = torch.tensor(y_test)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train_tensor, y_train_tensor)
print("finished training")

y_test_pred = knn.predict(X_test_tensor)
print(f"\nDev F1 Score: {f1_score(y_test_tensor, y_test_pred)}\n")