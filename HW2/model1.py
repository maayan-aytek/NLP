from data_load import data_loader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score


TRAIN_PATH = "data/train.tagged"
TEST_PATH = "data/dev.tagged"
EMBEDDING_PATH = 'word2vec-google-news-300'

X_train, y_train = data_loader(TRAIN_PATH, EMBEDDING_PATH)
print("finished load train")
X_test, y_test = data_loader(TEST_PATH, EMBEDDING_PATH)
print("finished load test")

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
print("finished training")

y_test_pred = knn.predict(X_test)
print(f1_score(y_test, y_test_pred))