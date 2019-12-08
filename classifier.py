import copy

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_fscore_support as score
import matplotlib.pyplot as plt

from PhemeDataset import PhemeDatasetES

dataset = PhemeDatasetES(hosts="localhost:9200", index_name="twitter")

representations = dataset.get_source_tweet_representations("charliehebdo")
representations.extend(dataset.get_source_tweet_representations("germanwings-crash"))
representations.extend(dataset.get_source_tweet_representations("sydneysiege"))
representations.extend(dataset.get_source_tweet_representations("ottawashooting"))
test_set = dataset.get_source_tweet_representations("ferguson")
X = []
y = []
for representation in representations:
    x = [representation['time_span'],  # 0
         representation['early_reaction_count'],  # 1
         representation['mid_reaction_count'],  # 2
         representation['all_reaction_count'],  # 3
         representation['is_sensitive'],  # 4
         # User
         representation['is_geo_enabled'],  # 5
         representation['has_description'],  # 6
         representation['description_word_count'],  # 7
         representation['role_score'],  # 8
         representation['user_follower_count'],  # 9
         representation['is_verified'],  # 10
         representation['favorites_count'],  # 11
         # Tweet
         representation['has_question_mark'],  # 12
         representation['question_mark_count'],  # 13
         representation['has_exclamation_mark'],  # 14
         representation['exclamation_mark_count'],  # 15
         representation['has_dotdotdot_mark'],  # 16
         representation['dotdotdot_mark_count']  # 17

         ]
    X.append(x.copy())
    y.append(representation['isRumor'])
X_test = []
y_test = []
for representation in test_set:
    x = [representation['time_span'],
         representation['early_reaction_count'],
         representation['mid_reaction_count'],
         representation['all_reaction_count'],
         representation['is_sensitive'],
         # User
         representation['is_geo_enabled'],
         representation['has_description'],
         representation['description_word_count'],
         representation['role_score'],
         representation['user_follower_count'],
         representation['is_verified'],
         representation['favorites_count'],
         # Tweet
         representation['has_question_mark'],
         representation['question_mark_count'],
         representation['has_exclamation_mark'],
         representation['exclamation_mark_count'],
         representation['has_dotdotdot_mark'],
         representation['dotdotdot_mark_count']

         ]
    X_test.append(x.copy())
    y_test.append(representation['isRumor'])

# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

X_train = X
y_train = y


def classify_with_rfc(train_data, train_data_y, test_data, test_data_y):
    cls = RandomForestClassifier()
    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    cls.fit(train_data, train_data_y)
    accuracy = cls.score(test_data, test_data_y)
    print("Accuracy: " + str(accuracy))
    return accuracy, cls


def classify_with_linear_regression(train_data, train_data_y, test_data, test_data_y):
    lr = LinearRegression()
    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    lr.fit(train_data, train_data_y)
    accuracy = lr.score(test_data, test_data_y)
    print("Accuracy: " + str(accuracy))
    return accuracy, lr


def classify_with_svm(train_data, train_data_y, test_data, test_data_y):
    svc = SVC()
    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    svc.fit(train_data, train_data_y)
    accuracy = svc.score(test_data, test_data_y)
    print("Accuracy: " + str(accuracy))
    return accuracy, svc


def classify_with_dt(train_data, train_data_y, test_data, test_data_y):
    svc = DecisionTreeClassifier(random_state=0)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    svc.fit(train_data, train_data_y)
    accuracy = svc.score(test_data, test_data_y)
    print("Accuracy: " + str(accuracy))
    return accuracy, svc


precisions_rf = list()
precisions_svm = list()
precisions_dt = list()

recall_rf = list()
recall_svm = list()
recall_dt = list()


def evaluate_model(model, X_test, y_test, type):
    predicted_y = model.predict(X_test)
    report = classification_report(y_test, predicted_y)
    matrix = confusion_matrix(y_test, predicted_y)
    precision, recall, fscore, support = score(y_test, predicted_y, average='macro')
    if type == "rf":
        precisions_rf.append(precision)
        recall_rf.append(recall)
    elif type == "svm":
        precisions_svm.append(precision)
        recall_svm.append(recall)
    else:
        precisions_dt.append(precision)
        recall_dt.append(recall)
    print(report)
    print(matrix)


for i in range(3, 17):
    print("For the K-Selected feature value: " + str(i))
    selector = SelectKBest(chi2, k=i)
    X_train_new = selector.fit_transform(X.copy(), y.copy()).tolist()
    selected_features = selector.get_support(indices=True)
    print("Selected Feature Indices: " + str(selected_features))
    X_test_new = copy.deepcopy(X_test)
    for data_num in range(len(X_test_new)):
        X_test_new[data_num] = [i for j, i in enumerate(X_test_new[data_num]) if j in selected_features]

    print("Random Forrest: ")
    _, model = classify_with_rfc(X_train_new, y, X_test_new, y_test)
    evaluate_model(model, X_test_new, y_test, "rf")

    print("SVM: ")
    _, model = classify_with_svm(X_train_new, y, X_test_new, y_test)
    evaluate_model(model, X_test_new, y_test, "svm")

    print("Decision Tree: ")
    _, model = classify_with_dt(X_train_new, y, X_test_new, y_test)
    evaluate_model(model, X_test_new, y_test, "dt")
    print("----------------------------------------------------")

plt.ylabel('Precision for RF')
plt.xlabel('K-Selected Features')
plt.plot(range(3, 17), precisions_rf, label='Precision', marker='o')
plt.plot(range(3, 17), recall_rf, label='Recall', marker='x')
plt.legend()
plt.show()

plt.ylabel('Precision for SVM')
plt.xlabel('K-Selected Features')
plt.plot(range(3, 17), precisions_svm, label='Precision', marker='o')
plt.plot(range(3, 17), recall_svm, label='Recall', marker='x')
plt.legend()
plt.show()

plt.ylabel('Precision for DT')
plt.xlabel('K-Selected Features')
plt.plot(range(3, 17), precisions_dt, label='Precision', marker='o')
plt.plot(range(3, 17), recall_dt, label='Recall', marker='x')
plt.legend()
plt.show()
