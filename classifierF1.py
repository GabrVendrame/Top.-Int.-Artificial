import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Carrega o conjunto de dados
dataset = np.load("features.npy", allow_pickle=True)

# Listas para armazenar as características e labels
X = []
y = []

# Loop para extração de características e labels
for data in dataset:
    X.append(data[0])
    y.append(data[1])

# Listas para armazenar os resultados das métricas de avaliação
result_knn = []
result_svm = []
result_rf = []
result_knnrf = []
result_svmrf = []

# Listas para armazenar os valores preditos e reais para cada algoritmo de classificação
predicted_knn = np.array([])
predicted_svm = np.array([])
predicted_rf = np.array([])
predicted_knnrf = np.array([])
predicted_svmrf = np.array([])

actual_knn = np.array([])
actual_svm = np.array([])
actual_rf = np.array([])
actual_knnrf = np.array([])
actual_svmrf = np.array([])

metrics_knn = np.zeros((5, 4))
metrics_svm = np.zeros((5, 4))
metrics_rf = np.zeros((5, 4))
metrics_knnrf = np.zeros((5, 4))
metrics_svmrf = np.zeros((5, 4))

count_iter = 1
# Estratifica a base de dados em 5 partes
skf = StratifiedKFold(n_splits=5)
for train_index, test_index in skf.split(X,y):
    X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
    y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]  
    print(f"Iteração {count_iter}")

    # Treina o classificador KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    pred_knn = knn.predict(X_test)

    # Calcula as métricas de avaliação do classificador KNN
    accuracy_knn = knn.score(X_test, y_test)
    precision_knn = precision_score(y_test, pred_knn, average='weighted')
    recall_knn = recall_score(y_test, pred_knn, average='weighted')
    f1score_knn = f1_score(y_test, pred_knn, average='weighted')
    print(f"KNN:\nAccuracy: {accuracy_knn}, Precision: {precision_knn}, Recall: {recall_knn}, F1-Score: {f1score_knn}\n")
    result_knn.append([accuracy_knn, precision_knn, recall_knn, f1score_knn])

    # Matriz de confusão do KNN
    predicted_knn = np.append(predicted_knn, pred_knn)
    actual_knn = np.append(actual_knn, y_test)

    metrics_knn[count_iter-1] = [accuracy_knn, precision_knn, recall_knn, f1score_knn]

    # Treina o classificador SVM
    svm = SVC(kernel='poly', C= 100 , gamma=1)
    svm.fit(X_train, y_train)
    
    pred_svm = svm.predict(X_test)  

    # Calcula as métricas de avaliação do classificador SVM
    accuracy_svm = svm.score(X_test, y_test)
    precision_svm = precision_score(y_test, pred_svm, average='weighted')
    recall_svm = recall_score(y_test, pred_svm, average='weighted')
    f1score_svm = f1_score(y_test, pred_svm, average='weighted')
    print(f"SVM:\nAccuracy: {accuracy_svm}, Precision: {precision_svm}, Recall: {recall_svm}, F1-Score: {f1score_svm}\n")
    result_svm.append([accuracy_svm, precision_svm, recall_svm, f1score_svm])

    # Matriz de confusão do SVM
    predicted_svm = np.append(predicted_svm, pred_svm)
    actual_svm = np.append(actual_svm, y_test)   

    metrics_svm[count_iter-1] = [accuracy_svm, precision_svm, recall_svm, f1score_svm]
    
    # Treina o classificador RandomForest
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)

    pred_rf = rf.predict(X_test)

    # Calcula as métricas de avaliação do classificador RandomForest
    accuracy_rf = rf.score(X_test, y_test)
    precision_rf = precision_score(y_test, pred_rf, average='weighted')
    recall_rf = recall_score(y_test, pred_rf, average='weighted')
    f1score_rf = f1_score(y_test, pred_rf, average='weighted')
    print(f"Random Forest:\nAccuracy: {accuracy_rf}, Precision: {precision_rf}, Recall: {recall_rf}, F1-Score: {f1score_rf}\n")
    result_rf.append([accuracy_rf, precision_rf, recall_rf, f1score_rf])

    # Matriz de confusão do RandomForest
    predicted_rf = np.append(predicted_rf, pred_rf)
    actual_rf = np.append(actual_rf, y_test)

    metrics_rf[count_iter-1] = [accuracy_rf, precision_rf, recall_rf, f1score_rf]

    # Treina o ensemble entre KNN e RandomForest
    knnrf = VotingClassifier(estimators=[('knn', knn), ('rf', rf)], voting='hard')
    knnrf.fit(X_train, y_train)

    pred_knnrf = knnrf.predict(X_test)

    # Calcula as métricas de avaliação para o ensemble KNN e RandomForest
    accuracy_knnrf = knnrf.score(X_test, y_test)
    precision_knnrf = precision_score(y_test, pred_knnrf, average='weighted')
    recall_knnrf = recall_score(y_test, pred_knnrf, average='weighted')
    f1score_knnrf = f1_score(y_test, pred_knnrf, average='weighted')
    print(f"KNN+RF:\nAccuracy: {accuracy_knnrf}, Precision: {precision_knnrf}, Recall: {recall_knnrf}, F1-Score: {f1score_knnrf}\n")
    result_knnrf.append([accuracy_knnrf, precision_knnrf, recall_knnrf, f1score_knnrf])

    # Matriz de confusão do ensemble KNN e RandomForest
    predicted_knnrf = np.append(predicted_knnrf, pred_knnrf)
    actual_knnrf = np.append(actual_knnrf, y_test)

    metrics_knnrf[count_iter-1] = [accuracy_knnrf, precision_knnrf, recall_knnrf, f1score_knnrf]

    # Treina o ensemble entre SVM e RandomForest
    svmrf = VotingClassifier(estimators=[('svm', svm), ('rf', rf)], voting='hard')
    svmrf.fit(X_train, y_train)

    pred_svmrf = svmrf.predict(X_test)

    # Calcula as métricas de avaliação para o ensemble SVM e RandomForest
    accuracy_svmrf = svmrf.score(X_test, y_test)
    precision_svmrf = precision_score(y_test, pred_svmrf, average='weighted')
    recall_svmrf = recall_score(y_test, pred_svmrf, average='weighted')
    f1score_svmrf = f1_score(y_test, pred_svmrf, average='weighted')
    print(f"SVM+RF:\nAccuracy: {accuracy_svmrf}, Precision: {precision_svmrf}, Recall: {recall_svmrf}, F1-Score: {f1score_svmrf}\n")
    result_svmrf.append([accuracy_svmrf, precision_svmrf, recall_svmrf, f1score_svmrf])

    # Matriz de confusão do ensemble SVM e RandomForest
    predicted_svmrf = np.append(predicted_svmrf, pred_svmrf)
    actual_svmrf = np.append(actual_svmrf, y_test)

    metrics_svmrf[count_iter-1] = [accuracy_svmrf, precision_svmrf, recall_svmrf, f1score_svmrf]

    count_iter+=1
classes = ['AT', 'FER', 'MCL', 'MERC', 'RP', 'RBR', 'REN', 'WIL']

# Calcula a matrix de confusão para cada classificador
cm_knn = confusion_matrix(actual_knn, predicted_knn)
cm_svm = confusion_matrix(actual_svm, predicted_svm)
cm_rf = confusion_matrix(actual_rf, predicted_rf)
cm_knnrf = confusion_matrix(actual_knnrf, predicted_knnrf)
cm_svmrf = confusion_matrix(actual_svmrf, predicted_svmrf)

std_knn = np.std(metrics_knn, axis=0)
std_svm = np.std(metrics_svm, axis=0)
std_rf = np.std(metrics_rf, axis=0)
std_knnrf = np.std(metrics_knnrf, axis=0)
std_svmrf = np.std(metrics_svmrf, axis=0)

# Média de todas as métricas de avaliação para cada classificador
with open("results.txt", 'w') as file:
    file.write(f"KNN: {np.mean(result_knn, axis=0)}\nSVM: {np.mean(result_svm, axis=0)}\nRandomForest: {np.mean(result_rf, axis=0)}\n\nKNN+RandomForest: {np.mean(result_knnrf, axis=0)}\nSVM+RandomForest: {np.mean(result_svmrf, axis=0)}")
# print(f"KNN: {np.mean(result_knn, axis=0)}\nSVM: {np.mean(result_svm, axis=0)}\nRandomForest: {np.mean(result_rf, axis=0)}\n\nKNN+RandomForest: {np.mean(result_knnrf, axis=0)}\nSVM+RandomForest: {np.mean(result_svmrf, axis=0)}")

with open("CMs.txt", 'w') as file:
    file.write(f"KNN:\n{cm_knn}\n\nSVM:\n{cm_svm}\n\nRandomForest:\n{cm_rf}\n\nKNN+SVM:\n\nKNN+RandomForest:\n{cm_knnrf}\n\nSVM+RandfomForest:\n{cm_svmrf}")
print('Done')

with open("desviopadrao.txt", 'w') as file:
    file.write(f"Desvio Padrão KNN: {np.mean(std_knn)}\nDesvio Padrão SVM: {np.mean(std_svm)}\nDesvio Padrão RF: {np.mean(std_rf)}\nDesvio Padrão KNNRF: {np.mean(std_knnrf)}\nDesvio Padrão SVMRF: {np.mean(std_svmrf)}")

disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=classes)
disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=classes)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=classes)
disp_knnrf = ConfusionMatrixDisplay(confusion_matrix=cm_knnrf, display_labels=classes)
disp_svmrf = ConfusionMatrixDisplay(confusion_matrix=cm_svmrf, display_labels=classes)

#Plota as matrizes e gera as imagens
disp_knn.plot(colorbar=False, cmap='YlGn', xticks_rotation='vertical')
disp_svm.plot(colorbar=False, cmap='YlGn', xticks_rotation='vertical')
disp_rf.plot(colorbar=False, cmap='YlGn', xticks_rotation='vertical')
disp_knnrf.plot(colorbar=False, cmap='YlGn', xticks_rotation='vertical')
disp_svmrf.plot(colorbar=False, cmap='YlGn', xticks_rotation='vertical')


plt.show()