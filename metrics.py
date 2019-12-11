from sklearn.metrics import explained_variance_score, r2_score
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import h5py
import pandas as pd
import numpy as np

def predictMetrics(embeddings, df_train, df_test):
    train = np.zeros((df_train.shape[0], 2*embeddings.shape[1]))
    y_train = np.zeros((df_train.shape[0], 1))
    test = np.zeros((df_test.shape[0], 2*embeddings.shape[1]))
    y_test = np.zeros((df_test.shape[0], 1))

    for i in range(0, df_train.shape[0]):
        train[i,:] = np.append(embeddings[int(df_train['src'].iloc[i])], embeddings[int(df_train['dst'].iloc[i])])
        y_train[i,0] = df_train['rating'].iloc[i]

    print(df_test.shape[0])
    for i in range(0, df_test.shape[0]):
        test[i,:] = np.append(embeddings[int(df_test['src'].iloc[i])], embeddings[int(df_test['dst'].iloc[i])])
        y_test[i,0] = df_test['rating'].iloc[i]
    

    model = svm.LinearSVR()
    model.fit(train, y_train)
    y_pred_train = model.predict(train)
    y_pred_test = model.predict(test)
    print("Explained Variance for Train Dataset : {}".format(explained_variance_score(y_pred_train, y_train, multioutput='uniform_average')))
    print("Explained Variance for Test Dataset : {}".format(explained_variance_score(y_pred_test, y_test, multioutput='uniform_average')))

    print("R2 coefficient for Train Dataset : {}".format(r2_score(y_pred_train, y_train, multioutput='uniform_average')))
    print("R2 coefficient for Test Dataset : {}".format(r2_score(y_pred_test, y_test, multioutput='uniform_average')))
    return None


def get_explained_variance_score(embeddings,G):
    A = np.squeeze(np.asarray(nx.adjacency_matrix(G).todense()))
    num_nodes = A.shape[0]
    X = []
    y = []
    for i in range(0,N):
        for j in range(0,N):
            if(A[i][j]!=0):
                X.append(np.append(embeddings[i],embeddings[j]))
                y.append(A[i,j])

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    hyperparameter_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    print("# Tuning hyperparameters for SVR")
    reg =  GridSearchCV(svm.SVR(), hyperparameter_grid, cv=5)
    reg.fit(X_train,y_train)
    print("Best parameters set found:")
    print(reg.best_params_)

    y_pred = reg.predict(X_test)
    return explained_variance_score(y_test,y_pred,multioutput='uniform_average')

if __name__ == "__main__":
    with h5py.File('embeddings\\Embeddings_64_soc-sign-bitcoinotc_8.h5', 'r') as f:
        embeddings_otc = np.array(f.get('Embeddings'))
    
    with h5py.File('embeddings\\Embeddings_128_soc-sign-bitcoinalpha.h5', 'r') as f:
        embeddings_alpha = np.array(f.get('Embeddings'))

    df_train_otc = pd.read_csv("datasets\\train_soc-sign-bitcoinotc.csv")
    df_test_otc = pd.read_csv("datasets\\test_soc-sign-bitcoinotc.csv")
    df_train_alpha = pd.read_csv("datasets\\train_soc-sign-bitcoinalpha.csv")
    df_test_alpha = pd.read_csv("datasets\\test_soc-sign-bitcoinalpha.csv")

    print(embeddings_otc.shape)
    print(embeddings_alpha.shape)
    print(df_test_otc.shape, df_test_otc.shape)
    print("\n")
    print("Prediction Scores for bitcoin OTC dataset")
    predictMetrics(embeddings_otc, df_train_otc, df_test_otc)

    print("\n")
    print("Prediction Scores for bitcoin alpha dataset")
    predictMetrics(embeddings_alpha, df_train_alpha, df_test_alpha)


