from sklearn.metrics import explained_variance_score
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import GridSearchCV

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
