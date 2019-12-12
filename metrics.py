import sklearn
from sklearn.metrics import explained_variance_score, r2_score, classification_report, mean_squared_log_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from numpy.random import normal
import h5py
import pickle
import pandas as pd
import numpy as np
from src.MLP import *

def predictMetrics(embeddings, df_train, df_test):
    train = np.zeros((df_train.shape[0], 2*embeddings.shape[1]))
    y_train = np.zeros((df_train.shape[0],1))
    test = np.zeros((df_test.shape[0], 2*embeddings.shape[1]))
    y_test = np.zeros((df_test.shape[0],1))

    for i in range(0, df_train.shape[0]):
        train[i,:] = np.append(embeddings[int(df_train['source'].iloc[i])], embeddings[int(df_train['target'].iloc[i])])
        y_train[i,0] = df_train['weight'].iloc[i]

    print(df_test.shape[0])
    for i in range(0, df_test.shape[0]):
        test[i,:] = np.append(embeddings[int(df_test['source'].iloc[i])], embeddings[int(df_test['target'].iloc[i])])
        y_test[i,0] = df_test['weight'].iloc[i]
    
    y_test = y_test + normal(loc=0.0, scale=0.5, size=y_test.shape)
    y_train = y_train + normal(loc=0.0, scale=0.5, size=y_train.shape)

    scaler = MinMaxScaler()
    scaler.fit(y_train)
    print(scaler.data_range_)
    y_train_tr = scaler.transform(y_train).reshape(-1,)
    y_test_tr = scaler.transform(y_test).reshape(-1,)
    print(y_train_tr.shape, y_test_tr.shape)

    param_dist = {'n_estimators': [50, 100],
                'learning_rate' : [0.1, 1, 3, 5],
                'loss' : ['linear', 'square'] }

    model = RandomizedSearchCV(AdaBoostRegressor(),
                        param_distributions = param_dist,
                        cv=4,
                        n_iter = 10,
                        n_jobs=-1)

    model.fit(train, y_train_tr)
    print(model.best_params_)
    y_pred_train = model.predict(train)
    y_pred_test = model.predict(test)

    print("Explained Variance for Train Dataset : {}".format(explained_variance_score(y_pred_train, y_train_tr, multioutput='uniform_average')))
    print("Explained Variance for Test Dataset : {}".format(explained_variance_score(y_pred_test, y_test_tr, multioutput='uniform_average')))

    print("R2 coefficient for Train Dataset : {}".format(r2_score(y_pred_train, y_train_tr, multioutput='uniform_average')))
    print("R2 coefficient for Test Dataset : {}".format(r2_score(y_pred_test, y_test_tr, multioutput='uniform_average')))

    print("mean_square_log_error coefficient for Train Dataset : {}".format(mean_squared_log_error(y_pred_train, y_train_tr, multioutput='uniform_average')))
    print("mean_square_log_error for Test Dataset : {}".format(mean_squared_log_error(y_pred_test, y_test_tr, multioutput='uniform_average')))

    print("neg_root_mean_squared_error coefficient for Train Dataset : {}".format(mean_squared_error(y_pred_train, y_train_tr, multioutput='uniform_average')))
    print("neg_root_mean_squared_error for Test Dataset : {}".format(mean_squared_error(y_pred_test, y_test_tr, multioutput='uniform_average')))
    

    # print(classification_report(y_train, y_pred_train))
    # print(classification_report(y_test, y_pred_test))
    return None


def predictMetricsClassification(embeddings, df_train, df_test):
    train = np.zeros((df_train.shape[0], 2*embeddings.shape[1]))
    y_train = np.zeros((df_train.shape[0],1))
    test = np.zeros((df_test.shape[0], 2*embeddings.shape[1]))
    y_test = np.zeros((df_test.shape[0],1))

    y_test = df_test['weight'].values
    y_train = df_train['weight'].values

    for i in range(0, df_train.shape[0]):
        train[i,:] = np.append(embeddings[int(df_train['source'].iloc[i])], embeddings[int(df_train['target'].iloc[i])])

    for i in range(0, df_test.shape[0]):
        test[i,:] = np.append(embeddings[int(df_test['source'].iloc[i])], embeddings[int(df_test['target'].iloc[i])])
    
    y_train = np.where(y_train>0, 1, 0)
    y_test = np.where(y_test>0, 1, 0)


    print(np.unique(y_train))

    model = MLPClassifier(input_dim=int(2*embeddings.shape[1]), output_dims=1)

    print(model.summary())
    history = model.fit(x=train, y=y_train, batch_size=256, epochs=5, verbose=1)
    train_eval = model.evaluate(x=train, y=y_train)
    test_eval = model.evaluate(x=test, y=y_test)
    
    print("Evaluation Scores for Training Data ")
    print(train_eval)

    print("Evaluation Scores for Testing Data ")
    print(test_eval)
    return history, model
    

if __name__ == "__main__":
    with h5py.File('embeddings\\Embeddings_soc-sign-bitcoinotc_3.h5', 'r') as f:
        embeddings_otc = np.array(f.get('Embeddings'))
    
    with h5py.File('embeddings\\Embeddings_soc-sign-bitcoinalpha_3.h5', 'r') as f:
        embeddings_alpha = np.array(f.get('Embeddings'))

    df_train_otc = pd.read_csv("datasets\\mod\\train_soc-sign-bitcoinotc.csv")
    df_test_otc = pd.read_csv("datasets\\mod\\test_soc-sign-bitcoinotc.csv")
    df_train_alpha = pd.read_csv("datasets\\mod\\train_soc-sign-bitcoinalpha.csv")
    df_test_alpha = pd.read_csv("datasets\\mod\\test_soc-sign-bitcoinalpha.csv")

    print("\n")
    print("Prediction Scores for bitcoin OTC dataset")
    history, model = predictMetricsClassification(embeddings_otc, df_train_otc, df_test_otc)
    tf.keras.models.save_model(model, "results\\")
    with open('results\\MLPClassifier_TrainHist_{}.pkl'.format('bitcoinotc'), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    del model
    print("\n")
    print("Prediction Scores for bitcoin alpha dataset")
    print(np.sum(embeddings_alpha), df_train_alpha.shape, df_test_alpha.shape)
    history = predictMetricsClassification(embeddings_alpha, df_train_alpha, df_test_alpha)
    tf.keras.models.save_model(model, "results\\")
    with open('results\\MLPClassifier_TrainHist_{}.pkl'.format('bitcoinalpha'), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    

    # print("Positives in Test and Train Datsets", y_train)
    # print(".....................................Classification Report for Adaboost Classifier........................................................")
    
    # param_dist = {'n_estimators': [50, 100, 150],
    #             'learning_rate' : [0.1, 1, 3, 5]}

    # model_ADABOOST = RandomizedSearchCV(AdaBoostClassifier(),
    #                     param_distributions = param_dist,
    #                     cv=4,
    #                     n_iter = 10,
    #                     n_jobs=-1)

    # model_ADABOOST.fit(train, y_train)
    # y_pred_train = model_ADABOOST.predict(train)
    # y_pred_test = model_ADABOOST.predict(test)
    # print(model_ADABOOST.best_params_)
    # print('...........................For Train Dataset.....................................')
    # print(classification_report(y_train, y_pred_train))
    # print('...........................For Test Dataset.....................................')
    # print(classification_report(y_test, y_pred_test))

    # print(".....................................Classification Report for SVM Classifier........................................................")
    
    # param_dist = {'C': [0.1, 1, 10],
    #             'kernel' : ['rbf', 'linear'],
    #             'gamma': ['auto']}

    # model_SVM = RandomizedSearchCV(SVC(),
    #                     param_distributions = param_dist,
    #                     cv=4,
    #                     n_iter = 10,
    #                     n_jobs=-1)

    # model_SVM.fit(train, y_train)
    # y_pred_train = model_SVM.predict(train)
    # y_pred_test = model_SVM.predict(test)
    # print(model_SVM.best_params_)
    # print('...........................For Train Dataset.....................................')
    # print(classification_report(y_train, y_pred_train))
    # print('...........................For Test Dataset.....................................')
    # print(classification_report(y_test, y_pred_test))
    # return None