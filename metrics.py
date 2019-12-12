import sklearn, sys
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
import logging
import tensorflow as tf
from keras import backend as K


def predictMetricsRegression(embeddings, df_train, df_test):
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
    
    y_test = y_test*10 + normal(loc=0.0, scale=5, size=y_test.shape)
    y_train = y_train*10 + normal(loc=0.0, scale=5, size=y_train.shape)

    scaler = MinMaxScaler()
    scaler.fit(y_train.reshape(-1,1))
    print(scaler.data_range_)
    y_train = scaler.transform(y_train.reshape(-1,1))
    y_test = scaler.transform(y_test.reshape(-1,1))
    #print(y_train.shape, y_test.shape)

    model = MLPRegressor(input_dim=int(2*embeddings.shape[1]), output_dims=1)

    print(model.summary())
    history = model.fit(x=train, y=y_train, batch_size=128, epochs=20, verbose=1)
    train_eval = model.evaluate(x=train, y=y_train)
    test_eval = model.evaluate(x=test, y=y_test)
    
    print("Evaluation Scores for Training Data ")
    print(train_eval)

    print("Evaluation Scores for Testing Data ")
    print(test_eval)
    return history, model

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


    log = open("Classification_MLP.log", "a")
    sys.stdout = log

    print("........................Weight Prediction as a Regression Problem..................................")
    print("\n")
    print("Weight Prediction Scores for bitcoin OTC dataset")
    history, model = predictMetricsRegression(embeddings_otc, df_train_otc, df_test_otc)
    tf.keras.models.save_model(model, "results\\MLPRegressor_OTC.h5py")
    with open('results\\MLPRegressor_TrainHist_{}.pkl'.format('bitcoinotc'), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    del model

    print("\n")
    print("Weight Prediction Scores for bitcoin alpha dataset")
    # predictMetrics(embeddings_alpha, df_train_alpha, df_test_alpha)
    history, model = predictMetricsRegression(embeddings_alpha, df_train_alpha, df_test_alpha)
    tf.keras.models.save_model(model, "results\\MLPRegressor_alpha.h5py")
    with open('results\\MLPRegressor_TrainHist_{}.pkl'.format('bitcoinalpha'), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    del model

    print("........................Sign Prediction as a Classification Problem.................................")
    print("\n")
    print("Sign Prediction Scores for bitcoin OTC dataset")
    history, model = predictMetricsClassification(embeddings_otc, df_train_otc, df_test_otc)
    tf.keras.models.save_model(model, "results\\MLPClassifier_OTC.h5py")
    with open('results\\MLPClassifier_TrainHist_{}.pkl'.format('bitcoinotc'), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    del model
    
    print("\n")
    print("Sign Prediction Scores for bitcoin alpha dataset")
    print(np.sum(embeddings_alpha), df_train_alpha.shape, df_test_alpha.shape)
    history, model = predictMetricsClassification(embeddings_alpha, df_train_alpha, df_test_alpha)
    tf.keras.models.save_model(model, "results\\MLPClassifier_alpha.h5py")
    with open('results\\MLPClassifier_TrainHist_{}.pkl'.format('bitcoinalpha'), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)