import numpy as np
from dataset_processing import load_and_transform_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from networkTorch_multiClass import Network, train, predict
import pandas as pd

def obj_funct(args):
    nb_try = 10
    X = args["X"].to_numpy()
    X_test = args["X_tst"].to_numpy()
    Y = args['Y']
    Y_test = args["Y_tst"]
    for i in range(nb_try):
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, shuffle=True, stratify=Y)
        batch_size = int(X_train.shape[0]*0.05)
        model = Network(X_train.shape[1], args["nbRules"], args["nbOutput"], loc_mean=args["loc_mean"])
        nbCond, best_epoch = train(model, X_train, Y_train, X_val, Y_val, batch_size, args["nbOutput"], 
                                    learning_rate=args["lr"], lambda_and=args['lambda_and'], 
                                    epochs=args["epochs"], device=args["device"], callback=args['callback'], 
                                    class_weights=args["wLoss"], limit=args["limit"], l2_lambda=args["l2_lambda"])
        train_pred = predict(model, X_train)
        train_acc = accuracy_score(Y_train, train_pred)
        val_pred = predict(model, X_val)
        val_acc = accuracy_score(Y_val, val_pred)
        test_pred = predict(model, X_test)
        test_acc = accuracy_score(Y_test, test_pred)
        with open(args['filename']+'.csv','a+') as fd:
            row = "" + args["name"] + "," + str(args["nbRules"]) + "," + str(args["lr"]) + "," + str(args["lambda_and"]) + "," + str(args["loc_mean"]) + "," + str(args["epochs"])+ ','+ str(args["limit"]) + ','+ str(args["callback"])+ ','+ str(args["wLoss"])+ "," + str(i)+","+ str(best_epoch) + "," + str(nbCond) + "," + str(train_acc) + "," + str(val_acc) + "," + str(test_acc)+ "\n"
            fd.write(row)

def worker(name):
    X, Y, X_tst, Y_tst = load_and_transform_data(name, doFeatureBinarizer=True, doOrdinalEncode=False)
    print("dataset", name, X.shape)
    nbOutput = len(np.unique(Y))
    device = "cpu" #cuda, gpu
    filename = name+"_train_test"
    csv = "parameters.csv"
    df = pd.read_csv(csv, sep=",")
    i = df[df.name==name].index[0]
    lr = df.lr[i]
    lambda_and = df.lambda_and[i]
    loc_mean = df.loc_mean[i]
    limit = df.limit[i]
    l2_lambda = df.l2_lambda[i]
    epochs = df.epochs[i]
    wLoss = df.class_weights[i]
    callback = df.callback[i]
    for r in range(2,21,2):
        params = {"nbRules": r,
                  "lr": lr,
                  "lambda_and": lambda_and,
                  "epochs": epochs,
                  "X": X,
                  "Y": Y,
                  "X_tst": X_tst,
                  "Y_tst": Y_tst,
                  "nbOutput": nbOutput,
                  "loc_mean": loc_mean,
                  "wLoss": wLoss,
                  "limit": limit,
                  "l2_lambda": l2_lambda,
                  "callback": callback,
                  "name": name, 
                  "device": device,
                  "filename": filename}
        obj_funct(params)
        

name = "mushroom"
#["adult", "magic", "house", "heloc", "mushroom", "chess", "ads", "nursery", "car", "pageblocks", "pendigits", "contraceptivemc", "drive", "covtype"]
worker(name)
