import cv2 as cv
import numpy as np
import os
from tqdm import tqdm
from svm import SVM

from sklearn.svm import LinearSVC
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

BASE_DIR = r"C:\DS102\Lab3\chest_xray"

def collect_data(split="train"):

    images=[]
    labels=[]

    normal="NORMAL"
    pneumonia="PNEUMONIA"

    for img_file in tqdm(os.listdir(os.path.join(BASE_DIR,split,normal)),desc=f"{split}-NORMAL"):
        path=os.path.join(BASE_DIR,split,normal,img_file)
        img=cv.imread(path)

        if img is not None:
            img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            img=cv.resize(img,(128,128))
            img=img.reshape(-1)

            images.append(img)
            labels.append(-1)


    for img_file in tqdm(os.listdir(os.path.join(BASE_DIR,split,pneumonia)),desc=f"{split}-PNEUMONIA"):
        path=os.path.join(BASE_DIR,split,pneumonia,img_file)
        img=cv.imread(path)

        if img is not None:
            img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            img=cv.resize(img,(128,128))
            img=img.reshape(-1)

            images.append(img)
            labels.append(1)


    X=np.stack(images)
    X=X.astype(np.float32)/255.0
    y=np.array(labels)

    return X,y


if __name__=="__main__":

    print("Loading data...")

    X_train,y_train=collect_data("train")
    X_test,y_test=collect_data("test")


    mean=X_train.mean(axis=0)
    std=X_train.std(axis=0)+1e-8

    X_train=(X_train-mean)/std
    X_test=(X_test-mean)/std


    ##################################################
    # Custom SVM
    ##################################################

    model=SVM(C=1.0,lr=0.000001,n_iterations=150)

    print("\nTraining custom SVM...")
    model.fit(X_train,y_train)

    model.plot_loss()

    print("\nTesting custom SVM...")
    custom_scores=model.get_metrics(X_test,y_test)

    print("\nCustom SVM Results:")
    for k,v in custom_scores.items():
        print(f"{k}: {v:.4f}")


    ##################################################
    # Library SVM
    ##################################################

    print("\nTraining sklearn LinearSVC...")

    svm_lib=LinearSVC(
        C=1.0,
        loss="hinge",
        max_iter=150,
        tol=0.000001,
        dual=True
    )

    svm_lib.fit(X_train,y_train)

    y_pred=svm_lib.predict(X_test)

    lib_scores={
        "Accuracy":accuracy_score(y_test,y_pred),
        "Precision":precision_score(y_test,y_pred),
        "Recall":recall_score(y_test,y_pred),
        "F1":f1_score(y_test,y_pred)
    }


    print("\nLibrary SVM Results:")
    for k,v in lib_scores.items():
        print(f"{k}: {v:.4f}")


    ##################################################
    # Comparison
    ##################################################
    print("\n================ COMPARISON ================")
    print(f"{'Metric':<12}{'Custom':<12}{'Library':<12}")

    for metric in ["Accuracy","Precision","Recall","F1"]:
        print(
            f"{metric:<12}"
            f"{custom_scores[metric]:<12.4f}"
            f"{lib_scores[metric]:<12.4f}"
        )