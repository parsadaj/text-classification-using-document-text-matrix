import sys
import os
from utils import *
import json

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
# from sklearn.decomposition import PCA

def main():
    try:
        data_path, out_dir = sys.argv[1:]
    except ValueError:
        HW_path = os.getcwd()
        data_path, out_dir = os.path.join(HW_path, 'data', 'persica.csv'), os.path.join(HW_path, 'results')
    
    print('######### Reading Data #########')
    term_doc_train, term_doc_test, doc_label_train, doc_label_test, index_to_label = explore_data(data_path)

    with open("index_to_label", "w") as fp:
        json.dump(index_to_label, fp) 
    
    # term_doc_train, term_doc_test, doc_label_train, doc_label_test = load_data(data_path)
    print('###### Done Reading Data #######')

    print('###### Calculating TF-IDF ######')
    tf_idf_train = tf_idf(term_doc_train)
    tf_idf_test = tf_idf(term_doc_test)
    print('### Done Calculating TF-IDF ####')

    print('######## Performing SVD ########')
    R = 400
    doc_vector_train, doc_vector_test = SVD(tf_idf_train, tf_idf_test, R)
    print('######## Done with SVD #########')
    
    # np.save('doc_vector_train', doc_vector_train)
    # np.save('doc_vector_test', doc_vector_test)
    # np.save('doc_label_train', doc_label_train)
    # np.save('doc_label_test', doc_label_test)
    
    # doc_vector_train = np.load('doc_vector_train.npy')
    # doc_vector_test = np.load('doc_vector_test.npy')
    # doc_label_train = np.load('doc_label_train.npy')
    # doc_label_test = np.load('doc_label_test.npy')
    
    
    # pca = PCA(300)
    # X = pca.fit_transform(doc_vector_train)
    # X_test = pca.transform(doc_vector_test)
    
    

    classifiers = [
        SVC(C=10), #######
        SVC(kernel='sigmoid'), #######
        RandomForestClassifier(max_depth=8, max_features='log2'), #######
        RandomForestClassifier(max_depth=10, max_features='sqrt'), #######
        MLPClassifier(alpha=1, hidden_layer_sizes=(200,100)), #######
        MLPClassifier(alpha=1, hidden_layer_sizes=(400)),
        MLPClassifier(alpha=1, hidden_layer_sizes=(100)), #######
    ]

    n_classifier = len(classifiers)
    for i, classifier in enumerate(classifiers):

        print(f'#### Training and Testing on Model {i+1} / {n_classifier} ####')
    
        classifier.fit(doc_vector_train, doc_label_train)
        y_hat = classifier.predict(doc_vector_train)
        
        confusion_matrix_train = confusion_matrix(doc_label_train, y_hat)
        acc_train = accuracy_score(doc_label_train, y_hat)
        metrics_train = [confusion_matrix_train, acc_train]

        
        y_hat_test = classifier.predict(doc_vector_test)
        confusion_matrix_test = confusion_matrix(doc_label_test, y_hat_test)
        acc_test = accuracy_score(doc_label_test, y_hat_test)

        metrics_test = [confusion_matrix_test, acc_test]

        
        log_metrics_to_file(i+1, repr(classifier), metrics_train, metrics_test, out_dir)
        print(f'########### Done with Model {i+1} ###########')


        

if __name__ == "__main__":
    main()