import numpy as np
import os
import re
import subprocess
from sklearn.decomposition import TruncatedSVD


URL = r" <URL> "
NUM = r" <NUM> "
USER = r" <USERNAME> "

SEP = ' '
NEWLINE = '\n'

STOP_WORDS = [
    'در',
    'از',
    'که',
    'با',
    'و',
    'یا',
    'به',
    'تا',
    'حتی',
    'ها',
    'را',
    'برای',
    'اما',
    'چون'
]
def preprocess_text(text: str):
    text = re.sub("\.[\.]+|[()\[\]{}<>\"\']", '', text)

    text = re.sub("ك", 'ک', text)
    text = re.sub('ي', 'ی', text)
    text = re.sub('ؤ', 'و', text)
    text = re.sub('[أآ]', 'ا', text)
    text = re.sub('ة', 'ه', text)

    url_pattern = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    text = re.sub(url_pattern, URL, text)

    ip_pattern = r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})(:(\d)+)?'
    text = re.sub(ip_pattern, URL, text)

    email_pattern = '\S+@\S+'
    text = re.sub(email_pattern, URL, text)

    legal_farsi = 'ا-ی'
    legal_arabic = 'ئ'
    legal_english = 'a-zA-z'
    legal_number = '0-9۰-۹'
    legal_other = '\s<>\n'

    illegal_pattern = "[^{}]".format(legal_arabic + legal_english + legal_farsi + legal_number + legal_other)

    text = re.sub(illegal_pattern, ' ', text)

    text = re.sub("(?<=[a-zئA-Zا-ی])(?=\d)", ' ', text)
    text = re.sub("(?<=\d)(?=[a-zئA-Zا-ی])", ' ', text)

    num_pattern = r"\b\d+\b"
    text = re.sub(num_pattern, NUM, text)
    

    words = re.split("\s+", text.strip())

    
    return words

def explore_data(data_path):
    f = open(data_path)
    
    label_to_index = {}
    index_to_label = {}
    
    word_to_index = {}
    index_to_word = {}
    tf = dict(train=[], test=[])
    doc_labels = dict(train=[], test=[])

    n_lines = get_num_lines(data_path) // 7 + 1
    counter = 0
    
    while True:
        if (counter+1) % 1000 == 0:
            print('{} / {}'.format(counter+1, n_lines), end='\r')
        counter += 1
        
        doc_number = f.readline()
        doc_title = f.readline()
        doc = f.readline()
        date = f.readline()
        time = f.readline()
        label_jozi = f.readline()
        label_kolli = f.readline()

        if len(doc) == 0:
            break
        
        doc_vector = [0 for _ in word_to_index]
        
        phase = np.random.choice(['train', 'test'], p=[0.8, 0.2])
        
        for word in preprocess_text(doc):
            if word in STOP_WORDS:
                continue
            try:
                doc_vector[word_to_index[word]] += 1
            except KeyError:
                if phase == 'train':
                    word_index = len(word_to_index)
                    word_to_index[word] = word_index
                    index_to_word[word_index] = word
                    doc_vector.append(1)
                
        tf[phase].append(doc_vector)
        
        try:
            label_index = label_to_index[label_kolli]
        except KeyError:
                label_index = len(label_to_index)
                label_to_index[label_kolli] = label_index
                index_to_label[label_index] = label_kolli
            
        doc_labels[phase].append(label_index)
    f.close()
    
    n_words = len(word_to_index)
    n_docs_train = len(tf['train'])
    n_docs_test = len(tf['test'])
    
    tf_train = np.zeros((n_words, n_docs_train))
    for i, document in enumerate(tf['train']):
        tf_train[:len(document), i] = document
    
    tf_test = np.zeros((n_words, n_docs_test))
    for i, document in enumerate(tf['test']):
        tf_test[:len(document), i] = document
     
    
    return tf_train, tf_test, np.array(doc_labels['train']), np.array(doc_labels['test']), index_to_label

def tf_idf(tf: np.ndarray):
    """calculates TF_IDF matrix

    Args:
        tf (2D array): term frequence; tf[i, j]: numebr of times term i occured in doc j

    Returns:
        2D array: tf_idf
    """
    ni = np.sum(tf >= 1, axis=1, keepdims=True) + 0.1
    N = tf.shape[1]
    idf = np.log(N/ni)
    return tf * idf
    
def get_metrics(confusion_matrix, actual_dim=0, predicted_dim=1):
    """calculates accuracy, precision and recall from confusion matrix

    Args:
        confusion_matrix (2D array)

    Returns:
        tuple: (accuracy, precision, recall)
    """
    diag = np.diag(confusion_matrix)
    accuracy = np.sum(diag) / np.sum(confusion_matrix)
    precison = diag / np.sum(confusion_matrix, axis=actual_dim)
    recall = diag / np.sum(confusion_matrix, axis=predicted_dim)
    
    return accuracy, precison, recall

def evaluate_model(model, term_doc_test, doc_label_test, UR, SR):
    pass

def SVD(tf_train, tf_test, R=None):
    svd = TruncatedSVD(n_components=R)
    new_train = svd.fit_transform(tf_train.T)
    new_test = svd.transform(tf_test.T)
    
    return new_train, new_test
    # """performs SVD decomposition and keeps only best R components

    # Args:
    #     term_doc (2D array[MxN]): term frequence; tf[i, j]: numebr of times term i occured in doc j
    #     R (int, optional): number of components to keep. Defaults to number of singular values.

    # Returns:
    #     tuple: u[MxR], v[RxR], vh[RxN]
    # """
    # u, s, vh = np.linalg.svd(term_doc)
    # if R is None:
    #     R = len(s)
    # return u[:, :R], np.diag(s[:R]), vh[:R, :]

def log_metrics_to_file(model_number, model_name, train_metrics, test_metrics, out_dir, create_if_not_exists=True):
    """logs model hyperparameters and results to file

    Args:
        model_number (int): a number to distinct different models
        model_name (str): name of the classifier used
        train_metrics (list): [accuracy, precision, recall]
        test_metrics (list): [accuracy, precision, recall]
        out_dir (str): directory to result folder
        args_types (list): types of the hyperparameters
        create_if_not_exists (bool, optional): if out_dir doesn't exist creates it. Defaults to True.
    """
    file_name = os.path.join(out_dir, 'logs', str(model_number) + '.txt')
    if (not os.path.exists(os.path.join(out_dir, 'logs'))) and create_if_not_exists:
        os.makedirs(os.path.join(out_dir, 'logs'))
    f = open(file_name, 'w')
    f.write(str(model_name) + '\n\n')
    f.write('Train:\naccuracy: {acc}\nconfusion_matrix:\n{conf}\n\n'.format(acc=str(train_metrics[1]), conf=str(train_metrics[0])))
    f.write('Test:\naccuracy: {acc}\nconfusion_matrix:\n{conf}\n\n'.format(acc=str(test_metrics[1]), conf=str(test_metrics[0])))
    f.close()

def pad_list(l: list, content, width):
    l.extend([content for _ in range((width - len(l)))])
    return l
    
def get_num_lines(path):
    """counts number of lines in the given file

    Args:
        path (string): path to file

    Returns:
        int: number of lines
    """
    return int(subprocess.check_output(['wc', '-l', path]).split()[0])

if __name__ == '__main__':
    preprocess_text('سلام بمنلت. تانخلتپلذ لذر؟ بیهت نن تد پن پت م!')