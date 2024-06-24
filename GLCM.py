from sklearn import svm
from skimage.feature import graycomatrix, graycoprops
from sklearn import preprocessing
import numpy as np
import cv2 as cv
import time
from typing import NoReturn
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

class GLCMFeatureExtraction:
    '''
    Essa classe é responsável por extrair características de textura de imagens utilizando GLCM
    (Gray-Level Co-occurrence Matrix), que é uma matriz que descreve a frequência com que dois
    pixels de níveis de cinza específicos estão localizados em uma determinada relação espacial.
    Atualmente, está introduzindo o modelo de machine learning SVM (Support Vector Machine) para
    classificação de imagens.
    '''
    def __init__(self, dataset_path:str)->NoReturn:
        self.model = svm.SVC()
        
        self.train_features = []
        self.test_features = []
        
        self.train_image_path = ''
        self.test_image_path = ''
        self.train_feature_path = ''
        self.test_feature_path = ''
        self.create_directories(self)
        self.train_images, self.test_images, self.train_labels, self.test_labels = self.prepare_data(self, dataset_path)
        
    @staticmethod
    def extract_features(images)->np.array:
        '''
        Esse método extrai as características GLCM das imagens,
        como contraste, dissimilaridade, homogeneidade, energia e correlação.
        
        Args:
            images (np.array): As imagens a serem processadas.
        
        Returns:
            features_list (np.array): As características extraídas.
        '''
        features_list = []
        distances = [1]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        for image in images:
            if len(image.shape) > 2:
                image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            
            glcm = graycomatrix(image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
            
            contrast = graycoprops(glcm, 'contrast').flatten()
            dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
            homogeneity = graycoprops(glcm, 'homogeneity').flatten()
            energy = graycoprops(glcm, 'energy').flatten()
            correlation = graycoprops(glcm, 'correlation').flatten()
            
            features = np.hstack([contrast, dissimilarity, homogeneity, energy, correlation])
            features_list.append(features)
        return np.array(features_list, dtype=object)

    def train(self) -> np.array:
        '''
        Esse método realiza o pipeline de extração das características GLCM 
        das imagens de treinamento, o qual utiliza o modelo de machine learning
        SVM.
        
        Args:
            None
            
        Returns:
            None
        '''
        print(f'========= TREINANDO ========= ')
        encoder = preprocessing.LabelEncoder()
        train_encoded_labels = encoder.fit_transform(self.train_labels)
        print(f'São {len(self.train_images)} imagens em treino')
        self.train_features = self.extract_features(self.train_images)
        self.train_labels = train_encoded_labels
        self.save_data(self.train_feature_path, train_encoded_labels, self.train_features, encoder.classes_)
        self.model.fit(self.train_features, self.train_labels)
     
    def test(self) -> np.array:
        '''
        Esse método realiza o pipeline de extração das características GLCM 
        das imagens de teste. Além disso, ele salva os resultados obtidos através
        da chamada do método save_results().
        
        Args:
            None
            
        Returns:
            None
        '''
        print(f'=========== TESTANDO =========== ')
        encoder = preprocessing.LabelEncoder()
        test_encoded_labels = encoder.fit_transform(self.test_labels)
        print(f'São {len(self.test_images)} imagens em teste')
        self.test_features = self.extract_features(self.test_images)
        self.test_labels = test_encoded_labels
        self.save_data(self.test_feature_path, test_encoded_labels, self.test_features, encoder.classes_) 
        self.save_results()
    
    @staticmethod
    def create_directories(self) -> NoReturn:
        '''
        Esse método cria os diretórios necessários para armazenar os arquivos gerados.
        
        Args:
            None
        '''
        self.train_feature_path = './features_labels/glcm/train/'
        self.test_feature_path = './features_labels/glcm/test/'
        
        os.makedirs(self.train_feature_path, exist_ok=True)
        os.makedirs(self.test_feature_path, exist_ok=True)
        
    @staticmethod
    def encodeLabels(labels: np.array)->np.array:
        '''
        Esse método codifica os rótulos das imagens em números inteiros.
        
        Args:
            labels (np.array): Rótulos das imagens.
            
        Returns:
            encoded_labels (np.array): Rótulos codificados.
        '''
        startTime = time.time()
        print(f'Codificando rótulos em números inteiros')
        encoder = preprocessing.LabelEncoder()
        encoded_labels = encoder.fit_transform(labels)
        elapsedTime = round(time.time() - startTime, 2)
        print(f'Codificação feita em {elapsedTime}s')
        return np.array(encoded_labels, dtype=object), encoder.classes_
        
    @staticmethod
    def save_data(path:str, labels:np.array, features:np.array, encoded_classes:np.array)-> NoReturn:
        '''
        Método para salvar as características extraídas em arquivos
        de formato .csv, juntamente com os rótulos e as classes codificadas.
        Além disso, ele exibe a matriz de confusão e a acurácia obtida através
        da predição.
        
        Args:
            path (str): O diretório onde serão salvas as características.
            labels (np.array): Os rótulos das imagens.
            features (np.array): As características extraídas.
            encoded_classes (np.array): As classes codificadas.
        '''
        startTime = time.time()
        print(f'Salvando dados')
        
        label_filename = f'{labels=}'.split('=')[0] + '.csv'
        feature_filename = f'{features=}'.split('=')[0] + '.csv'
        encoder_filename = f'{encoded_classes=}'.split('=')[0] + '.csv'
        np.savetxt(path + label_filename, labels, delimiter=',', fmt='%i')
        np.savetxt(path + feature_filename, features, delimiter=',')
        np.savetxt(path + encoder_filename, encoded_classes, delimiter=',', fmt='%s')
        
        elapsedTime = round(time.time() - startTime, 2)
        print(f'Dados salvos em {elapsedTime}s')
        
    @staticmethod
    def prepare_data(self, path: str)-> np.array:
        '''
        Método para preparar os dados para treinamento e teste, dividindo
        as imagens em 80% para treinamento e 20% para teste e alocando-as
        em diretórios específicos.
        
        Args:
            path (str): O diretório onde estão as imagens.
        
        Returns:
            images_train (np.array): Imagens de treinamento.
            images_test (np.array): Imagens de teste.
            labels_train (np.array): Rótulos das imagens de treinamento.
            labels_test (np.array): Rótulos das imagens de teste.
        '''
        images = []
        labels = []
        if os.path.exists(path):
            for dirpath, _, filenames in os.walk(path):   
                if len(filenames) > 0:
                    folder_name = os.path.basename(dirpath)
                    for index, file in enumerate(filenames):
                        label = folder_name
                        labels.append(label)
                        full_path = os.path.join(dirpath, file)
                        image = cv.imread(full_path)
                        images.append(image)
            images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=0.2, random_state=1, stratify=labels)
            
            self.create_directories(self)

            return images_train, images_test, np.array(labels_train, dtype=object), np.array(labels_test, dtype=object)
        
        else:
            raise FileNotFoundError(f'O diretório {path} não existe.')
        
    def save_results(self) -> NoReturn:
        '''
        Método para salvar os resultados obtidos, imprimindo a acurácia e 
        exibindo a matriz de confusão.
        
        Args:
            None
            
        Returns:
            None
        
        '''
        predicted_labels = self.model.predict(self.test_features)

        accuracy = accuracy_score(self.test_labels, predicted_labels)
        print(f'Acurácia: {accuracy * 100:.2f}%')
        
        cm = confusion_matrix(self.test_labels, predicted_labels)

        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')
        plt.show()