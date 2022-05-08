import os
import warnings

warnings.filterwarnings("ignore")
import datetime
import time
import cv2
import pickle

from numpy import savez_compressed, asarray, load, expand_dims
from mtcnn.mtcnn import MTCNN
from os import listdir
from os.path import isdir
from PIL import Image
from keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC


from augmentation import rewrite_to_augmented

CFD = os.path.dirname(os.path.abspath(__file__))


class FaceTrainer:

    def __init__(self, project_dirpath):
        self.dataset_train = os.path.join(project_dirpath, 'faces_dataset\\train\\')
        self.dataset_val = os.path.join(project_dirpath, "faces_dataset\\val\\")
        self.faces_npz = os.path.join(project_dirpath, "face_train_dataset.npz")
        self.keras_facenet = os.path.join(project_dirpath, "keras_facenet\\model\\facenet_keras.h5")
        self.faces_embeddings = os.path.join(project_dirpath, "faces_dataset_embeddings.npz")
        self.svm_classifier = os.path.join(project_dirpath, "SVM_classifier.sav")

    def load_dataset(self, directory):
        """Load a dataset that contains one subdir for each
         class that in turn contains images."""
        X, y = [], []
        # enumerate all folders named with class labels
        for subdir in listdir(directory):
            path = directory + subdir + '\\'
            # skip any files that might be in the dir
            if not isdir(path):
                continue
            # load all faces in the subdirectory
            faces = self.load_faces(path)
            # create labels
            labels = [subdir for _ in range(len(faces))]
            print(f">loaded {len(faces)} examples for class: {subdir}")
            X.extend(faces)
            y.extend(labels)
        return asarray(X), asarray(y)

    def load_faces(self, directory):
        """
        Load images and extract faces for all images in a directory
        """
        faces = []
        # enumerate files
        for filename in listdir(directory):
            path = directory + filename
            print(path)
            # get face or augment it
            face = self.extract_face(path)
            if face is None:
                print(f'I can`t find a person in {filename}!\nI will try to use augmentation.\n')
                back = cv2.imread('backg.jpg')
                rewrite_to_augmented(path, back)
                continue
            faces.append(face)
        return faces

    def extract_face(self, filename, required_size=(160, 160)):
        """
        Extract a single face from a given photograph
        """
        image = Image.open(filename)
        # convert to RGB, if needed
        image = image.convert('RGB')
        # convert to array
        pixels = asarray(image)
        # create the detector, using default weights
        detector = MTCNN()
        # detect faces in the image
        results = detector.detect_faces(pixels)
        if len(results) == 0:
            return
        x1, y1, width, height = results[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1:y2, x1:x2]
        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(required_size)
        return asarray(image)

    def create_faces_npz(self):
        """Creates npz file for all the faces in train_dir, val_dir"""
        # Load the training data set
        train_X, train_y = self.load_dataset(self.dataset_train)
        print("Training data set loaded.")
        # load test dataset
        test_X, test_y = self.load_dataset(self.dataset_val)
        print("Testing data set loaded.")
        # save arrays to one file in compressed format
        savez_compressed(self.faces_npz, train_X, train_y, test_X, test_y)
        return

    def create_faces_embedding_npz(self):
        """Create npz file for all the face embeddings in train_dir, val_dir"""
        data = load(self.faces_npz)
        train_X, train_y, test_X, test_y = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
        print('Loaded: ', train_X.shape, train_y.shape, test_X.shape, test_y.shape)
        # load the facenet model
        model = load_model(self.keras_facenet)
        print('Keras Facenet Model Loaded')
        # convert each face in the train set to an embedding
        newTrainX = self.face_to_embedings(train_X, model)
        newTestX = self.face_to_embedings(test_X, model)
        # save arrays to one file in compressed format
        savez_compressed(self.faces_embeddings, newTrainX, train_y, newTestX, test_y)
        return

    def face_to_embedings(self, faces, model):
        """Convert each face in the train set to an embedding."""
        embedings = []
        for face_pixels in faces:
            embedding = self.get_embedding(model, face_pixels)
            embedings.append(embedding)
        embedings = asarray(embedings)
        return embedings

    def get_embedding(self, model, face_pixels):
        """Get the face embedding for one face"""
        # scale pixel values
        face_pixels = face_pixels.astype('float32')
        # standardize pixel values across channels (global)
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        # transform face into one sample
        samples = expand_dims(face_pixels, axis=0)
        # make prediction to get embedding
        yhat = model.predict(samples)
        return yhat[0]

    def classifier(self):
        """Create a Classifier for the Faces Dataset"""
        # load dataset
        data = load(self.faces_embeddings)
        train_X, train_y, test_X, test_y = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
        print(f'Dataset: train={train_X.shape[0]}, test={test_X.shape[0]}')
        # normalize input vectors
        in_encoder = Normalizer(norm='l2')
        train_X = in_encoder.transform(train_X)
        test_X = in_encoder.transform(test_X)
        # label encode targets
        out_encoder = LabelEncoder()
        out_encoder.fit(train_y)
        train_y = out_encoder.transform(train_y)
        test_y = out_encoder.transform(test_y)
        # fit model
        model = SVC(kernel='linear', probability=True)
        model.fit(train_X, train_y)
        # save the model to disk
        filename = self.svm_classifier
        pickle.dump(model, open(filename, 'wb'))
        # predict
        yhat_train = model.predict(train_X)
        yhat_test = model.predict(test_X)
        # score
        score_train = accuracy_score(train_y, yhat_train)
        score_test = accuracy_score(test_y, yhat_test)
        # summarize
        print(f'Accuracy: train={score_train * 100:.3f}, test={score_test * 100:.3f}')
        return

    def start(self):
        start_time = time.time()
        st = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
        print("-----------------------------------------------------------------------------------------------")
        print(f"Face trainer Initiated at {st}")
        print("-----------------------------------------------------------------------------------------------")
        # Get faces from the images
        self.create_faces_npz()
        # Get embeddings for all the extracted faces
        self.create_faces_embedding_npz()
        # Classify the faces
        self.classifier()
        end_time = time.time()
        et = datetime.datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')
        print("-----------------------------------------------------------------------------------------------")
        print(f"Face trainer Completed at {et}")
        print(f"Total time Elapsed {round(end_time - start_time)} secs")
        print("-----------------------------------------------------------------------------------------------")

        return


if __name__ == "__main__":
    facetrainer = FaceTrainer(CFD)
    facetrainer.start()
