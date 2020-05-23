import os

import cv2
# face verification with the VGGFace2 model
from cv2.cv2 import imwrite
from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input


# all faces
def extract_all_faces(user, filename, id, required_size=(224, 224)):
    # load image from file
    pixels = pyplot.imread('/home/maria/git/{0}/{1}'.format(user, filename))
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract
    face_array = []
    face_id = 0
    for i in range(len(results)):
        x, y, width, height = results[i]['box']
        if x < 0: x = 0
        if y < 0: y = 0
        x2, y2 = x + width, y + height
        face = pixels[y:y2, x:x2]
        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array.append(asarray(image))
    return face_array


# extract faces and calculate face embeddings for a list of photo files
def get_embeddings(user):
    filenames = os.listdir('/home/maria/git/%s' % user)
    faces = []
    # faces[face_array[asarray]]
    for f in filenames:
        faces.extend(extract_all_faces(user, f, id))
    # convert into an array of samples
    #print(faces)
    samples = asarray(faces, 'float32')
    # prepare the face for the model, e.g. center pixels
    samples = preprocess_input(samples, version=2)
    # create a vggface model
    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    # perform prediction
    yhat = model.predict(samples)
    return yhat


# determine if a candidate face is a match for a known face
def is_match(known_embedding, candidate_embedding, thresh=0.5):
    # calculate distance between embeddings
    score = cosine(known_embedding, candidate_embedding)
    if score <= thresh:
        print('>face is a Match (%.3f <= %.3f)' % (score, thresh))
        scoreNum = 1
    else:
        print('>face is NOT a Match (%.3f > %.3f)' % (score, thresh))
        scoreNum = 0
    return scoreNum


User1 = 'user1'
User2 = 'user2'

embeddings1 = get_embeddings(User1)
embeddings2 = get_embeddings(User2)

score = 0
for face1 in embeddings1:
    for face2 in embeddings2:
        if is_match(face1, face2):
            score += 1
        break
print('score between {0} and {1} is {2}'.format(User1, User2, score))
