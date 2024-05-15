
from tensorflow.keras.utils import img_to_array
from keras.models import load_model
import cv2
import numpy as np
from facenet_pytorch import MTCNN
from essentials_image import image_input, open_bounding_box1


class Emotion:
    '''
        Emotion - Class for detect the emotion from the detected faces in the image
    '''

    def __init__(self, cap, mtcnn):
        '''
            __init__ - initialize the variables
        '''

        self.cap = cap
        self.mtcnn = mtcnn  # MTCNN object

        self.model = load_model('face expression/Emotion_model.h5')

        # Load Emojis
        # set flag = -1 while loading emojis for alpha channel
        Angry = cv2.imread(
            "face expression/Emoji/angry.png", -1)
        Happy = cv2.imread(
            "face expression/Emoji/happy.png", -1)
        Neutral = cv2.imread(
            "face expression/Emoji/neutral.png", -1)
        Sad = cv2.imread(
            "face expression/Emoji/sad1.png", -1)
        Surprise = cv2.imread(
            "face expression/Emoji/surprise.png", -1)

        self.classes = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.class_images = [Angry, Happy, Neutral, Sad, Surprise]

    def image_input():
        '''
            Video_input - input module for the models  

            Varibles:
                mtcnn = MTCNN face detection module. 
                        'keep_all = True' for capturing all the faces in the frame .
                        'device = 'cuda'' set device 'cuda' for GPU and 'cpu' for use cpu for processing.
                        'margin = 80' margin around face detection bonding box.

                cap = Read the image.

                Width, Height = Width and Height of the output window

            returns cap, mtcnn

        '''
        mtcnn = MTCNN(keep_all=True, device='cpu', margin=40)

        cap = cv2.imread('face expression/test.jpg')

        return cap, mtcnn

    def open_bounding_box1(frame, x, y, x1, y1):  # Custom Bounding Box
        '''
            annotation_box - Annotation box creates parallelogram box which extends the hexagon bounding box

            Keyword Arguments:
                frame = Frame from the VideoCapture.
                x = Top-Left-Row of the bounding box.
                y = Top-Left-Column of the bounding box.
                x1 = Bottom-Right-Row co-ordinate of the bounding box.
                y1 = Bottom-Right-Column co-ordinate of the bounding box.

            Variables:
                outer_corner_radius = Corner radius of the outer box. 
                outer_extra_length = Length of line of outer box from the corner.
                inner_corner_radius = Corner radius of the inner box. 
                inner_extra_length = Length of line of inner box from the corner.
                total_radius = Combined radius of outer and inner box corner.
                total_extra = Combined length of outer and inner box line from the corner.
                outer_color = Outer box color.
                inner_color = Inner box color.
        '''

        outer_corner_radius = round((x1-x)/12)
        outer_extra_length = 2*outer_corner_radius
        inner_corner_radius = round(outer_corner_radius/2)
        outer_extra_length = 2*inner_corner_radius

        total_radius = inner_corner_radius + outer_corner_radius
        total_extra = outer_extra_length + outer_corner_radius

        outer_color = (106, 202, 46)
        # inner_color = (230, 224, 25)

        ## ---Outer Box---##

        # Top-Left outer
        cv2.ellipse(frame, center=(x+outer_corner_radius, y+outer_corner_radius), axes=(outer_corner_radius,
                    outer_corner_radius), angle=180, startAngle=0, endAngle=90, color=outer_color, thickness=3)
        cv2.line(frame, (x+outer_corner_radius, y),
                 (x+total_extra, y), outer_color, 3)
        cv2.line(frame, (x, y+outer_corner_radius),
                 (x, y+total_extra), outer_color, 3)

        # Bottom-Left outer
        cv2.ellipse(frame, center=(x+outer_corner_radius, y1-outer_corner_radius), axes=(outer_corner_radius,
                    outer_corner_radius), angle=90, startAngle=0, endAngle=90, color=outer_color, thickness=3)
        cv2.line(frame, (x+outer_corner_radius, y1),
                 (x+total_extra, y1), outer_color, 3)
        cv2.line(frame, (x, y1-outer_corner_radius),
                 (x, y1-total_extra), outer_color, 3)

        # Top-Right outer
        cv2.ellipse(frame, center=(x1-outer_corner_radius, y+outer_corner_radius), axes=(outer_corner_radius,
                    outer_corner_radius), angle=270, startAngle=0, endAngle=90, color=outer_color, thickness=3)
        cv2.line(frame, (x1-outer_corner_radius, y),
                 (x1-total_extra, y), outer_color, 3)
        cv2.line(frame, (x1, y+outer_corner_radius),
                 (x1, y+total_extra), outer_color, 3)

        # Bottom-Right outer
        cv2.ellipse(frame, center=(x1-outer_corner_radius, y1-outer_corner_radius), axes=(outer_corner_radius,
                    outer_corner_radius), angle=0, startAngle=0, endAngle=90, color=outer_color, thickness=3)
        cv2.line(frame, (x1-outer_corner_radius, y1),
                 (x1-total_extra, y1), outer_color, 3)
        cv2.line(frame, (x1, y1-outer_corner_radius),
                 (x1, y1-total_extra), outer_color, 3)

    def emotion_detetion(self):
        '''
        age_detection module for detect the age category from the face detected in the image

        Variables:
            image_cropped_list = cropped face matrix list from the image
            prob_list = probability of the detected matrix is the face
            boxes = 2D array of detected face co-ordinates
            x = Top-Left-Row of the bounding box.
            y = Top-Left-Column of the bounding box.
            x1 = Bottom-Right-Row co-ordinate of the bounding box.
            y1 = Bottom-Right-Column co-ordinate of the bounding box.
            scores = List of the accuracy of each class
            target = Predicted class based on the maximum accuracy of the class

        '''

        # Read image
        image = self.cap
        height = round(image.shape[1] // 3.5)
        width = round(image.shape[0] // 3.5)

        print(height, width)

        image = cv2.resize(image, (height, width))

        # Store face and face probability into variable
        image_cropped_list, prob_list = self.mtcnn(image, return_prob=True)

        if image_cropped_list is not None:

            faces, _ = self.mtcnn.detect(image)  # Detect face

            for i, prob in enumerate(prob_list):

                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

                if prob > 0.9:

                    face = faces[i]

                    # Set bounding box co-ordinate value in to the variables
                    x, y, x1, y1 = int(face[0]), int(
                        face[1]), int(face[2]), int(face[3])

                    # Crop only face part from whole image
                    match = image[y:y1, x:x1]

                    # Check x and y value should not be zero. If this value is zero that means some part of face/bounding box is out of image edge
                    if match.shape[0] != 0 and match.shape[1] != 0:

                        # Resize an image
                        data = cv2.resize(
                            match, (48, 48), interpolation=cv2.INTER_LINEAR)
                        data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
                        # Normalization an image
                        data = data.astype('float') / 255.0
                        data = img_to_array(data)
                        data = np.expand_dims(data, axis=0)  # Reshape a data

                        if np.sum([data]) != 0:
                            # prediction accuracy value for each class
                            scores = self.model.predict(data)

                            # max accuracy of the class
                            max_score = np.max(scores)

                        else:
                            pass

                        if max_score > 0.50:

                            # predicted class
                            target = np.argmax(scores, axis=1)[0]

                            # Draw rectangle over face
                            open_bounding_box1(image, x, y, x1, y1)

                            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
                            emoji = self.class_images[target]
                            h, w, c = emoji.shape

                            # division factor for emoji size
                            division = (x1-x)/800

                            # extended factor for y co-ordinate
                            extended_y = int((y1-y)/3.3)
                            # extended factor for x co-ordinate
                            extended_x = int((x1-x)/3)

                            h = int(h * division)
                            w = int(w * division)

                            if h > 75:
                                h = 75
                                w = 75

                            if h < 25:
                                h = 25
                                w = 25

                            emoji = cv2.resize(emoji, (w, h))

                            for i in range(0, w):
                                for j in range(0, h):
                                    if emoji[i, j][3] != 0:  # alpha 0
                                        image[y-extended_y + i, x +
                                              extended_x + j] = emoji[i, j]

        height = round(image.shape[1] * 2.0)
        width = round(image.shape[0] * 2.0)

        # print(height, width)

        image = cv2.resize(image, (height, width))  # Resize an image

        # Display output
        cv2.imshow("Emotion Detection", image)

        cv2.waitKey(0)

        cv2.destroyAllWindows()


cap, mtcnn = image_input()
emotion = Emotion(cap, mtcnn)
emotion.emotion_detetion()
