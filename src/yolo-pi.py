#!/usr/bin/env python
import colorsys
import os
import cv2 # using opencv 3
import sys
import random
import json
import datetime 
from kafka import KafkaProducer
import numpy as np
from keras import backend as K
from keras.models import load_model
from PIL import Image, ImageFont, ImageDraw
from yad2k.models.keras_yolo import yolo_eval, yolo_head


try:
    producer = KafkaProducer(bootstrap_servers='localhost:9092')
    
except Exception as e:
    print("Could not connect to kafka stream", e)
    sys.exit(1)


# using code from yad2k
def recognize_image(image, sess, boxes, scores, classes, is_fixed_size):
    cv2_im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(cv2_im)
    if is_fixed_size: 
        resized_image = image.resize(
            tuple(reversed(model_image_size)), Image.BICUBIC)
        image_data = np.array(resized_image, dtype='float32')
    else:
        new_image_size = (image.width - (image.width % 32),
                      image.height - (image.height % 32))
        #resized_image = cv2.resize(image, new_image_size)
        resized_image = image.resize(new_image_size, Image.BICUBIC)
        image_data = np.array(resized_image, dtype='float32')
        #print (image_data.shape)

    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)
    out_boxes, out_scores, out_classes = sess.run(
            [boxes, scores, classes],
            feed_dict={
                yolo_model.input: image_data,
                input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
    #print('Found {} boxes '.format(len(out_boxes)))
    

    font = ImageFont.truetype(
            font='font/FiraMono-Medium.otf',
            size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300
    json_data = []

    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        label = '{} {:.2f}'.format(predicted_class, score)
        json_data.append({"item" : predicted_class, "score" : str(score)})
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        #print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # My kingdom for a good redistributable image drawing library.
        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=colors[c])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=colors[c])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw
    # Data of what we have found. 
    val = json.dumps(json_data).encode()
    tkey = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
    producer.send('yolo', key=tkey.encode(), value=val)
    return image


sess = K.get_session()

model_path = 'model_data/tiny-yolo-voc.h5'
anchors_path = 'model_data/tiny-yolo-voc_anchors.txt'
classes_path = 'model_data/pascal_classes.txt'
#model_path = 'model_data/yolo.h5'
#anchors_path = 'model_data/yolo_anchors.txt'
#classes_path = 'model_data/coco_classes.txt'
with open(classes_path) as f:
        class_names = f.readlines()
class_names = [c.strip() for c in class_names]

with open(anchors_path) as f:
    anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)

yolo_model = load_model(model_path)
num_classes = len(class_names)
num_anchors = len(anchors)

model_output_channels = yolo_model.layers[-1].output_shape[-1]
assert model_output_channels == num_anchors * (num_classes + 5), \
        'Mismatch between model and given anchor and class sizes. ' \
        'Specify matching anchors and classes with --anchors_path and ' \
        '--classes_path flags.'
print('{} model, anchors, and classes loaded.'.format(model_path))

model_image_size = yolo_model.layers[0].input_shape[1:3]
is_fixed_size = model_image_size != (None, None)
# seems to return true most of the time.
#print(is_fixed_size)

# Generate colors for drawing bounding boxes.
hsv_tuples = [(x / len(class_names), 1., 1.)
              for x in range(len(class_names))]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(
    map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
        colors))
random.seed(10101)  # Fixed seed for consistent colors across runs.
random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
random.seed(None)  # Reset seed to default.

# Generate output tensor targets for filtered bounding boxes.
# TODO: Wrap these backend operations with Keras layers.
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
input_image_shape = K.placeholder(shape=(2, ))
boxes, scores, classes = yolo_eval(
    yolo_outputs,
    input_image_shape,
    score_threshold=.3,
    iou_threshold=.5)

#cv2.namedWindow("preview")
print("Starting video capture...")
vc = cv2.VideoCapture(0)
rval, frame = vc.read()
if rval == False:
    print("Can't read video capture.  Exiting.")
    sys.exit(1)

while True:
    if frame is not None:
        pil_image = recognize_image(frame, sess, boxes, scores, classes, is_fixed_size)
        # https://stackoverflow.com/questions/14134892/convert-image-from-pil-to-opencv-format
        open_cv_image = np.array(pil_image)
        # uncomment below if you want to see the image of the person.  
        #cv2.imshow("preview", open_cv_image)

    rval, frame = vc.read()

    i = cv2.waitKey(1)
    if i & 0xFF == ord('q'):
        sess.close()
        break

vc.release()
cv2.destroyAllWindows()


