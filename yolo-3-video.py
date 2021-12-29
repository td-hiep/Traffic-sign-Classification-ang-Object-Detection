import numpy as np
import cv2
import time

video = cv2.VideoCapture('videos/IMG_6389.MOV')

writer = None

h, w = None, None

with open('yolo-coco-data/coco.names') as f:

    labels = [line.strip() for line in f]

network = cv2.dnn.readNetFromDarknet('yolo-coco-data/yolov3-tiny.cfg',
                                     'yolo-coco-data/yolov3-tiny.weights')

layers_names_all = network.getLayerNames()

layers_names_output = \
    [layers_names_all[i - 1] for i in network.getUnconnectedOutLayers()]

probability_minimum = 0.5

threshold = 0.3

colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

f = 0
t = 0

while True:
    ret, frame = video.read()

    if not ret:
        break

    if w is None or h is None:
        h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)

    network.setInput(blob)  # setting blob as input to the network
    start = time.time()
    output_from_network = network.forward(layers_names_output)
    end = time.time()

    f += 1
    t += end - start

    bounding_boxes = []
    confidences = []
    class_numbers = []

    for result in output_from_network:
        for detected_objects in result:
            scores = detected_objects[5:]
            class_current = np.argmax(scores)
            confidence_current = scores[class_current]
            if confidence_current > probability_minimum:
                box_current = detected_objects[0:4] * np.array([w, h, w, h])

                x_center, y_center, box_width, box_height = box_current
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))

                bounding_boxes.append([x_min, y_min,
                                       int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                class_numbers.append(class_current)


    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                               probability_minimum, threshold)

    if len(results) > 0:
        for i in results.flatten():
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

            colour_box_current = colours[class_numbers[i]].tolist()
            cv2.rectangle(frame, (x_min, y_min),
                          (x_min + box_width, y_min + box_height),
                          colour_box_current, 2)
            text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
                                                   confidences[i])

            cv2.putText(frame, text_box_current, (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)

    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        writer = cv2.VideoWriter('videos/result_IMG_6387.MOV', fourcc, 30,
                                 (frame.shape[1], frame.shape[0]), True)

    writer.write(frame)

print()
print('Total number of frames', f)
print('Total amount of time {:.5f} seconds'.format(t))
print('FPS:', round((f / t), 1))

video.release()
writer.release()
