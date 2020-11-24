import os
import json
import cv2

# Opening JSON file
with open('result_20201122.json', 'r') as f:
    data = json.load(f)
    out_data = []
    for item in data:
        filename = item['filename']
        image = cv2.imread(filename)
        height, width, _ = image.shape
        out_element = {'bbox': [], 'score': [], 'label': []}
        for obj in item['objects']:
            label = int(obj['name'])
            center_x = obj['relative_coordinates']['center_x'] * width
            center_y = obj['relative_coordinates']['center_y'] * height
            box_width = obj['relative_coordinates']['width'] * width
            box_height = obj['relative_coordinates']['height'] * height
            box_top = center_y - box_height/2
            box_left = center_x - box_width/2
            box_bottom = center_y + box_height/2
            box_right = center_x + box_width/2
            score = obj['confidence']
            bbox = (box_top, box_left, box_bottom, box_right)
            out_element['bbox'].append(bbox)
            out_element['score'].append(score)
            out_element['label'].append(label)
        out_data.append(out_element)


ret = json.dumps(out_data)

with open('result_0856040.json', 'w') as fp:
    fp.write(ret)
