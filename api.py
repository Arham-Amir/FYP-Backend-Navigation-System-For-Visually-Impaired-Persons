import json
import cv2
import time
from ultralytics import YOLO
import IPython.display as ipd
import numpy as np
from flask import Flask, request, jsonify
import base64
import traceback

# Initialize YOLO model
model = YOLO("yolov8m.pt")
object_id_to_name = {
    0: 'فرد', 1: 'سائیکل', 2: 'گاڑی', 3: 'موٹرسائیکل', 4: 'ہوائی جہاز', 5: 'بس', 6: 'ٹرین', 7: 'ٹرک', 8: 'کشتی',
    9: 'ٹریفک لائٹ', 10: 'فائر ہائیڈرینٹ', 11: 'اسٹاپ سائن', 12: 'پارکنگ میٹر', 13: 'بینچ', 14: 'پرندہ',
    15: 'بلی', 16: 'کتا', 17: 'گھوڑا', 18: 'بھیڑ', 19: 'گائے', 20: 'ہاتھی', 21: 'ریچھ', 22: 'زیبرا', 23: 'زرافہ',
    24: 'بیک پیک', 25: 'چھائی', 26: 'ہینڈ بیگ', 27: 'ٹائی', 28: 'سوٹ کیس', 29: 'فرسبی', 30: 'اسکی', 31: 'برف بورڈ',
    32: 'کھیل', 33: 'پٹنگ', 34: 'بیس بال بیٹ', 35: 'بیس بال گلوو', 36: 'اسکیٹ بورڈ', 37: 'سرف بورڈ',
    38: 'ٹینس ریکٹ', 39: 'بوتل', 40: 'شراب کی گلاس', 41: 'کپ', 42: 'کانسی', 43: 'چھری', 44: 'چمچ', 45: 'کٹورا',
    46: 'کیلا', 47: 'سیب', 48: 'سینڈوچ', 49: 'مالٹا', 50: 'بروکلی', 51: 'گاجر', 52: 'ہوٹ ڈاگ', 53: 'پزا',
    54: 'ڈونٹ', 55: 'کیک', 56: 'کرسی', 57: 'کاؤچ', 58: 'پلانٹ پوٹ', 59: 'بیڈ', 60: 'ڈائننگ ٹیبل',
    61: 'ٹوائلٹ', 62: 'ٹی وی', 63: 'لیپ ٹاپ', 64: 'ماؤس', 65: 'ریموٹ', 66: 'کی بورڈ', 67: 'سیل فون',
    68: 'مائیکروویو', 69: 'افران', 70: 'ٹوسٹر', 71: 'سنک', 72: 'ریفریجریٹر', 73: 'کتاب', 74: 'گھڑی',
    75: 'گلدان', 76: 'قینچی', 77: 'ٹیڈی بیر', 78: 'بال کی ڈرائر', 79: 'ٹوتھ برش'
}
urdu_numbers_dict = {
    1: 'ایک',
    2: 'دو',
    3: 'تین',
    4: 'چار',
    5: 'پانچ',
    6: 'چھے',
    7: 'سات',
    8: 'آٹھ',
    9: 'نو'
}

ColisionHeight = 80
ClosenessHeight = 180
CenterThreshold = 70
MinDistanceThreshold = 2
dict = {}
dict["last"] = "streight"
dict["choice"] = -1

def object_distance(w, h):
  return ((2 * 3.14 * 180) / (w + h * 360) * 1000 + 3)
def detection(frame, model):
    results = model.track(frame, persist=True)
    shape = results[0].boxes.orig_shape
    cls = results[0].boxes.cls
    xywh = results[0].boxes.xywh
    ids = results[0].boxes.id
    result_array = [shape, cls, xywh, ids, results[0]]
    return result_array
def setStreight():
  if (dict["last"] != "streight"):
    dict["last"] = "streight"
    return True
  else:
    return False
def setSlow():
  if dict["last"] != "near":
    dict["last"] = "near"
    return True
  else:
    return False
def setStopAndCompleteTurn():
  if dict["last"] != "stop":
    dict["last"] = "stop"
    return True
  else:
    return False
def setStopLeft():
  if dict["last"] != "stop":
    dict["last"] = "stop"
    return True
  else:
    return False
def setStopRight():
  if dict["last"] != "stop":
    dict["last"] = "stop"
    return True
  else:
    return False
def extractClasses(ids, classes, filtered_listC):
  class_name = ""
  if len(filtered_listC) == 1:
    class_name = object_id_to_name[int(classes[filtered_listC[0]])]
  else:
    for i, dis in enumerate(filtered_listC):
      class_name += object_id_to_name[int(classes[dis])]
      class_name += " "
      class_name += urdu_numbers_dict[int(ids[dis])]
      if i != len(filtered_listC)-1:
        class_name += " , "
  return class_name

def setChoice(ch):
  if (ch == 3 and dict['choice'] == 4) or (ch == 4 and dict['choice'] == 3):
    dict['choice'] = 2
  elif ch > dict['choice']:
    dict['choice'] = ch

def checkColision(imageB, objB):
  imageW, imageH = imageB[1], imageB[0]
  left = int(objB[0] - objB[2] / 2)
  right = int(objB[0] + objB[2] / 2)
  top = int(objB[1] - objB[3] / 2)
  bottom = int(objB[1] + objB[3] / 2)

  centerLeft, centerRight = [imageW/2 - CenterThreshold, imageW/2 + CenterThreshold]
  if (not (left<centerLeft and right<centerLeft) and not (left>centerRight and right>centerRight)):
    if bottom > imageH - ColisionHeight and (left < centerRight or right > centerLeft): #check colision
      if left < centerLeft and right > centerRight:
        setChoice(2)
      elif left < centerLeft:
        setChoice(3)
      elif right > centerRight:
        setChoice(4)
      return True
    elif bottom > imageH - ClosenessHeight and bottom < imageH - ColisionHeight and (left < centerRight or right > centerLeft): #check closeness
      setChoice(1)
      return True
    else:
      setChoice(0)
      return False
  setChoice(0)
  return False

def guide_user(frame, model):
    shape, classes, boxes, ids, result = detection(frame, model)
    all_boxes_distance_list = []
    message = ""
    detection_list = []

    dict["choice"] == 0

    if ids is not None:
      for i, x in enumerate(classes):
       class_id = int(x.item())
       detection_list.append(f"Id: {int(ids[i].item())} {object_id_to_name[class_id]}")
      for i, obj in enumerate(boxes):
        w, h = obj[2], obj[3]
        distancei = object_distance(w, h)
        all_boxes_distance_list.append(distancei)
      min_distance = min(all_boxes_distance_list)

      #                         check agar 1 ya 1 se zyada cheezon se takranye lagye hain to
      collision = False
      filtered_listD = [num for num in all_boxes_distance_list if num <= min_distance + MinDistanceThreshold]
      filtered_listC = []
      for dis in filtered_listD:
        cols = checkColision(shape, boxes[all_boxes_distance_list.index(dis)])
        if cols:
          collision = True
          tempList = [index for index, value in enumerate(all_boxes_distance_list) if value == dis]
          filtered_listC.extend(tempList)

    # no hurdle wala case
    if dict["choice"] == 0:
      if setStreight():
        message = f"سیدھے چلتے جائیں"
        time.sleep(0.3)
      else:
        temp = {}
        temp["left"] = []
        temp["right"] = []
        for dis in filtered_listD:
          x = all_boxes_distance_list.index(dis)
          if int(ids[x]) in dict:
            dict[int(ids[x])] += 1
          else:
            dict[int(ids[x])] = 1
            print(boxes[x])
            if boxes[x][0] < shape[1] // 2:
                    temp["left"].append(x)
            elif boxes[x][0] > shape[1] // 2:
                    temp["right"].append(x)
        if temp["left"] != [] and temp["right"] != []:
          left_objects = extractClasses(ids, classes, temp["left"])
          right_objects = extractClasses(ids, classes, temp["right"])
          message = f"آپ بائیں جانب {left_objects} اور دائیں جانب {right_objects} کے قریب جا رہے ہیں"
        elif temp["left"] != []:
          left_objects = extractClasses(ids, classes, temp["left"])
          message = f"آپ بائیں جانب {left_objects} کے قریب جا رہے ہیں"
        elif temp["right"] != []:
          right_objects = extractClasses(ids, classes, temp["right"])
          message = f"آپ دائیں جانب {right_objects} کے قریب جا رہے ہیں"
    # near wala case
    elif dict["choice"] == 1 :
      if setSlow():
        class_name = extractClasses(ids, classes, filtered_listC)
        message = f"آگے {class_name} سے بچنے کے لئے آہستہ چلیں"
      dict["choice"] = -1
    # stop And Complete Turn wala case
    elif dict["choice"] == 2:
      if setStopAndCompleteTurn():
          class_name = class_name = extractClasses(ids, classes, filtered_listC)
          message = f"رک جائیں {class_name} سے بچنے کے لئے بائیں جانب جائیں  "
      dict["choice"] = -1
    # stop And Right Turn wala case
    elif dict["choice"] == 3:
      if setStopLeft():
        class_name = class_name = extractClasses(ids, classes, filtered_listC)
        message = f"رک جائیں {class_name} سے بچنے کے لئے تھوڑا دائیں جانب ہو جائیں  "
      dict["choice"] = -1
    # stop And Left Turn wala case
    elif dict["choice"] == 4:
      if setStopRight():
        class_name = class_name = extractClasses(ids, classes, filtered_listC)
        message = f"رک جائیں {class_name} سے بچنے کے لئے تھوڑا بائیں جانب ہو جائیں  "
      dict["choice"] = -1
    response = {
          'speech': message,
          'boxes': boxes.tolist(),
          'names': detection_list
      }

    return json.dumps(response), 200


app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload():
    try:
      base64_image = request.form.get('image')
      image_data = base64.b64decode(base64_image)
      nparr = np.frombuffer(image_data, np.uint8)
      image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
      rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

      # Call the guide_user function with the rotated image
      response = guide_user(rotated_image, model)
      print(response)

      return jsonify(response), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': 'An error occurred'}), 500

@app.route('/', methods=['POST'])
def hello_world():
    print('Hello World')
    return jsonify({'width': '10'}), 200

if __name__ == '__main__':
    app.run()
