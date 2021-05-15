import rospy # Python library for ROS
from sensor_msgs.msg import Image # Image is the message type
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import sys
sys.path.remove("/opt/ros/melodic/lib/python2.7/dist-packages")
import cv2 # OpenCV library
import time 


import torch
import torchvision.models as models
import torchvision.transforms as transforms

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(device)
model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True).to(device)
model.eval()
transforms_composed = transforms.Compose([
        transforms.ToTensor()])

def forward_model(img_arr):
    img_tensor = transforms_composed(img_arr).unsqueeze(0)
    outputs = model(img_tensor.to(device))[0]
    probs = outputs["scores"].cpu().detach().numpy()
    boxes = outputs["boxes"].cpu().detach().numpy()
    labels = outputs["labels"].cpu().detach().numpy()
    #masks = outputs["masks"].cpu().detach().numpy()
    
    keep = probs > 0.5
    probs = probs[keep]
    labels = labels[keep]
    boxes = boxes[keep]    
    #masks = masks[keep]

    draw_arr = draw_bboxes(img_arr, boxes, labels, probs)
    return draw_arr

def draw_bboxes(img_arr, boxes, labels, scores):
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img_arr, (x1, y1), (x2, y2), (0,255,0), 3)
        cv2.putText(img_arr, COCO_INSTANCE_CATEGORY_NAMES[int(label)], 
                    (x1+10, y1+10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
    return img_arr


def publish_message():
    # Node is publishing to the video_frames topic using 
    # the message type Image
    pub = rospy.Publisher('image', Image, queue_size=10)
    # Tells rospy the name of the node.
    # Anonymous = True makes sure the node has a unique name. Random
    # numbers are added to the end of the name.
    rospy.init_node('detection_pub_py3', anonymous=True)
    # Go through the loop 10 times per second
    rate = rospy.Rate(10) # 10hz

    # Create a VideoCapture object
    cap = cv2.VideoCapture(0)   # laptop cam (0) / usb 435i (2)
    assert cap.isOpened(), 'Cannot capture source'

    # Used to convert between ROS and OpenCV images
    br = CvBridge()

    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()

        frame = forward_model(frame)

        # While ROS is still running.
        if rospy.is_shutdown():
            print("ROS is not running")
            time.sleep(3)
            continue         
        # Print debugging information to the terminal
        rospy.loginfo('publishing video frame')
            
        # Publish the image.
        # The 'cv2_to_imgmsg' method converts an OpenCV
        # image to a ROS image message
        pub.publish(br.cv2_to_imgmsg(frame))
                
        # Sleep just enough to maintain the desired rate
        # rate.sleep()
        print("FPS is {:.2f}".format(1 / (time.time() - start_time)))
        
if __name__ == '__main__':
    try:
        publish_message()
    except rospy.ROSInterruptException:
        pass
