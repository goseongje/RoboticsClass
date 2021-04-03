import rospy # Python library for ROS
from sensor_msgs.msg import Image # Image is the message type
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import sys
sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
import cv2 # OpenCV library
import time 

def publish_message():
    # Node is publishing to the video_frames topic using 
    # the message type Image
    pub = rospy.Publisher('image', Image, queue_size=10)
    # Tells rospy the name of the node.
    # Anonymous = True makes sure the node has a unique name. Random
    # numbers are added to the end of the name.
    rospy.init_node('image_pub_py', anonymous=True)
    # Go through the loop 10 times per second
    rate = rospy.Rate(10) # 10hz

    # Create a VideoCapture object
    cap = cv2.VideoCapture(0)
    print(cap)    
    assert cap.isOpened(), 'Cannot capture source'

    # Used to convert between ROS and OpenCV images
    br = CvBridge()

    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()

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
