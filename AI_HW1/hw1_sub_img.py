import sys
sys.path.append("/opt/ros/melodic/lib/python2.7/dist-packages")
import rospy # Python library for ROS
# from sensor_msgs.msg import Image # Image is the message type
from std_msgs.msg import String # String is the message type
# from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
# import cv2 # OpenCV library
 
def callback(msg):
    
    # Used to convert between ROS and OpenCV images
    # br = CvBridge()
 
    # Output debugging information to the terminal
    rospy.loginfo("receiving video frame")

    # Convert ROS Image message to OpenCV image
    #current_frame = br.imgmsg_to_cv2(data)
    current_frame = msg.data
    
    # Pring String MSG
    print(current_frame)
      
def receive_message():
    # Tells rospy the name of the node.
    # Anonymous = True makes sure the node has a unique name. Random
    # numbers are added to the end of the name. 
    rospy.init_node('string_sub_py', anonymous=True)
    
    # Node is subscribing to the video_frames topic
    rospy.Subscriber('string', String, callback)
    
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()
    
    # Close down the video stream when done
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    receive_message()