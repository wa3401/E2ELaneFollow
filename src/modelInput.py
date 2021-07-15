import rospy
from ryanModel import RyanModel
from sensor_msgs.msg import Image
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError
import torch
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped

rospy.init_node('e2eNode', anonymous=True)

bridge = CvBridge()

def image_callback(img_msg):

    hmin, hmax, smin, smax, vmin, vmax = 0, 179, 0, 255, 248, 255

    try:
        cv_image = bridge.imgmsg_to_cv2(img_msg, "passthrough")
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))

    startImage = cv_image[0:1280, 400:720]
    yuvImage = cv.cvtColor(startImage, cv.COLOR_BGR2YUV)
    yuvImage[:, :, 0] = cv.equalizeHist(yuvImage[:, :, 0])
    yuvImage[:, 0, :] = cv.equalizeHist(yuvImage[:, 0, :])
    yuvImage[0, :, :] = cv.equalizeHist(yuvImage[0, :, :])
    normalized = cv.cvtColor(yuvImage, cv.COLOR_YUV2RGB)
    hsvImage = cv.cvtColor(normalized, cv.COLOR_RGB2HSV)
    lower = (hmin, smin, vmin)
    upper = (hmax, smax, vmax)
    filter = cv.inRange(hsvImage, lower, upper)
    # edgeImage = cv.Canny(colorFilter, 100, 200)
    image = cv.resize(filter, (200, 75))

    speed = 0.75
    angle = model(image)
    drive_msg_std = AckermannDriveStamped()
    drive_msg_std.drive.speed = speed
    drive_msg_std.drive.steering_angle = angle
    drive_pub.publish(drive_msg_std)

model = RyanModel()
model.load_state_dict(torch.load('/home/williamanderson/LaneFollower/savedModel.pth'))

sub_image = rospy.Subscriber("/camera/color/image_raw", Image, image_callback)

drive_pub = rospy.Publisher("/vesc/low_level/ackermann_cmd_mux/input/navigation", AckermannDriveStamped, queue_size=10)

while not rospy.is_shutdown():
    rospy.spin()





    

