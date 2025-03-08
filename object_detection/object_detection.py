import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import Twist, PoseStamped
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy
from cv_bridge import CvBridge
import cv2
import torch
import time
import torchvision
from torchvision.transforms import ToTensor
from ultralytics import YOLO
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
import math 
from rclpy.executors import MultiThreadedExecutor


class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')

        # Create QoS profile for pose subscription
        pose_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscribe to robot position (AMCL pose) with improved QoS
        self.pose_subscription = self.create_subscription(
            PoseWithCovarianceStamped, 
            '/amcl_pose', 
            self.pose_callback, 
            pose_qos_profile
        )

        # Subscribe to camera feed
        self.subscription = self.create_subscription(Image, '/camera/image_raw', self.listener_callback, 10)

        # Publisher for detection images
        self.publisher_ = self.create_publisher(Image, '/detection_image', 10)

        self.bridge = CvBridge()

        # Load YOLOv8s model (optimized for performance)
        self.model = YOLO('yolov8s.pt')
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize Qdrant client
        self.client = QdrantClient(url="http://localhost:6333")

        # Store the latest robot position
        self.robot_position = (0.0, 0.0, 0.0)

        self.get_logger().info("YOLOv8 Object Detection Node Initialized.")

    def pose_callback(self, msg):
        # Extract x, y and yaw from the pose
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        
        # Convert quaternion to yaw
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        # Update robot position
        self.robot_position = (x, y, yaw)
        
        # Log the updated pose (optional, can be removed in production)
        self.get_logger().info(f"Updated robot pose: x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}")

    def listener_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {str(e)}")
            return

        # Run YOLOv8 detection
        results = self.model(cv_image, verbose=False)[0]

        # Draw bounding boxes
        cv_image = self.draw_bboxes(cv_image, results)

        # Store detection results in Qdrant
        self.store_in_qdrant(results, cv_image)

        # Publish detection image
        try:
            detection_image = self.bridge.cv2_to_imgmsg(cv_image, 'bgr8')
            self.publisher_.publish(detection_image)
        except Exception as e:
            self.get_logger().error(f"Failed to convert and publish image: {str(e)}")

    def draw_bboxes(self, image, results, score_threshold=0.3):
        """ Draw bounding boxes around detected objects """
        for result in results.boxes:
            conf = result.conf[0].item()
            label_id = int(result.cls[0].item())

            if conf >= score_threshold:
                x1, y1, x2, y2 = map(int, result.xyxy[0])
                label_name = self.model.names[label_id]
                self.get_logger().info(f"Detected: {label_name} ({conf:.2f})")

                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw label
                cv2.putText(image, f"{label_name} {conf:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        return image

    def store_in_qdrant(self, results, image):

        # Collection name
        collection_name = "object_detections_ii"

        # Check if collection exists, create if not
        try:
            # Try to get collection info
            collection_info = self.client.get_collection(collection_name=collection_name)
        except Exception as e:
            # Collection doesn't exist, so create it
            try:
                # Configure vector size based on ResNet50 embedding
                resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
                resnet.fc = torch.nn.Identity()
                
                # Create a dummy embedding to get vector size
                with torch.no_grad():
                    dummy_img = torch.rand(1, 3, 224, 224)
                    vector_size = resnet(dummy_img).squeeze().numpy().shape[0]

                # Create collection with vector configuration
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        "size": vector_size,
                        "distance": "Cosine"  # Cosine similarity for image embeddings
                    }
                )
                self.get_logger().info(f"Created new collection: {collection_name}")
            except Exception as create_error:
                self.get_logger().error(f"Failed to create collection: {str(create_error)}")
                return

        timestamp = time.time()

        # Extract feature embeddings using ResNet50
        resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        resnet.fc = torch.nn.Identity()  # Remove final classification layer
        resnet.eval()

        # Prepare image tensor for embedding
        with torch.no_grad():
            img_tensor = ToTensor()(image).unsqueeze(0)
            img_embedding = resnet(img_tensor).squeeze().numpy()

        # Collect detections for batch upsert
        detection_points = []

        for result in results.boxes:
            conf = result.conf[0].item()
            label_id = int(result.cls[0].item())

            if conf >= 0.3:
                x1, y1, x2, y2 = map(int, result.xyxy[0])
                label_name = self.model.names[label_id]

                # Create a detection point for Qdrant
                detection_point = PointStruct(
                    id=int(timestamp * 1000) + len(detection_points),  # Unique ID
                    vector=img_embedding.tolist(),  # Use the same image embedding for all detections in the frame
                    payload={
                        "timestamp": timestamp,
                        "detection": {
                            "label": label_name,
                            "confidence": conf,
                            "bounding_box": {
                                "x1": x1, 
                                "y1": y1, 
                                "x2": x2, 
                                "y2": y2
                            }
                        },
                        "robot_position": {
                            "x": self.robot_position[0], 
                            "y": self.robot_position[1],
                            "yaw": self.robot_position[2]
                        },
                        "frame_metadata": {
                            "total_detections": len(results.boxes),
                            "detection_threshold": 0.3
                        }
                    }
                )
                detection_points.append(detection_point)

        # Batch upsert all detections
        if detection_points:
            try:
                self.client.upsert(
                    collection_name=collection_name,
                    points=detection_points
                )
                self.get_logger().info(f"Stored {len(detection_points)} detections in Qdrant.")
            except Exception as e:
                self.get_logger().error(f"Failed to store detections in Qdrant: {str(e)}")
        else:
            self.get_logger().info("No detections above threshold to store.")

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()

    # Create a MultiThreadedExecutor with the desired number of threads
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()