import cv2
import torch
import numpy as np


class CollisionDetection:
    """Advanced driving safety monitoring system using computer vision"""

    def __init__(
        self, model_weights="yolov5s.pt", video_source="input.mp4"
    ):
        # Core parameters
        self.model_path = model_weights
        self.video_source = video_source
        self.output_file = "driver_safety_output.mp4"
        self.frame_dimensions = (1280, 720)

        # Detection settings
        self.detection_threshold = 0.4
        self.inference_dimensions = 320

        # Safety zone configuration
        self.attention_zone = np.array(
            [[400, 720], [400, 400], [870, 400], [870, 720]], dtype=np.int32
        )
        self.reference_object_width = 50  # cm
        self.camera_focal_length = 1000  # px

        # Alert system configuration
        self.alert_zones = 12
        self.alert_base_y = 600
        self.zone_spacing = 10

        # Initialize components
        self._init_detector()
        self._init_video_streams()
        self._calculate_safety_zones()

    def _init_detector(self):
        """Initialize the object detection model"""
        if torch.cuda.is_available():
            compute_device = torch.device("cuda")
            print("Using GPU for computation.")
        elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            compute_device = torch.device("mps")  
            print("Using Apple Silicon GPU (MPS) for computation.")
        else:
            compute_device = torch.device("cpu")
        
        print(f"Using device: {compute_device}")
        
        self.detector = torch.hub.load(
            "ultralytics/yolov5", "custom", path=self.model_path
        )
        self.detector.to(compute_device)

    def _init_video_streams(self):
        """Set up video input and output streams"""
        self.video_reader = cv2.VideoCapture(self.video_source)
        self.video_fps = self.video_reader.get(cv2.CAP_PROP_FPS)

        video_codec = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(
            self.output_file, video_codec, self.video_fps, self.frame_dimensions
        )

    def _calculate_safety_zones(self):
        """Initialize safety zone boundaries"""
        self.safety_lines = [
            self.alert_base_y + (i * self.zone_spacing) for i in range(self.alert_zones)
        ]

    def estimate_distance(self, object_width_px):
        """Calculate approximate distance to object"""
        return (
            self.reference_object_width * self.camera_focal_length
        ) / object_width_px

    def get_tracking_point(self, box_coords, frame_width):
        """Determine the appropriate tracking point based on object position"""
        x1, y1, x2, y2 = box_coords

        # For objects on left side of frame, track right bottom corner
        if x1 < frame_width / 2:
            return int(x2), int(y2)
        # For objects on right side, track left bottom corner
        else:
            return int(x1), int(y2)

    def get_alert_message(self, severity_level):
        """Get appropriate alert message based on severity level"""
        if 4 <= severity_level <= 5:
            return "FORWARD COLLISION WARNING"
        elif 6 <= severity_level <= 8:
            return "COLLISION WARNING SEVERE"
        elif 9 <= severity_level <= 11:
            return "PAY ATTENTION & TAKE CONTROL"
        elif severity_level >= 12:
            return "EMERGENCY STOPPING ..!!"
        return ""

    def process_video(self):
        """Process video frames with safety monitoring"""
        while True:
            # Read the next frame
            success, frame = self.video_reader.read()
            if not success:
                break

            # Resize frame to target dimensions
            frame = cv2.resize(frame, self.frame_dimensions)

            # Process the frame with safety monitoring
            processed_frame = self._analyze_frame(frame)

            # Display and save the processed frame
            # cv2.imshow("Safety Monitoring System", processed_frame)
            self.video_writer.write(processed_frame)

            # Exit if 'q' is pressed
            # if cv2.waitKey(1) & 0xFF == ord("q"):
            #     break

        # Release resources
        self.video_reader.release()
        self.video_writer.release()
        cv2.destroyAllWindows()

    def _analyze_frame(self, frame):
        """Analyze frame for potential safety hazards"""
        height, width = frame.shape[:2]

        # Draw center line and monitoring zone
        cv2.line(frame, (width // 2, 0), (width // 2, height), (160, 160, 160), 2)
        cv2.polylines(frame, [self.attention_zone], True, (0, 200, 0), 2)

        # Initialize safety line status
        line_status = [(255, 0, 0) for _ in range(self.alert_zones)]
        breached_zones = []

        # Draw initial safety lines
        for i, line_y in enumerate(self.safety_lines):
            start_point = (self.attention_zone[0][0], line_y)
            end_point = (self.attention_zone[2][0], line_y)
            cv2.line(frame, start_point, end_point, line_status[i], 2)

        # Perform object detection
        detections = self.detector(frame, size=self.inference_dimensions).xyxy[0]

        # Process each detected object
        for detection in detections:
            x1, y1, x2, y2, confidence, class_id = detection.cpu().numpy()

            # Skip low-confidence detections
            if confidence < self.detection_threshold:
                continue

            # Determine tracking point based on object position
            track_x, track_y = self.get_tracking_point((x1, y1, x2, y2), width)

            # Check if object is in monitoring zone
            in_attention_zone = (
                cv2.pointPolygonTest(self.attention_zone, (track_x, track_y), False) > 0
            )

            if in_attention_zone:
                # Mark tracking point
                cv2.circle(frame, (track_x, track_y), 5, (0, 255, 0), -1)

                # Check for safety zone breaches
                for i, line_y in enumerate(self.safety_lines):
                    if y2 >= line_y:
                        line_status[i] = (0, 0, 255)  # Set line to red
                        breached_zones.append(i + 1)

                # Calculate distance to object
                object_width = x2 - x1
                distance = self.estimate_distance(object_width)

                # Draw bounding box for monitored object (red)
                cv2.rectangle(
                    frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2
                )

                # Show distance information
                cv2.putText(
                    frame,
                    f"Dist: {distance:.2f} cm",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (0, 255, 255),
                    1,
                )
            else:
                # Draw bounding box for unmonitored object (blue)
                cv2.rectangle(
                    frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2
                )

        # Update safety lines with current status
        for i, (line_y, color) in enumerate(zip(self.safety_lines, line_status)):
            start_point = (self.attention_zone[0][0], line_y)
            end_point = (self.attention_zone[2][0], line_y)
            cv2.line(frame, start_point, end_point, color, 2)

        # Display alerts if safety zones were breached
        if breached_zones:
            # Show brake indicator
            cv2.putText(
                frame, "BRAKE", (975, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
            )

            # Show maximum breach level
            max_breach = max(breached_zones)
            cv2.putText(
                frame,
                str(max_breach),
                (1000, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

            # Display appropriate alert message
            alert_message = self.get_alert_message(max_breach)
            if alert_message:
                cv2.putText(
                    frame,
                    alert_message,
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

        return frame


# Run the safety monitoring system
if __name__ == "__main__":
    cd = CollisionDetection(video_source="input.mp4")
    cd.process_video()
