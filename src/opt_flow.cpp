#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "geometry_msgs/msg/twist.hpp"

#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/highgui.hpp> // <-- NEW: Required for imshow and waitKey

using std::placeholders::_1;

// <-- Restored the class definition!
class FarnebackFlowNode : public rclcpp::Node {
public:
  FarnebackFlowNode() : Node("farneback_optical_flow") {
    declare_parameter<std::string>("image_topic", "/camera/image_raw");
    declare_parameter<std::string>("output_topic", "/mavros/setpoint_velocity/cmd_vel_unstamped");
    declare_parameter<std::string>("frame_id", "camera_link");

    declare_parameter<double>("flow_to_rad_scale", 0.002);
    declare_parameter<int>("roi_margin_px", 40);
    declare_parameter<double>("lateral_gain", 0.5);
    declare_parameter<double>("forward_speed", 0.3);
    declare_parameter<double>("slowdown_threshold", 5.0);

    // Farneback params
    declare_parameter<double>("pyr_scale", 0.5);
    declare_parameter<int>("levels", 3);
    declare_parameter<int>("winsize", 15);
    declare_parameter<int>("iterations", 3);
    declare_parameter<int>("poly_n", 5);
    declare_parameter<double>("poly_sigma", 1.2);
    declare_parameter<int>("flags", 0);

    image_topic_ = get_parameter("image_topic").as_string();
    output_topic_ = get_parameter("output_topic").as_string();
    frame_id_ = get_parameter("frame_id").as_string();
    scale_ = get_parameter("flow_to_rad_scale").as_double();
    roi_margin_px_ = get_parameter("roi_margin_px").as_int();

    pyr_scale_ = get_parameter("pyr_scale").as_double();
    levels_ = get_parameter("levels").as_int();
    winsize_ = get_parameter("winsize").as_int();
    iterations_ = get_parameter("iterations").as_int();
    poly_n_ = get_parameter("poly_n").as_int();
    poly_sigma_ = get_parameter("poly_sigma").as_double();
    flags_ = get_parameter("flags").as_int();

    cmd_pub_ = create_publisher<geometry_msgs::msg::Twist>(output_topic_, 10);
    sub_ = create_subscription<sensor_msgs::msg::Image>(
      image_topic_, rclcpp::SensorDataQoS(),
      std::bind(&FarnebackFlowNode::on_image, this, _1));

    RCLCPP_INFO(get_logger(), "Subscribed: %s", image_topic_.c_str());
    RCLCPP_INFO(get_logger(), "Publishing: %s", output_topic_.c_str());
  }

private:
  void compute_and_publish_cmd(const cv::Mat& flow, const builtin_interfaces::msg::Time& /*stamp*/) {
    int w = flow.cols, h = flow.rows;
    int mid = w / 2;

    // Compute mean flow magnitude for left and right halves
    cv::Mat mag, angle;
    std::vector<cv::Mat> flow_xy;
    cv::split(flow, flow_xy);
    cv::cartToPolar(flow_xy[0], flow_xy[1], mag, angle);

    cv::Rect left_roi(0, 0, mid, h);
    cv::Rect right_roi(mid, 0, w - mid, h);

    double left_mag = cv::mean(mag(left_roi))[0];
    double right_mag = cv::mean(mag(right_roi))[0];
    double total_mag = (left_mag + right_mag) / 2.0;

    double lateral_gain = get_parameter("lateral_gain").as_double();
    double forward_speed = get_parameter("forward_speed").as_double();
    double slowdown_thresh = get_parameter("slowdown_threshold").as_double();

    geometry_msgs::msg::Twist cmd;

    // Lateral: steer away from the side with more flow
    double imbalance = right_mag - left_mag;
    cmd.angular.z = imbalance * lateral_gain;

    // Forward: slow down when overall flow is high (obstacle ahead)
    double speed_scale = std::max(0.0, 1.0 - (total_mag / slowdown_thresh));
    cmd.linear.x = forward_speed * speed_scale;

    cmd_pub_->publish(cmd);
    }
  void on_image(const sensor_msgs::msg::Image::SharedPtr msg) {
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvShare(msg, "mono8");
    } catch (const cv_bridge::Exception &e) {
      RCLCPP_ERROR(get_logger(), "cv_bridge: %s", e.what());
      return;
    }

    const cv::Mat& gray = cv_ptr->image;

    if (prev_gray_.empty()) {
      prev_gray_ = gray.clone();
      prev_stamp_ = msg->header.stamp;
      return;
    }

    rclcpp::Time t_now(msg->header.stamp);
    rclcpp::Time t_prev(prev_stamp_);
    double dt = (t_now - t_prev).seconds();
    if (dt <= 1e-6) dt = 1.0 / 30.0;

    cv::Mat flow; // CV_32FC2
    cv::calcOpticalFlowFarneback(
      prev_gray_, gray, flow,
      pyr_scale_, levels_, winsize_, iterations_, poly_n_, poly_sigma_, flags_);

    // --- Visualizing the Flow Vectors ---
    cv::Mat debug_frame;
    cv::cvtColor(gray, debug_frame, cv::COLOR_GRAY2BGR);
    
    int step = 16; 
    for (int y = 0; y < debug_frame.rows; y += step) {
      for (int x = 0; x < debug_frame.cols; x += step) {
        const cv::Point2f& fxy = flow.at<cv::Point2f>(y, x);
        
        if (cv::norm(fxy) > 1.0) {
          cv::line(debug_frame, cv::Point(x, y), 
                   cv::Point(cvRound(x + fxy.x), cvRound(y + fxy.y)),
                   cv::Scalar(0, 255, 0), 1);
          cv::circle(debug_frame, cv::Point(x, y), 1, cv::Scalar(0, 255, 0), -1);
        }
      }
    }

    cv::imshow("Optical Flow Vectors", debug_frame);
    cv::waitKey(1);
    // -----------------------------------------

    

    compute_and_publish_cmd(flow, msg->header.stamp);
    prev_gray_ = gray.clone();
    prev_stamp_ = msg->header.stamp;
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
  std::string image_topic_, output_topic_, frame_id_;
  double scale_;
  int roi_margin_px_;

  double pyr_scale_;
  int levels_, winsize_, iterations_, poly_n_, flags_;
  double poly_sigma_;

  cv::Mat prev_gray_;
  builtin_interfaces::msg::Time prev_stamp_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FarnebackFlowNode>());
  rclcpp::shutdown();
  return 0;
}