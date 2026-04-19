#include <memory>
#include <string>
#include <deque>
#include <utility>
#include <cmath>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "geometry_msgs/msg/twist.hpp"

#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/highgui.hpp>

using std::placeholders::_1;

// ---------------------------------------------------------------------------
// Minimal PID with anti-windup integral clamp and derivative on measurement.
// "Derivative on measurement" means we differentiate the measured signal
// (IMU angular velocity for steering, total_mag for speed) rather than the
// error, which avoids derivative kick when the setpoint changes suddenly.
// ---------------------------------------------------------------------------
struct Pid {
  double kp{0.0}, ki{0.0}, kd{0.0};
  double integral_limit{10.0};   // symmetric clamp on integrator output
  double output_limit{0.0};      // 0 = no output clamp

  double integral{0.0};
  double prev_measurement{0.0};
  bool   first{true};

  void reset() { integral = 0.0; prev_measurement = 0.0; first = true; }

  double update(double error, double measurement, double dt) {
    if (dt <= 0.0) return 0.0;

    integral += error * dt;
    integral = std::clamp(integral, -integral_limit, integral_limit);

    double derivative = 0.0;
    if (!first) {
      derivative = -(measurement - prev_measurement) / dt;
    }
    prev_measurement = measurement;
    first = false;

    double out = kp * error + ki * integral + kd * derivative;
    if (output_limit > 0.0) {
      out = std::clamp(out, -output_limit, output_limit);
    }
    return out;
  }
};

// ---------------------------------------------------------------------------

class FarnebackFlowNode : public rclcpp::Node {
public:
  FarnebackFlowNode() : Node("farneback_optical_flow") {
    declare_parameter<std::string>("image_topic", "/camera/image_raw");
    declare_parameter<std::string>("output_topic", "/mavros/setpoint_velocity/cmd_vel_unstamped");
    declare_parameter<std::string>("imu_topic", "/mavros/imu/data");
    declare_parameter<std::string>("frame_id", "camera_link");

    declare_parameter<double>("forward_speed", 0.3);
    declare_parameter<double>("slowdown_threshold", 5.0);
    declare_parameter<int>("frame_gap", 2);

    // IMU rotation-to-pixel scale factors (pixels per radian).
    // Default 0 = disabled. Tune per camera using focal length as a starting point.
    declare_parameter<double>("imu_yaw_scale",   0.0);
    declare_parameter<double>("imu_roll_scale",  0.0);
    declare_parameter<double>("imu_pitch_scale", 0.0);

    // Steering PID — drives left/right flow imbalance toward 0.
    // D term uses IMU yaw rate as measurement (derivative on measurement).
    declare_parameter<double>("steer_kp", 0.5);
    declare_parameter<double>("steer_ki", 0.05);
    declare_parameter<double>("steer_kd", 0.1);
    declare_parameter<double>("steer_integral_limit", 5.0);
    declare_parameter<double>("steer_output_limit",   1.0);  // rad/s clamp

    // Speed PID — drives corrected total flow magnitude toward 0 (no-obstacle target).
    // D term uses total_mag as measurement.
    declare_parameter<double>("speed_kp", 0.3);
    declare_parameter<double>("speed_ki", 0.02);
    declare_parameter<double>("speed_kd", 0.05);
    declare_parameter<double>("speed_integral_limit", 3.0);
    declare_parameter<double>("speed_output_limit",   1.0);  // m/s clamp

    // Farneback params
    declare_parameter<double>("pyr_scale", 0.5);
    declare_parameter<int>("levels", 3);
    declare_parameter<int>("winsize", 15);
    declare_parameter<int>("iterations", 3);
    declare_parameter<int>("poly_n", 5);
    declare_parameter<double>("poly_sigma", 1.2);
    declare_parameter<int>("flags", 0);

    // Unused legacy params kept so existing launch files don't break
    declare_parameter<double>("flow_to_rad_scale", 0.002);
    declare_parameter<int>("roi_margin_px", 40);

    image_topic_  = get_parameter("image_topic").as_string();
    output_topic_ = get_parameter("output_topic").as_string();
    frame_id_     = get_parameter("frame_id").as_string();

    pyr_scale_  = get_parameter("pyr_scale").as_double();
    levels_     = get_parameter("levels").as_int();
    winsize_    = get_parameter("winsize").as_int();
    iterations_ = get_parameter("iterations").as_int();
    poly_n_     = get_parameter("poly_n").as_int();
    poly_sigma_ = get_parameter("poly_sigma").as_double();
    flags_      = get_parameter("flags").as_int();

    // Wire PID gains from parameters (live-tunable via ros2 param set)
    sync_pid_gains();

    cmd_pub_ = create_publisher<geometry_msgs::msg::Twist>(output_topic_, 10);

    sub_ = create_subscription<sensor_msgs::msg::Image>(
      image_topic_, rclcpp::SensorDataQoS(),
      std::bind(&FarnebackFlowNode::on_image, this, _1));

    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
      get_parameter("imu_topic").as_string(), rclcpp::SensorDataQoS(),
      std::bind(&FarnebackFlowNode::on_imu, this, _1));

    RCLCPP_INFO(get_logger(), "Subscribed image: %s", image_topic_.c_str());
    RCLCPP_INFO(get_logger(), "Subscribed IMU:   %s",
      get_parameter("imu_topic").as_string().c_str());
    RCLCPP_INFO(get_logger(), "Publishing:       %s", output_topic_.c_str());
  }

private:
  void sync_pid_gains() {
    steer_pid_.kp              = get_parameter("steer_kp").as_double();
    steer_pid_.ki              = get_parameter("steer_ki").as_double();
    steer_pid_.kd              = get_parameter("steer_kd").as_double();
    steer_pid_.integral_limit  = get_parameter("steer_integral_limit").as_double();
    steer_pid_.output_limit    = get_parameter("steer_output_limit").as_double();

    speed_pid_.kp              = get_parameter("speed_kp").as_double();
    speed_pid_.ki              = get_parameter("speed_ki").as_double();
    speed_pid_.kd              = get_parameter("speed_kd").as_double();
    speed_pid_.integral_limit  = get_parameter("speed_integral_limit").as_double();
    speed_pid_.output_limit    = get_parameter("speed_output_limit").as_double();
  }

  void on_imu(const sensor_msgs::msg::Imu::SharedPtr msg) {
    imu_angular_x_ = msg->angular_velocity.x;
    imu_angular_y_ = msg->angular_velocity.y;
    imu_angular_z_ = msg->angular_velocity.z;
  }

  void compute_and_publish_cmd(const cv::Mat& flow, double dt) {
    // Re-read gains so ros2 param set takes effect without a restart
    sync_pid_gains();

    int w = flow.cols, h = flow.rows;
    int mid = w / 2;

    cv::Mat mag, angle;
    std::vector<cv::Mat> flow_xy;
    cv::split(flow, flow_xy);
    cv::cartToPolar(flow_xy[0], flow_xy[1], mag, angle);
    (void)angle;

    cv::Rect left_roi(0, 0, mid, h);
    cv::Rect right_roi(mid, 0, w - mid, h);

    double left_mag  = cv::mean(mag(left_roi))[0];
    double right_mag = cv::mean(mag(right_roi))[0];
    double total_mag = (left_mag + right_mag) / 2.0;

    // --- IMU drift corrections ---
    double yaw_correction   = imu_angular_z_ * dt * get_parameter("imu_yaw_scale").as_double();
    double roll_correction  = imu_angular_x_ * dt * get_parameter("imu_roll_scale").as_double();
    double pitch_correction = std::abs(imu_angular_y_ * dt *
                                get_parameter("imu_pitch_scale").as_double());

    double imbalance       = (right_mag - left_mag) - yaw_correction - roll_correction;
    double corrected_total = std::max(0.0, total_mag - pitch_correction);

    // --- Steering PID ---
    // Error: want zero imbalance (balanced flow = centred obstacle field).
    // Measurement for D term: imu_angular_z_ (yaw rate IS the derivative of heading error).
    double steer_cmd = steer_pid_.update(imbalance, imu_angular_z_, dt);

    // --- Speed PID ---
    // Error: how far total flow exceeds zero (more flow = more obstacle → slow down).
    // Measurement for D term: corrected_total itself.
    double forward_speed = get_parameter("forward_speed").as_double();
    double speed_error   = corrected_total;  // setpoint is 0; drive toward clear field
    double speed_reduction = speed_pid_.update(speed_error, corrected_total, dt);
    double speed_cmd       = std::max(0.0, forward_speed - speed_reduction);

    geometry_msgs::msg::Twist cmd;
    cmd.angular.z = steer_cmd;
    cmd.linear.x  = speed_cmd;

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

    int frame_gap = std::max(1, static_cast<int>(get_parameter("frame_gap").as_int()));

    frame_buf_.push_back({cv_ptr->image.clone(), msg->header.stamp});
    while (static_cast<int>(frame_buf_.size()) > frame_gap + 1) {
      frame_buf_.pop_front();
    }

    if (static_cast<int>(frame_buf_.size()) < frame_gap + 1) {
      return;
    }

    const cv::Mat& ref_frame  = frame_buf_.front().first;
    const cv::Mat& curr_frame = frame_buf_.back().first;

    rclcpp::Time t_now(frame_buf_.back().second);
    rclcpp::Time t_ref(frame_buf_.front().second);
    double dt = (t_now - t_ref).seconds();
    if (dt <= 1e-6) dt = static_cast<double>(frame_gap) / 30.0;

    cv::Mat flow;
    cv::calcOpticalFlowFarneback(
      ref_frame, curr_frame, flow,
      pyr_scale_, levels_, winsize_, iterations_, poly_n_, poly_sigma_, flags_);

    cv::Mat debug_frame;
    cv::cvtColor(curr_frame, debug_frame, cv::COLOR_GRAY2BGR);
    constexpr int step = 16;
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

    compute_and_publish_cmd(flow, dt);
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr   imu_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr  cmd_pub_;

  std::string image_topic_, output_topic_, frame_id_;

  double pyr_scale_;
  int    levels_, winsize_, iterations_, poly_n_, flags_;
  double poly_sigma_;

  double imu_angular_x_{0.0};
  double imu_angular_y_{0.0};
  double imu_angular_z_{0.0};

  Pid steer_pid_;
  Pid speed_pid_;

  std::deque<std::pair<cv::Mat, builtin_interfaces::msg::Time>> frame_buf_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FarnebackFlowNode>());
  rclcpp::shutdown();
  return 0;
}
