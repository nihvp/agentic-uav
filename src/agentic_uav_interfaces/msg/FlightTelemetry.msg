std_msgs/Header header
bool armed
# PX4 navigation state name, e.g. OFFBOARD, AUTO_LAND, AUTO_RTL
string px4_nav_state
# action_node controller state, e.g. IDLE, TAKING_OFF, HOLDING, NAVIGATING
string controller_state
# Map frame (ENU)
geometry_msgs/Pose current_pose
geometry_msgs/Twist current_twist
# 0-100, NaN when no battery data has been received
float32 battery_percentage
bool geofence_ok
bool is_in_air
