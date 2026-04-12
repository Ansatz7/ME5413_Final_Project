# ME5413 Final Project — Autonomous Two-Floor Navigation

NUS ME5413 Autonomous Mobile Robotics Final Project AY25/26

![Ubuntu 20.04](https://img.shields.io/badge/OS-Ubuntu_20.04-informational?style=flat&logo=ubuntu&logoColor=white&color=2bbc8a)
![ROS Noetic](https://img.shields.io/badge/Tools-ROS_Noetic-informational?style=flat&logo=ROS&logoColor=white&color=2bbc8a)
![Python](https://img.shields.io/badge/Code-Python-informational?style=flat&logo=Python&logoColor=white&color=2bbc8a)
![C++](https://img.shields.io/badge/Code-C++-informational?style=flat&logo=c%2B%2B&logoColor=white&color=2bbc8a)

![cover_image](src/me5413_world/media/overview2526.png)

## Overview

A fully autonomous navigation system for a Jackal UGV completing a two-floor mission in Gazebo:

1. **Level 1 — Patrol & Count**: Autonomously patrol two rooms, count numbered boxes (1–9) using a custom YOLOv8 detector fused with 3D LiDAR depth.
2. **Ramp Traversal**: Exit Level 1, traverse a ramp to Level 2.
3. **Level 2 — Obstacle Avoidance & Final Positioning**: Select the passable exit (random cone placement), cruise the corridor scanning for target boxes, avoid a dynamic moving obstacle, and stop at the room containing the box number with the **fewest occurrences** from Level 1.

**Stack**: ROS Noetic · Gazebo 11 · `move_base` + TEB · `hdl_localization` (3D NDT) · YOLOv8

---

## Table of Contents

- [Repository Structure](#repository-structure)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Launch File Reference](#launch-file-reference)
- [System Architecture](#system-architecture)
  - [Node Graph](#node-graph)
  - [Coordinate System](#coordinate-system)
  - [Mission State Machine](#mission-state-machine)
- [Component Details](#component-details)
- [Configuration](#configuration)
- [Development & Testing Workflow](#development--testing-workflow)
- [License](#license)

---

## Repository Structure

```
ME5413_Final_Project/
├── src/
│   ├── me5413_world/                  # Main package
│   │   ├── launch/                    # All launch files
│   │   │   ├── world.launch           # Gazebo world + robot spawn (run this first)
│   │   │   ├── full_mission.launch    # ★ Complete autonomous mission
│   │   │   ├── level1_patrol.launch   # Level 1 patrol only
│   │   │   ├── level1_quick_check.launch  # Dev: skip to L1→L2 handoff
│   │   │   ├── level2_quick_check.launch  # Dev: teleport to L2, test L2 logic
│   │   │   ├── mapping.launch         # GMapping SLAM (for remapping)
│   │   │   ├── fast_lio_mapping.launch    # FAST-LIO SLAM (alternative)
│   │   │   ├── navigation.launch      # AMCL navigation (baseline)
│   │   │   ├── manual.launch          # Keyboard teleoperation
│   │   │   └── manual_yolo.launch     # Keyboard + YOLO debug view
│   │   ├── src/                       # Python nodes
│   │   │   ├── level1_patrol.py       # Level 1 autonomous patrol FSM
│   │   │   ├── auto_navigator.py      # Level 2 navigation FSM (main brain)
│   │   │   ├── box_counter.py         # YOLOv8 detection + 3D counting
│   │   │   ├── level1_quick_nav.py    # Dev: fast L1→L2 transition
│   │   │   ├── level2_quick_nav.py    # Dev: teleport to L2
│   │   │   └── object_spawner_gz_plugin.cpp  # Gazebo plugin: spawn cones/boxes
│   │   ├── config/
│   │   │   ├── costmap_common_params.yaml     # Obstacle/inflation layers
│   │   │   └── teb_local_planner_params.yaml  # TEB velocity/tolerance tuning
│   │   ├── maps/
│   │   │   ├── me5413_2d_map.pgm / .yaml      # 2D occupancy grid (0.05 m/px)
│   │   │   └── map_for_loc.pcd                # 3D point cloud for NDT localization
│   │   ├── models/
│   │   │   ├── number1/ … number9/            # Textured numbered box models
│   │   │   ├── box_detector.pt                # YOLOv8 trained weights (6.2 MB)
│   │   │   └── …                              # Bridge, buildings, cylinders
│   │   ├── worlds/
│   │   │   └── me5413_project_2526.world      # Gazebo SDF world definition
│   │   └── rviz/
│   │       └── hdl_locolization.rviz          # RViz display config
│   │
│   ├── interactive_tools/             # Custom RViz panel (respawn, goal)
│   ├── jackal_description/            # Modified Jackal URDF with custom sensors
│   ├── hdl_localization/              # 3D NDT localization (hdl_graph_slam)
│   ├── hdl_global_localization/       # Global localization service
│   ├── ndt_omp/                       # OpenMP-accelerated NDT
│   ├── fast_gicp/                     # Fast GICP point cloud registration
│   └── FAST_LIO_SLAM/                 # LiDAR-inertial odometry (optional SLAM)
```

---

## Dependencies

**System requirements:**
- Ubuntu 20.04
- ROS Noetic
- Python 3.8+, C++11+, CMake ≥ 3.0.2
- NVIDIA GPU recommended for real-time YOLOv8 inference

**ROS packages** (installed via `rosdep`):
- `move_base`, `navfn`, `teb_local_planner`
- `map_server`, `amcl`
- `roscpp`, `rospy`, `tf2`, `tf2_ros`, `tf2_geometry_msgs`
- `std_msgs`, `nav_msgs`, `geometry_msgs`, `sensor_msgs`, `visualization_msgs`
- `gazebo_ros`, `gazebo_plugins`
- `jackal_gazebo`, `jackal_navigation`, `jackal_control`
- `velodyne_simulator`, `pointcloud_to_laserscan`
- `rviz`, `jsk_rviz_plugins`
- `teleop_twist_keyboard`
- `pcl_ros`

**Python packages:**
```bash
pip install ultralytics opencv-python numpy
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/Ansatz7/ME5413_Final_Project.git
cd ME5413_Final_Project

# 2. Install ROS dependencies
rosdep install --from-paths src --ignore-src -r -y

# 3. Install any missing sensor drivers
sudo apt install -y \
  ros-noetic-sick-tim \
  ros-noetic-lms1xx \
  ros-noetic-velodyne-description \
  ros-noetic-pointgrey-camera-description \
  ros-noetic-jackal-control

# 4. Build the workspace
catkin_make
source devel/setup.bash

# 5. Install Gazebo models
mkdir -p ~/.gazebo/models

# Official Gazebo model library
git clone https://github.com/osrf/gazebo_models.git ~/gazebo_models
cp -r ~/gazebo_models/* ~/.gazebo/models/

# Our custom models (numbered boxes, bridge, etc.)
cp -r ~/ME5413_Final_Project/src/me5413_world/models/* ~/.gazebo/models/
```

> **YOLOv8 model weights** are included in the repository and require no separate download:
> - `src/me5413_world/models/box_detector.pt` — custom-trained digit detector (~6 MB)
> - `src/me5413_world/models/pretrained/yolov8n.pt` — YOLOv8n base weights (~6 MB)
> - `src/me5413_world/models/pretrained/yolo26n.pt` — alternative base weights (~5 MB)

> **Tip**: Add `source ~/ME5413_Final_Project/devel/setup.bash` to your `~/.bashrc` so you don't need to source it every terminal session.

---

## Quick Start

Every run requires two terminals. Terminal 1 stays open for the entire session.

```bash
# Terminal 1 — Gazebo simulation world (keep this running)
roslaunch me5413_world world.launch

# Terminal 2 — Full autonomous mission
roslaunch me5413_world full_mission.launch
```

The robot will:
1. Respawn all numbered boxes (8 s wait)
2. Patrol Level 1 — Room A → Room B → handoff point (~3–5 min)
3. Publish `/cmd_unblock` to open the exit gate, then traverse the ramp
4. Select the unobstructed exit on Level 2 via LiDAR scan
5. Cruise the Level 2 corridor, identify the target box number via YOLO
6. Avoid the dynamic moving obstacle, then stop at the correct room

---

## Launch File Reference

| Launch File | Purpose | Notes |
|---|---|---|
| `world.launch` | Gazebo simulation + robot spawn | Always run first |
| `full_mission.launch` | **Complete mission** (L1 patrol → ramp → L2 nav) | Primary launch |
| `level1_patrol.launch` | Level 1 patrol + box counting only | No L2 |
| `level1_quick_check.launch` | Skip patrol, run 2-point nav to handoff, trigger L2 | Dev / fast iteration |
| `level2_quick_check.launch` | Teleport directly to Level 2, start from exit selection | Dev / L2 testing |
| `mapping.launch` | GMapping SLAM for remapping | Use with `manual.launch` |
| `fast_lio_mapping.launch` | FAST-LIO LiDAR-inertial SLAM | Higher accuracy mapping |
| `navigation.launch` | AMCL navigation (2D baseline) | Not used in final mission |
| `manual.launch` | Keyboard teleoperation | Exploration / debug |
| `manual_yolo.launch` | Keyboard + YOLO debug image stream | Tune detection |

---

## System Architecture

### Node Graph

```
                     ┌─────────────────────────────────────────────────────┐
                     │                   world.launch                      │
                     │  Gazebo ──► /mid/points  /front/scan  /right/image  │
                     └────────────────────┬────────────────────────────────┘
                                          │
                     ┌────────────────────▼────────────────────────────────┐
                     │                full_mission.launch                  │
                     │                                                     │
  /mid/points ──────►│ velodyne_z_filter ──► /mid/points_filtered          │
                     │        │                      │                     │
                     │        ▼                      ▼                     │
                     │ hdl_localization         move_base                  │
                     │  (3D NDT)             (TEB local planner)           │
                     │  /tf (map→odom)        /move_base/goal              │
                     │        │                      │                     │
  /right/image ─────►│  box_counter.py ─────────────┤                     │
  /mid/points ──────►│  (YOLOv8 + depth)             │                     │
                     │  /me5413/box_count             │                     │
                     │  /me5413/yolo_raw              │                     │
                     │  /me5413/debug_image           │                     │
                     │        │                      │                     │
                     │        ▼                      │                     │
                     │  level1_patrol.py ────────────┤                     │
                     │  (waypoint FSM)               │                     │
                     │  /me5413/level1_done ─────────┤                     │
                     │  /cmd_unblock                 │                     │
                     │                               │                     │
                     │  auto_navigator.py ───────────┘                     │
                     │  (L2 FSM, waits for level1_done)                    │
                     └─────────────────────────────────────────────────────┘
```

### Coordinate System

The Gazebo world origin is offset from the ROS `map` frame origin:

```
map_coord = gazebo_coord + (22.5, 7.5)
```

| Location | Gazebo (x, y) | Map (x, y) |
|---|---|---|
| Robot spawn | (−22.5, −7.5) | (0, 0) |
| Level 1 Room A | — | x ∈ [4, 10], y ∈ [0, 15] |
| Level 1 Room B | — | x ∈ [14, 21], y ∈ [0, 15] |
| Ramp start | — | (10, −4) |
| Ramp end / L2 entry | — | (40.5, −3.3) |
| Level 2 corridor | — | x ∈ [26, 40], y ∈ [0, 15] |
| Level 2 exit wall | — | x ≈ 33.5 |

**Angular convention**: 0 = East, π/2 = North, π/−π = West, −π/2 = South (ROS standard).

**TF frame tree:**
```
world → map → odom → base_link → velodyne (3D LiDAR)
                                → front_laser (2D scan)
                                → camera_right
```

### Mission State Machine

```
                         [level1_done signal]
                                │
   START ──► L1 Patrol (13 wp) ─┤──► /cmd_unblock ──► Ramp climb
                                │
                         [level2_start signal]  ← (level2_quick_nav bypass)
                                │
                     ┌──────────▼──────────┐
                     │   Stage 4           │
                     │   Exit Selection    │
                     │  decision_point_1   │
                     │  left LiDAR ratio   │
                     │  < 5% → exit 1      │
                     │  ~30% → exit 2      │
                     └──────────┬──────────┘
                                │
                     ┌──────────▼──────────┐
                     │   Stage 5           │
                     │   Corridor YOLO     │
                     │  x=29.3, 4 stops    │
                     │  match target digit │
                     └──────────┬──────────┘
                                │
                     ┌──────────▼──────────┐
                     │   Stage 6           │
                     │   Dynamic Obstacle  │
                     │  wait for min dist  │
                     │  then enter room    │
                     └─────────────────────┘
```

---

## Component Details

### `level1_patrol.py` — Level 1 Patrol FSM

Navigates the robot through 13 waypoints covering Room A and Room B, then signals completion.

**Key behaviours:**
- Respawns numbered boxes at startup (waits 8 s for Gazebo to settle)
- Uses `move_base` SimpleActionClient with `early_stop_dist=0.4 m` for smooth corner turns
- Publishes `/me5413/level1_done` (`std_msgs/Bool`) at the handoff point
- Publishes `/cmd_unblock` (`std_msgs/Bool`) to remove the exit cone (10-second window)
- Visualises all waypoints as labelled spheres+arrows on `/patrol_waypoints`

**Patrol route (map frame):**

| # | Position (x, y) | Heading | Area |
|---|---|---|---|
| 0 | (3.5, 1.0) | North | Entry |
| 1 | (4.0, 16.0) | East | Room A upper-left |
| 2 | (11.5, 15.5) | South | Room A upper-right |
| 3 | (11.5, 7.5) | South | Room A mid-right |
| 4 | (13.0, 2.0) | NE | Corridor |
| 5 | (14.0, 16.0) | East | Room B upper-left |
| 6 | (21.0, 15.5) | South | Room B upper-right |
| 7 | (21.0, 1.0) | South | Room B lower-right |
| 8 | (19.0, −1.0) | West | Room B bottom |
| 9 | (13.5, −0.5) | North | Room B lower-left |
| 10 | (13.5, 7.5) | North | Room B mid |
| 11 | (12.0, 13.0) | SW | Corridor return |
| 12 | (11.0, −1.0) | West | Room A lower-right |
| **13** | **(8.0, −3.5)** | **South** | **Handoff / leave_level_1** |

---

### `box_counter.py` — YOLOv8 Detection & 3D Counting

Detects numbered boxes (1–9) from the right camera, associates detections with 3D positions from the Velodyne point cloud, and maintains a deduplicated count per digit.

**Inputs:**
- `/right/image_raw` — camera image (640×480)
- `/mid/points` — Velodyne 16-line 3D point cloud

**Outputs:**
- `/me5413/box_count` (`String`, JSON) — `{"1": 2, "3": 1, …}` running totals
- `/me5413/yolo_raw` (`String`, JSON) — digits seen in the current frame (no dedup)
- `/me5413/debug_image` — annotated image with bounding boxes, distance, and raw detections
- `/me5413/box_markers` — RViz MarkerArray with box positions and counts

**Detection pipeline:**
```
Camera frame
    │
    ▼ YOLOv8 inference (box_detector.pt, conf ≥ 0.9)
Bounding boxes (class = digit 1–9)
    │
    ▼ Depth association (Velodyne horizontal angle → point cloud column)
3D position (map frame, via TF)
    │
    ▼ Deduplication (DBSCAN clustering)
      same digit: merge if within 1.2 m
      any digit:  reject if within 0.4 m of existing detection
    │
    ▼ Confirmation (min 2 observations, ≥ 0.5 m apart)
Confirmed count entry
```

**Key parameters (configurable in launch):**

| Parameter | Default | Meaning |
|---|---|---|
| `min_conf` | 0.9 | YOLO confidence threshold |
| `min_obs` | 2 | Observations required before recording |
| `min_obs_dist` | 0.5 m | Min travel distance between observations |
| `dedup_same_digit` | 1.2 m | Merge radius for same digit |
| `dedup_any_digit` | 0.4 m | Reject radius for any digit (duplicate suppression) |
| `continuous_hz` | 2.0 Hz | Timed sampling rate |
| `move_dist_min` | 0.3 m | Spatial sampling interval |

---

### `auto_navigator.py` — Level 2 Navigation FSM

Waits for the Level 1 done signal, then executes a 6-stage state machine for Level 2.

**Subscriptions:**
- `/me5413/level1_done` — triggers start of ramp + L2 sequence
- `/me5413/level2_start` — alternative trigger (teleport test mode, skips stages 1–3)
- `/me5413/box_count` — real-time count updates (snapshots at L1 done)
- `/me5413/yolo_raw` — per-frame YOLO detections for corridor scanning
- `/front/scan` — 2D LiDAR for exit occupancy detection
- `/odometry/filtered` — velocity for stop detection

**Stages:**

| Stage | Description | Key method |
|---|---|---|
| 1 | Wait for `/me5413/level1_done` | — |
| 2 | Navigate to ramp entrance, publish `/cmd_unblock` | `send_goal` |
| 3 | Climb ramp (4 intermediate waypoints) | `send_goal` ×4 |
| **4** | **Exit selection** — face North, sample left-side (75–105°) LiDAR occupancy ratio; stable ratio < 5% → passable, ~30% → blocked | `_navigate_and_detect_exit` |
| **5** | **Corridor YOLO scan** — cruise x=29.3 corridor southward, sample `/me5413/yolo_raw` 2 s per stop | `_scan_corridor_for_target` |
| **6** | **Dynamic obstacle avoidance** — monitor right-side (−120° to −60°) LiDAR; wait for obstacle to pass its closest point then enter room | `_wait_obstacle_min_then_enter` |

**Exit detection detail (Stage 4):**

The robot stops at `decision_point_1` facing North (π/2). The left side (75–105°) points toward the exit. `_left_ratio()` counts the fraction of rays within 3 m:

- Sliding window of 5 samples, stable when max−min < 10%
- Mean < 5%  → exit is **open** (no cone)
- Mean 5–80% → exit is **blocked** (cone present), switch to exit 2
- Mean ≥ 80% → still scanning (wall, keep sampling)

**Dynamic obstacle avoidance detail (Stage 6):**

The robot parks at the corridor waypoint facing South (−π/2). The moving red cylinder travels along the east-west axis. Right-side laser (−120° to −60°) watches the approach:

```
State "waiting"    : obstacle > 3.0 m, do nothing
State "approaching": obstacle enters 3.0 m circle
                     → record min distance each tick
Break condition    : current_range > prev_min + 0.05 m
                     → obstacle has just passed the nearest point → go now
```

---

### `level1_quick_nav.py` — Fast L1→L2 Test Utility

Used by `level1_quick_check.launch`. Navigates directly to the handoff point (2 waypoints), then publishes a fake `box_count` + `/me5413/level1_done` to hand off to `auto_navigator` immediately. Useful for testing L2 logic without waiting for the full Level 1 patrol.

---

### `level2_quick_nav.py` — Level 2 Teleport Test Utility

Used by `level2_quick_check.launch`. Teleports the robot to Level 2 via `/gazebo/set_model_state`, publishes `/initialpose` (with correct z=2.6 for L2 floor height) three times to converge `hdl_localization`, waits 10 s, then fires `/me5413/level2_start`. This lets you test exit selection, YOLO corridor scanning, and dynamic obstacle avoidance in isolation.

---

## Configuration

### Costmap — `config/costmap_common_params.yaml`

| Parameter | Value | Notes |
|---|---|---|
| Sensor topic | `/mid/points_filtered` | z-filtered point cloud |
| `obstacle_range` | 2.0 m | Mark obstacles within this range |
| `raytrace_range` | 3.0 m | Clear free space up to this range |
| `inflation_radius` | 0.20 m | Safety margin around obstacles |
| `min/max_obstacle_height` | −100 / 100 m | Accept all heights (z-filter handles range) |
| `origin_z` | −1.0 m | Voxel layer base |
| `z_voxels` | 110 | Voxel layer height (covers both floors) |

### Local Planner — `config/teb_local_planner_params.yaml`

| Parameter | Value | Notes |
|---|---|---|
| `max_vel_x` | 0.6 m/s | Forward speed |
| `max_vel_x_backwards` | 0.2 m/s | |
| `max_vel_theta` | 1.0 rad/s | Rotation speed |
| `xy_goal_tolerance` | 0.3 m | Position tolerance |
| `yaw_goal_tolerance` | 2.5 rad | Effectively relaxed (robot turns after arriving) |
| `min_obstacle_dist` | 0.22 m | Footprint + small margin |
| `enable_homotopy_class_planning` | false | Disabled to reduce CPU load |

### Point Cloud Z-Filter (in launch files)

Velodyne point cloud is filtered to z ∈ [0.2, 2.0] m (sensor frame) before being fed to `hdl_localization` and `move_base`. This removes ground returns and ceiling noise, which is essential on Level 2 where the robot is elevated and raw z-values would confuse a non-filtered costmap.

---

## Development & Testing Workflow

### Testing Level 2 only (fastest iteration)

```bash
# Terminal 1
roslaunch me5413_world world.launch

# Terminal 2 — teleports to L2, runs exit selection + YOLO + obstacle avoidance
roslaunch me5413_world level2_quick_check.launch
```

### Testing Level 1 → Level 2 handoff (medium speed)

```bash
# Terminal 1
roslaunch me5413_world world.launch

# Terminal 2 — 2-point nav to handoff, then triggers full L2 sequence
roslaunch me5413_world level1_quick_check.launch
```

### Full mission end-to-end

```bash
# Terminal 1
roslaunch me5413_world world.launch

# Terminal 2
roslaunch me5413_world full_mission.launch
```

### Monitoring during a run

| Topic | What it shows |
|---|---|
| `/me5413/debug_image` | Live camera with YOLO bounding boxes, distance, and raw detections |
| `/me5413/box_count` | Running count JSON — useful to `rostopic echo` |
| `/auto_waypoints` | All L2 navigation waypoints in RViz (MarkerArray) |
| `/patrol_waypoints` | All L1 waypoints in RViz (MarkerArray) |
| `/me5413/box_markers` | Detected box positions in RViz |
| `/move_base/local_costmap/costmap` | Live local costmap (good for debugging obstacle avoidance) |

### Remapping

The pre-built maps (`me5413_2d_map.pgm` and `map_for_loc.pcd`) should work as-is. If you need to remap:

```bash
# GMapping (2D)
roslaunch me5413_world mapping.launch
# Drive the robot manually, then:
roscd me5413_world/maps/
rosrun map_server map_saver -f my_map map:=/map

# FAST-LIO (3D, higher accuracy)
roslaunch me5413_world fast_lio_mapping.launch
```

---

## License

Released under the [MIT License](LICENSE).

Original project template by [NUS Advanced Robotics Centre](https://github.com/NUS-Advanced-Robotics-Centre/ME5413_Final_Project).  
Full autonomous mission implementation by the AY25/26 team.
