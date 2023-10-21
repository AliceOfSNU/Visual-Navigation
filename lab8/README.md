
# What is this lab about?

### Teddy Bear Localization and sparse reconstruction
In this lab, I used a YOLO-v3 based detection library [darknet](https://github.com/pjreddie/darknet) to
localize a teddy bear in the tum [rgbd teddy dataset](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download#freiburg3_teddy), given the camera poses and rgb images.

![Alt text](<Peek 2023-10-21 21-22.gif>)

I have used a separate node to perform pose graph optimization in real time. (pgo) This uses the gtsam library from lab7. 

** I will soon be extending this to triangulate ORB features to create a sparse point reconstruction of the teddy bear.**



# Installation

1. Build lab_8:
```
catkin build lab_8
```

Remember to source your workspace:
```
source {VNAV_HOME}/vnav_ws/devel/setup.bash
```

# Usage

1. For Teddy Bear localization, I have created a launch file.
```
 roslaunch lab_8 localization.launch
```
This will launch the main deliverable_2, the darknet server and the pgo backend.

2. You'll need to run the rviz separately. I have setup a .rviz file.
From another terminal, run

```
rviz -d rviz/deliverable_2.rviz
```