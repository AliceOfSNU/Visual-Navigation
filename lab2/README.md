# What is this lab about?

### Using the tf package and understanding transforms between frames

from drone1's perspective, which is circling around the world origin, the parabolic trajectory of drone2 becomes a ellipse.

![
](<Peek 2023-09-17 19-30.gif>)

## HOW TO RUN THIS PACKAGE:
suppose you have set up a catkin_workspace, copy the folders
- mav_comm
- two_drones_pkg

into the src folder, and run 
```
catkin build
```


To run,
```
roslaunch two_drones_pkg two_drones.launch
```