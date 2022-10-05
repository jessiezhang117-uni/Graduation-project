cmake_minimum_required(VERSION 2.8.3)

project(delta_drv90l_support)

find_package(catkin REQUIRED
    robot_state_publisher
    joint_state_publisher
    joint_state_publisher_gui)

catkin_package()

foreach(dir config launch meshes urdf)
   install(DIRECTORY ${dir}/
      DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/${dir})
endforeach()
