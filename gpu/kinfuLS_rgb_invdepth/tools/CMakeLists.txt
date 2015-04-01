if(OPENNI_FOUND)
  if(NOT VTK_FOUND)
    set(DEFAULT FALSE)
    set(REASON "VTK was not found.")
  else(NOT VTK_FOUND)
    set(DEFAULT TRUE)
    set(REASON)
    set(VTK_USE_FILE ${VTK_USE_FILE} CACHE INTERNAL "VTK_USE_FILE")
    include (${VTK_USE_FILE})
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
  endif(NOT VTK_FOUND)

  FILE(GLOB hdrs "*.h*")
  include_directories(${VTK_INCLUDE_DIRS})


  ## STANDALONE TEXTURE MAPPING
  set(the_target pcl_kinfuLS_rgb_texture_output)
  set(srcs standalone_texture_mapping.cpp )

  source_group("Source Files" FILES ${srcs} )

  PCL_ADD_EXECUTABLE_OPT_BUNDLE(${the_target} ${SUBSYS_NAME} ${srcs} ${hdrs})
  target_link_libraries(${the_target} pcl_common pcl_io ${OPENNI_LIBRARIES} pcl_visualization pcl_gpu_kinfuLS_rgb_invdepth pcl_kdtree pcl_features pcl_surface)
if(OpenCV_FOUND) 
        target_link_libraries(${the_target} ${OpenCV_LIBS}) 
endif() 
	
  ## KINECT FUSION
  set(the_target pcl_kinfuLS_rgb_invdepth)
  set(srcs kinfuRGBD_app.cpp capture.cpp evaluation.cpp)

  source_group("Source Files" FILES ${srcs} )

  PCL_ADD_EXECUTABLE_OPT_BUNDLE(${the_target} ${SUBSYS_NAME} ${srcs} ${hdrs})
  target_link_libraries(${the_target} pcl_common pcl_io ${OPENNI_LIBRARIES} pcl_visualization pcl_gpu_kinfuLS_rgb_invdepth pcl_octree)  
if(OpenCV_FOUND) 
        target_link_libraries(${the_target} ${OpenCV_LIBS}) 
endif()   
  ## STANDALONE MARCHING CUBES
  set(the_target pcl_kinfuLS_rgb_invdepth_mesh_output)
  set(srcs process_kinfuLS_rgb_invdepth_output.cpp)

  PCL_ADD_EXECUTABLE_OPT_BUNDLE(${the_target} ${SUBSYS_NAME} ${srcs} ${hdrs})
  target_link_libraries(${the_target} pcl_common pcl_io ${OPENNI_LIBRARIES} pcl_visualization pcl_gpu_kinfuLS_rgb_invdepth pcl_filters)
 if(OpenCV_FOUND) 
        target_link_libraries(${the_target} ${OpenCV_LIBS}) 
endif()  
  ## RECORD MAPS_RGB
  set(the_target pcl_record_kinect_maps_rgb_LSrgbd)
  set(srcs record_maps_rgb.cpp)

  PCL_ADD_EXECUTABLE_OPT_BUNDLE(${the_target} ${SUBSYS_NAME} ${srcs} ${hdrs})
  target_link_libraries(${the_target} pcl_common pcl_io ${OPENNI_LIBRARIES} pcl_visualization pcl_gpu_kinfuLS_rgb_invdepth pcl_filters)
   if(OpenCV_FOUND) 
        target_link_libraries(${the_target} ${OpenCV_LIBS}) 
endif()    

endif(OPENNI_FOUND)
