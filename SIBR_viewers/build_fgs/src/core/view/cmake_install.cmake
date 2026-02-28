# Install script for directory: D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/view

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "CMAKE_OBJDUMP-NOTFOUND")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
     "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/alpha_colored_mesh.frag;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/alpha_colored_mesh.vert;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/alpha_colored_per_triangle_normals.geom;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/alpha_colored_per_triangle_normals.vert;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/alpha_colored_per_vertex_normals.geom;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/alpha_colored_per_vertex_normals.vert;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/alpha_points.frag;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/alpha_points.vert;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/alpha_uv_tex.frag;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/alpha_uv_tex_array.frag;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/alphaimgview.fp;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/alphaimgview.vp;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/anaglyph.fp;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/anaglyph.vp;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/axisgizmo.fp;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/axisgizmo.vp;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/camstub.fp;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/camstub.vp;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/depth.fp;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/depth.vp;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/depthonly.fp;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/depthonly.vp;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/image_viewer.frag;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/image_viewer.vert;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/mesh_color.fp;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/mesh_color.vp;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/mesh_debugview.fp;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/mesh_debugview.vp;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/mesh_normal.fp;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/mesh_normal.vp;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/number.fp;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/number.vp;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/skybox.fp;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/skybox.vp;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/text-imgui.fp;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/text-imgui.vp;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/texture.fp;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/texture.vp;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/topview.fp;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/topview.vp;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/uv_mesh.vert")
    if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
      message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
      message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    file(INSTALL DESTINATION "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core" TYPE PROGRAM FILES
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/view/shaders/alpha_colored_mesh.frag"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/view/shaders/alpha_colored_mesh.vert"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/view/shaders/alpha_colored_per_triangle_normals.geom"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/view/shaders/alpha_colored_per_triangle_normals.vert"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/view/shaders/alpha_colored_per_vertex_normals.geom"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/view/shaders/alpha_colored_per_vertex_normals.vert"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/view/shaders/alpha_points.frag"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/view/shaders/alpha_points.vert"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/view/shaders/alpha_uv_tex.frag"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/view/shaders/alpha_uv_tex_array.frag"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/view/shaders/alphaimgview.fp"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/view/shaders/alphaimgview.vp"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/view/shaders/anaglyph.fp"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/view/shaders/anaglyph.vp"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/view/shaders/axisgizmo.fp"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/view/shaders/axisgizmo.vp"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/view/shaders/camstub.fp"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/view/shaders/camstub.vp"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/view/shaders/depth.fp"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/view/shaders/depth.vp"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/view/shaders/depthonly.fp"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/view/shaders/depthonly.vp"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/view/shaders/image_viewer.frag"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/view/shaders/image_viewer.vert"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/view/shaders/mesh_color.fp"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/view/shaders/mesh_color.vp"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/view/shaders/mesh_debugview.fp"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/view/shaders/mesh_debugview.vp"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/view/shaders/mesh_normal.fp"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/view/shaders/mesh_normal.vp"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/view/shaders/number.fp"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/view/shaders/number.vp"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/view/shaders/skybox.fp"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/view/shaders/skybox.vp"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/view/shaders/text-imgui.fp"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/view/shaders/text-imgui.vp"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/view/shaders/texture.fp"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/view/shaders/texture.vp"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/view/shaders/topview.fp"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/view/shaders/topview.vp"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/view/shaders/uv_mesh.vert"
      )
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
     "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/lib/sibr_view.lib")
    if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
      message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
      message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    file(INSTALL DESTINATION "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/lib" TYPE STATIC_LIBRARY OPTIONAL FILES "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/build_fgs/src/core/view/sibr_view.lib")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
     "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/bin/sibr_view.dll")
    if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
      message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
      message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    file(INSTALL DESTINATION "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/bin" TYPE SHARED_LIBRARY FILES "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/build_fgs/src/core/view/sibr_view.dll")
    if(EXISTS "$ENV{DESTDIR}/D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/bin/sibr_view.dll" AND
       NOT IS_SYMLINK "$ENV{DESTDIR}/D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/bin/sibr_view.dll")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "CMAKE_STRIP-NOTFOUND" "$ENV{DESTDIR}/D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/bin/sibr_view.dll")
      endif()
    endif()
  endif()
endif()

