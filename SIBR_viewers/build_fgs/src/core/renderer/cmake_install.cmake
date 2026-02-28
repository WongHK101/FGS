# Install script for directory: D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/renderer

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
     "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/addshadow.frag;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/blur.frag;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/colored_mesh.frag;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/colored_mesh.vert;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/copy.frag;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/copy_depth.frag;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/depthRenderer.fp;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/depthRenderer.vp;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/emotive_relight.frag;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/emotive_relight.vert;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/hdrEnvMap.frag;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/hdrEnvMap.vert;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/longlat.gp;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/longlat.vp;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/longlatColor.fp;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/longlatDepth.fp;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/noproj.vert;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/normalRenderer.fp;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/normalRenderer.vp;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/normalRendererGen.gp;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/normalRendererGen.vp;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/poisson_diverg.frag;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/poisson_interp.frag;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/poisson_jacobi.frag;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/poisson_restrict.frag;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/positionReflectedDirRenderer.frag;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/positionReflectedDirRenderer.vert;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/positionRenderer.frag;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/positionRenderer.vert;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/shadowMapRenderer.fp;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/shadowMapRenderer.vp;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/texture-invert.frag;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/texture.frag;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/texture.vert;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/textured_mesh.frag;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/textured_mesh.vert;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core/textured_mesh_flipY.vert")
    if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
      message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
      message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    file(INSTALL DESTINATION "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/shaders/core" TYPE PROGRAM FILES
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/renderer/shaders/addshadow.frag"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/renderer/shaders/blur.frag"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/renderer/shaders/colored_mesh.frag"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/renderer/shaders/colored_mesh.vert"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/renderer/shaders/copy.frag"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/renderer/shaders/copy_depth.frag"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/renderer/shaders/depthRenderer.fp"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/renderer/shaders/depthRenderer.vp"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/renderer/shaders/emotive_relight.frag"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/renderer/shaders/emotive_relight.vert"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/renderer/shaders/hdrEnvMap.frag"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/renderer/shaders/hdrEnvMap.vert"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/renderer/shaders/longlat.gp"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/renderer/shaders/longlat.vp"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/renderer/shaders/longlatColor.fp"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/renderer/shaders/longlatDepth.fp"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/renderer/shaders/noproj.vert"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/renderer/shaders/normalRenderer.fp"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/renderer/shaders/normalRenderer.vp"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/renderer/shaders/normalRendererGen.gp"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/renderer/shaders/normalRendererGen.vp"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/renderer/shaders/poisson_diverg.frag"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/renderer/shaders/poisson_interp.frag"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/renderer/shaders/poisson_jacobi.frag"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/renderer/shaders/poisson_restrict.frag"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/renderer/shaders/positionReflectedDirRenderer.frag"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/renderer/shaders/positionReflectedDirRenderer.vert"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/renderer/shaders/positionRenderer.frag"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/renderer/shaders/positionRenderer.vert"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/renderer/shaders/shadowMapRenderer.fp"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/renderer/shaders/shadowMapRenderer.vp"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/renderer/shaders/texture-invert.frag"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/renderer/shaders/texture.frag"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/renderer/shaders/texture.vert"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/renderer/shaders/textured_mesh.frag"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/renderer/shaders/textured_mesh.vert"
      "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/core/renderer/shaders/textured_mesh_flipY.vert"
      )
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "sibr_renderer_install" OR NOT CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
     "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/lib/sibr_renderer.lib")
    if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
      message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
      message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    file(INSTALL DESTINATION "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/lib" TYPE STATIC_LIBRARY OPTIONAL FILES "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/build_fgs/src/core/renderer/sibr_renderer.lib")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "sibr_renderer_install" OR NOT CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
     "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/bin/sibr_renderer.dll")
    if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
      message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
      message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    file(INSTALL DESTINATION "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/bin" TYPE SHARED_LIBRARY FILES "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/build_fgs/src/core/renderer/sibr_renderer.dll")
    if(EXISTS "$ENV{DESTDIR}/D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/bin/sibr_renderer.dll" AND
       NOT IS_SYMLINK "$ENV{DESTDIR}/D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/bin/sibr_renderer.dll")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "CMAKE_STRIP-NOTFOUND" "$ENV{DESTDIR}/D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/bin/sibr_renderer.dll")
      endif()
    endif()
  endif()
endif()

