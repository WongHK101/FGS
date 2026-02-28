# Install script for directory: D:/dataset/FGS/FGS-0202v1/SIBR_viewers/src/projects/basic/apps/pointBased

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

if(CMAKE_INSTALL_COMPONENT STREQUAL "SIBR_PointBased_app_install" OR NOT CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
     "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/bin/SIBR_PointBased_app.exe")
    if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
      message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
      message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    file(INSTALL DESTINATION "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/bin" TYPE EXECUTABLE FILES "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/build_fgs/src/projects/basic/apps/pointBased/SIBR_PointBased_app.exe")
    if(EXISTS "$ENV{DESTDIR}/D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/bin/SIBR_PointBased_app.exe" AND
       NOT IS_SYMLINK "$ENV{DESTDIR}/D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/bin/SIBR_PointBased_app.exe")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "CMAKE_STRIP-NOTFOUND" "$ENV{DESTDIR}/D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/bin/SIBR_PointBased_app.exe")
      endif()
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "SIBR_PointBased_app_install" OR NOT CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
     "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/bin/msvcp140.dll;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/bin/msvcp140_1.dll;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/bin/msvcp140_2.dll;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/bin/msvcp140_atomic_wait.dll;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/bin/msvcp140_codecvt_ids.dll;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/bin/vcruntime140_1.dll;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/bin/vcruntime140.dll;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/bin/concrt140.dll")
    if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
      message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
      message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    file(INSTALL DESTINATION "D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/bin" TYPE FILE FILES
      "C:/Program Files (x86)/Microsoft Visual Studio/2019/Professional/VC/Redist/MSVC/14.29.30133/x64/Microsoft.VC142.CRT/msvcp140.dll"
      "C:/Program Files (x86)/Microsoft Visual Studio/2019/Professional/VC/Redist/MSVC/14.29.30133/x64/Microsoft.VC142.CRT/msvcp140_1.dll"
      "C:/Program Files (x86)/Microsoft Visual Studio/2019/Professional/VC/Redist/MSVC/14.29.30133/x64/Microsoft.VC142.CRT/msvcp140_2.dll"
      "C:/Program Files (x86)/Microsoft Visual Studio/2019/Professional/VC/Redist/MSVC/14.29.30133/x64/Microsoft.VC142.CRT/msvcp140_atomic_wait.dll"
      "C:/Program Files (x86)/Microsoft Visual Studio/2019/Professional/VC/Redist/MSVC/14.29.30133/x64/Microsoft.VC142.CRT/msvcp140_codecvt_ids.dll"
      "C:/Program Files (x86)/Microsoft Visual Studio/2019/Professional/VC/Redist/MSVC/14.29.30133/x64/Microsoft.VC142.CRT/vcruntime140_1.dll"
      "C:/Program Files (x86)/Microsoft Visual Studio/2019/Professional/VC/Redist/MSVC/14.29.30133/x64/Microsoft.VC142.CRT/vcruntime140.dll"
      "C:/Program Files (x86)/Microsoft Visual Studio/2019/Professional/VC/Redist/MSVC/14.29.30133/x64/Microsoft.VC142.CRT/concrt140.dll"
      )
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "SIBR_PointBased_app_install" OR NOT CMAKE_INSTALL_COMPONENT)
  set(target 						"SIBR_PointBased_app")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "SIBR_PointBased_app_install" OR NOT CMAKE_INSTALL_COMPONENT)
  set(inst_run_CONFIG_TYPE 			"Release")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "SIBR_PointBased_app_install" OR NOT CMAKE_INSTALL_COMPONENT)
  set(inst_run_INSTALL_FOLDER 		"D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "SIBR_PointBased_app_install" OR NOT CMAKE_INSTALL_COMPONENT)
  set(app	 						"D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/bin/SIBR_PointBased_app.exe")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "SIBR_PointBased_app_install" OR NOT CMAKE_INSTALL_COMPONENT)
  set(dirsToLookFor 				"D:/dataset/FGS/FGS-0202v1/SIBR_viewers/install/bin;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/boost/boost-1.71/lib64-msvc-14.1;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/assimp/Assimp-4.1.0/lib64;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/glew/glew-2.0.0/lib64;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/opencv/opencv-4.5.0/build/x64/vc16/bin;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/ffmpeg/lib;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/boost/boost-1.71;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/assimp/Assimp-4.1.0;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/glew/glew-2.0.0;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/ffmpeg/bin;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/glew/glew-2.0.0/lib;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/glew/glew-2.0.0/bin;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/assimp/Assimp-4.1.0/lib;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/assimp/Assimp-4.1.0/bin;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/ffmpeg/bin/lib;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/ffmpeg/bin/lib64;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/ffmpeg/bin/bin;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/ffmpeg/doc;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/ffmpeg/doc/lib;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/ffmpeg/doc/lib64;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/ffmpeg/doc/bin;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/ffmpeg/examples;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/ffmpeg/examples/lib;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/ffmpeg/examples/lib64;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/ffmpeg/examples/bin;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/ffmpeg/include;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/ffmpeg/include/lib;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/ffmpeg/include/lib64;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/ffmpeg/include/bin;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/ffmpeg/lib/lib;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/ffmpeg/lib/lib64;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/ffmpeg/lib/bin;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/ffmpeg/presets;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/ffmpeg/presets/lib;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/ffmpeg/presets/lib64;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/ffmpeg/presets/bin;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/embree3/embree-3.6.1.x64.vc14.windows;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/embree3/embree-3.6.1.x64.vc14.windows/lib;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/embree3/embree-3.6.1.x64.vc14.windows/lib64;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/embree3/embree-3.6.1.x64.vc14.windows/bin;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/eigen3/eigen;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/eigen3/eigen/lib;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/eigen3/eigen/lib64;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/eigen3/eigen/bin;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/boost/boost-1.71/lib;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/boost/boost-1.71/lib64;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/boost/boost-1.71/bin;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/boost;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/nativefiledialog/nfd;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/nativefiledialog/nfd/lib;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/nativefiledialog/nfd/lib64;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/nativefiledialog/nfd/bin;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/opencv/opencv-4.5.0;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/opencv/opencv-4.5.0/lib;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/opencv/opencv-4.5.0/lib64;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/opencv/opencv-4.5.0/bin;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/glfw/glfw-3.2.1;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/glfw/glfw-3.2.1/lib;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/glfw/glfw-3.2.1/lib64;D:/dataset/FGS/FGS-0202v1/SIBR_viewers/extlibs/glfw/glfw-3.2.1/bin")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "SIBR_PointBased_app_install" OR NOT CMAKE_INSTALL_COMPONENT)
  			if("${CMAKE_INSTALL_CONFIG_NAME}" STREQUAL "${inst_run_CONFIG_TYPE}")
				message(STATUS "Installing ${target} dependencies...")

				file(GET_RUNTIME_DEPENDENCIES
					EXECUTABLES ${app}
					RESOLVED_DEPENDENCIES_VAR _r_deps
					UNRESOLVED_DEPENDENCIES_VAR _u_deps
					CONFLICTING_DEPENDENCIES_PREFIX _c_deps
					DIRECTORIES ${dirsToLookFor}
					PRE_EXCLUDE_REGEXES "api-ms-*"
					POST_EXCLUDE_REGEXES ".*system32/.*\\.dll" ".*SysWOW64/.*\\.dll"
				)
			
				if(_u_deps)
					message(WARNING "There were unresolved dependencies for executable ${EXEC_FILE}: \"${_u_deps}\"!")
				endif()
				if(_c_deps_FILENAMES)
					message(WARNING "There were conflicting dependencies for executable ${EXEC_FILE}: \"${_c_deps_FILENAMES}\"!")
				endif()
			
				foreach(_file ${_r_deps})
					file(INSTALL
					DESTINATION "${inst_run_INSTALL_FOLDER}/bin"
					TYPE SHARED_LIBRARY
					FOLLOW_SYMLINK_CHAIN
					FILES "${_file}"
				)
				endforeach()
			endif()
		
endif()

