find_package(Torch REQUIRED)

file(GLOB SOURCES "*.cpp")
add_library(src SHARED ${SOURCES})
target_include_directories(src PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(src ${TORCH_LIBRARIES})

install(TARGETS src
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  PUBLIC_HEADER DESTINATION include
)
