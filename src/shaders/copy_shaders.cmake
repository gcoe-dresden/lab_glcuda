cmake_minimum_required(VERSION 2.8)

message(STATUS "${SOURCE_DIR}/shaders/ DESTINATION ${DESTINATION_DIR}/shaders/")
file( COPY ${SOURCE_DIR}/shaders/ DESTINATION ${DESTINATION_DIR}/shaders/ )
