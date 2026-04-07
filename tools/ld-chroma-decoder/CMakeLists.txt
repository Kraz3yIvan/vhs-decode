# ============================================================
# lddecode-chroma  (static library)
# ============================================================

# Required for M_PI on MSVC / MinGW
add_compile_definitions(_USE_MATH_DEFINES)

add_library(lddecode-chroma STATIC
    comb.cpp
    componentframe.cpp
    decoder.cpp
    decoderpool.cpp
    framecanvas.cpp
    monodecoder.cpp
    outputwriter.cpp
    palcolour.cpp
    sourcefield.cpp
    transformpal.cpp
    transformpal2d.cpp
    transformpal3d.cpp
)

# AUTOMOC needed: decoder.h, monodecoder.h etc. use Q_OBJECT
set_target_properties(lddecode-chroma PROPERTIES AUTOMOC ON)

target_include_directories(lddecode-chroma
    PUBLIC  .
    PRIVATE ${FFTW_INCLUDE_DIR}
)

target_link_libraries(lddecode-chroma
    PRIVATE
        Qt::Core
        ${FFTW_LIBRARY}
        lddecode-library
)

# ============================================================
# ONNX Runtime
#
# Tested with 1.16.3 (CUDA 11.8) and 1.19.2 (CUDA 12.2).
#
# Override at configure time with -DORT_ROOT=<path>, e.g.:
#   Linux:   -DORT_ROOT=/opt/onnxruntime-linux-x64-gpu-1.16.3
#   Windows: -DORT_ROOT=C:/ort/onnxruntime-win-x64-gpu-1.16.3
#
# To build with GPU support, add -DUSE_CUDA=ON at configure time.
# Without it, the CPU-only ORT package can be used instead.
# ============================================================
option(USE_CUDA "Enable NVIDIA GPU acceleration via CUDA 11.8" OFF)

if(NOT DEFINED ORT_ROOT)
    if(WIN32)
        if(USE_CUDA)
            set(ORT_ROOT "C:/ort/onnxruntime-win-x64-gpu-1.16.3"
                CACHE PATH "ONNX Runtime root directory")
        else()
            set(ORT_ROOT "C:/ort/onnxruntime-win-x64-1.16.3"
                CACHE PATH "ONNX Runtime root directory")
        endif()
    else()
        if(USE_CUDA)
            set(ORT_ROOT "/opt/onnxruntime-linux-x64-gpu-1.16.3"
                CACHE PATH "ONNX Runtime root directory")
        else()
            set(ORT_ROOT "/opt/onnxruntime-linux-x64-1.16.3"
                CACHE PATH "ONNX Runtime root directory")
        endif()
    endif()
endif()

message(STATUS "ONNX Runtime root: ${ORT_ROOT}")
message(STATUS "CUDA acceleration:  ${USE_CUDA}")

target_include_directories(lddecode-chroma PRIVATE "${ORT_ROOT}/include")
target_link_directories(lddecode-chroma    PRIVATE "${ORT_ROOT}/lib")
target_link_libraries(lddecode-chroma      PRIVATE onnxruntime)

# Pass USE_CUDA flag through to the C++ preprocessor so comb.cpp can see it
if(USE_CUDA)
    target_compile_definitions(lddecode-chroma PRIVATE USE_CUDA)
endif()

# On Linux, embed the .so search path so LD_LIBRARY_PATH is not needed
if(UNIX)
    set_target_properties(lddecode-chroma PROPERTIES
        BUILD_RPATH   "${ORT_ROOT}/lib"
        INSTALL_RPATH "${ORT_ROOT}/lib"
    )
endif()

# ============================================================
# ld-chroma-decoder  (executable)
# ============================================================

add_executable(ld-chroma-decoder
    decoder.cpp
    decoderpool.cpp
    main.cpp
    monodecoder.cpp
    ntscdecoder.cpp
    paldecoder.cpp
)

# AUTOMOC needed: ntscdecoder.h, paldecoder.h, decoder.h all use Q_OBJECT
set_target_properties(ld-chroma-decoder PROPERTIES AUTOMOC ON)

target_include_directories(ld-chroma-decoder PRIVATE ${FFTW_INCLUDE_DIR})

target_link_libraries(ld-chroma-decoder
    PRIVATE
        Qt::Core
        lddecode-library
        lddecode-chroma
)

install(TARGETS ld-chroma-decoder)

