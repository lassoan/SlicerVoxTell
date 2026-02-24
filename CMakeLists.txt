cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(VoxTell)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/lassoan/SlicerVoxTell")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Andras Lasso (PerkLab, Queen's University)")
set(EXTENSION_DESCRIPTION "VoxTell lets users segment 3D medical scans in 3D Slicer using plain-language prompts (for example, liver or right kidney).")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/lassoan/SlicerVoxTell/main/VoxTell.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/lassoan/SlicerVoxTell/main/screenshot01.jpg")
set(EXTENSION_DEPENDS "PyTorch")

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(VoxTell)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
