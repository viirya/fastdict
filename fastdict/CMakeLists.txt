cmake_minimum_required(VERSION 2.8.3)
FIND_PACKAGE(PythonInterp)
FIND_PACKAGE(PythonLibs)
FIND_PACKAGE(Boost COMPONENTS python serialization) # math_tr1)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++0x")

ENABLE_TESTING()
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS} "/home/phd/viirya/bin/include")
LINK_LIBRARIES(${Boost_LIBRARIES} "/home/phd/viirya/bin/lib/libboost_numpy.so") # Deprecated but so convenient!

PYTHON_ADD_MODULE(fastdict fastdict.cpp) # unordered_map.hpp unordered_collections_save_imp.hpp unordered_collections_load_imp.hpp)
FILE(COPY fastdict.py fastdict_int.py fastdict_compress_int.py DESTINATION .)
ADD_TEST(FastDict fastdict.py fastdict_int.py fastdict_compress_int.py)
