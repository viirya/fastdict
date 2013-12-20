cmake_minimum_required(VERSION 2.8.3)
FIND_PACKAGE(PythonInterp)
FIND_PACKAGE(PythonLibs)
FIND_PACKAGE(Boost COMPONENTS python serialization)

ENABLE_TESTING()
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
LINK_LIBRARIES(${Boost_LIBRARIES}) # Deprecated but so convenient!

PYTHON_ADD_MODULE(fastdict fastdict.cpp)
FILE(COPY fastdict.py fastdict_int.py DESTINATION .)
ADD_TEST(FastDict fastdict.py fastdict_int.py)
