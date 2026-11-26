CXX := mpicxx
CXXFLAGS := -O3 -std=c++17 -Wall -Wextra -Iinclude

SRCS := src/main.cpp src/dist_csr.cpp src/test_utils.cpp src/tric_lite.cpp src/ditric.cpp
TARGET := tc_mpi

all: $(TARGET)

$(TARGET): $(SRCS) include/dist_graph.hpp include/test_utils.hpp
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET)

clean:
	rm -f $(TARGET)
