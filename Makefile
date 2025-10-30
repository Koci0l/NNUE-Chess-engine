CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -DNDEBUG -pthread
TARGET = engine-Time

all: main.cpp nnue.cpp
	$(CXX) $(CXXFLAGS) main.cpp nnue.cpp -o $(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all clean