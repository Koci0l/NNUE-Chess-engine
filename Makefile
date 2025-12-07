CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -DNDEBUG -pthread

# Define target executable name
TARGET = engine-DD

# Find all .cpp files in src/ directory
SOURCES = $(wildcard src/*.cpp)

# Define object files (optional, or just compile directly)
OBJECTS = $(SOURCES:.cpp=.o)

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all clean