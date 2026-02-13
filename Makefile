CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=x86-64-v3 -mbmi2 -DNDEBUG -pthread

TARGET = Kociolek-2.0.exe

SOURCES = $(wildcard src/*.cpp)

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all clean