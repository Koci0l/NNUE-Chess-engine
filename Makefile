CXX ?= g++
CXXFLAGS ?= -std=c++17 -O3 -march=x86-64-v3 -mbmi2 -DNDEBUG -pthread -static

EVALFILE   ?= 768-1024x2-1-8.bin
POLICYFILE ?= quantised.bin

CXXFLAGS += -DEVALFILE='"$(EVALFILE)"'
CXXFLAGS += -DPOLICYFILE='"$(POLICYFILE)"'

TARGET ?= $(or $(EXE), Kociolek-2.1.3.exe)

SOURCES = $(wildcard src/*.cpp)

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all clean