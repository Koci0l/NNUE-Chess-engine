CXX ?= g++
CXXFLAGS ?= -std=c++17 -O3 -march=x86-64-v3 -mbmi2 -DNDEBUG -pthread -static

EVALFILE   ?= 768-1024x2-1-8.bin
POLICYFILE ?= quantised.bin

CXXFLAGS += -DEVALFILE='"$(EVALFILE)"'
CXXFLAGS += -DPOLICYFILE='"$(POLICYFILE)"'

TARGET ?= $(or $(EXE), Kociolek-2.1.3.exe)

SOURCES = $(wildcard src/*.cpp)

all: $(TARGET)

# Optional: regenerate embed when quantised.bin changes
# (OpenBench does not need this if policy_embed.h is committed)
src/policy_embed.h: quantised.bin tools/embed_policy.py
	python tools/embed_policy.py quantised.bin src/policy_embed.h

$(TARGET): $(SOURCES) src/policy_embed.h
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $(TARGET)

embed: src/policy_embed.h

clean:
	rm -f $(TARGET)

.PHONY: all clean embed