.RECIPEPREFIX := >

CXX ?= g++

CXXFLAGS ?= -std=c++17 -O3 -march=x86-64-v3 -mbmi2 -DNDEBUG -pthread -static

PYTHON ?= python

# ============================================================================
# Network files
# ============================================================================

EVALFILE ?= 768-1024x2-1-8.bin

# Big policy net.
POLICYFILE ?= quantised-dual-layer-hard-2048.bin

# Small 64hl policy net.
POLICYFILE_SMALL ?= quantised-64.bin

# Actual files used for embedding.
# Override these for OpenBench workers.
POLICY_EMBED_BIN       ?= $(POLICYFILE)
POLICY_SMALL_EMBED_BIN ?= $(POLICYFILE_SMALL)

# ============================================================================
# Policy hidden sizes
# ============================================================================

# Big net hidden layer size.
KOCIOLEK_POLICY_HL ?= 2048

# Small net hidden layer size.
POLICY_SMALL_HL ?= 64

# ============================================================================
# Defines
# ============================================================================

CXXFLAGS += -DEVALFILE='"$(EVALFILE)"'
CXXFLAGS += -DPOLICYFILE='"$(POLICYFILE)"'
CXXFLAGS += -DPOLICYFILE_SMALL='"$(POLICYFILE_SMALL)"'

CXXFLAGS += -DKOCIOLEK_POLICY_HL=$(KOCIOLEK_POLICY_HL)
CXXFLAGS += -DPOLICY_SMALL_HL=$(POLICY_SMALL_HL)

# ============================================================================
# Target
# ============================================================================

TARGET ?= $(or $(EXE), Kociolek-2.2.exe)

SOURCES = $(wildcard src/*.cpp)

all: $(TARGET)

# ============================================================================
# Embedded policy headers
# ============================================================================

src/policy_embed.h: $(POLICY_EMBED_BIN) tools/embed_policy.py
> $(PYTHON) tools/embed_policy.py $(POLICY_EMBED_BIN) src/policy_embed.h --name g_policy_embed

src/policy_embed_small.h: $(POLICY_SMALL_EMBED_BIN) tools/embed_policy.py
> $(PYTHON) tools/embed_policy.py $(POLICY_SMALL_EMBED_BIN) src/policy_embed_small.h --name g_policy_small_embed

embed: src/policy_embed.h src/policy_embed_small.h

# ============================================================================
# Binary
# ============================================================================

$(TARGET): $(SOURCES) src/policy_embed.h src/policy_embed_small.h
> $(CXX) $(CXXFLAGS) $(SOURCES) -o $(TARGET)

# ============================================================================
# Clean
# ============================================================================

clean:
> rm -f $(TARGET)

clean-embed:
> rm -f src/policy_embed.h src/policy_embed_small.h

.PHONY: all embed clean clean-embed