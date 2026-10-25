.RECIPEPREFIX := >

CXX ?= g++

CXXFLAGS ?= -std=c++17 -O3 -flto=auto -march=x86-64-v3 -DNDEBUG -pthread -static 

PYTHON ?= python

# ============================================================================
# Network files
# ============================================================================

EVALFILE ?= quantised-v14.bin

# Big policy net.
POLICYFILE ?= Policy-big.bin

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

# Match runner: NNUE vs policy, with the quiescence-search switch.
ARENA ?= arena

SOURCES = $(wildcard src/*.cpp)

# Everything except the two entry points (main.cpp / uci.cpp / arena.cpp) is
# shared engine code. The arena does not link uci.cpp, so it never pulls in the
# (huge) embedded policy blobs -- it always loads the nets from disk at runtime.
ENGINE_SOURCES = $(filter-out src/main.cpp src/uci.cpp src/arena.cpp, $(SOURCES))

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
# NNUE vs Policy arena (loads both nets from disk, no embedding needed)
# ============================================================================

$(ARENA): $(ENGINE_SOURCES) src/arena.cpp
> $(CXX) $(CXXFLAGS) $(ENGINE_SOURCES) src/arena.cpp -o $(ARENA)

# ============================================================================
# Opening book (optional): make uho [UHO_BOOK=UHO_Lichess_4852_v1.epd]
# ============================================================================

UHO_BOOK ?= UHO_4060_v1.epd

uho:
> $(PYTHON) tools/fetch_uho.py $(UHO_BOOK)

# ============================================================================
# Clean
# ============================================================================

clean:
> rm -f $(TARGET) $(ARENA)

clean-embed:
> rm -f src/policy_embed.h src/policy_embed_small.h

.PHONY: all embed uho clean clean-embed