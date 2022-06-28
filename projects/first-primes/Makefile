# Binary target
TGTDIR  := bin
TARGET  ?= primes
TESTTGT  = test

# Compiler and linker
CC			:= clang++
LD			:= clang++
CCMACRO ?= 
LDOPTS	:= -Lbuild -L/opt/homebrew/lib -lomp
INCOPTS := -Iinclude -Itemplates -I/opt/homebrew/include -I../../common/include
CCFLAGS	:= -O3 -Xpreprocessor -fopenmp -Wall -Wextra -pedantic -Wno-sign-compare -std=c++20 $(CCMACRO)
 
# Code files
INCS     = $(shell find include -type f -name '*.hpp')
INCS     = $(shell find ../../common/include -type f -name '*.hpp')
SRCS		 = $(shell find src -type f -name '*.cpp') 
OBJS	   = $(patsubst src/%,build/%,$(SRCS:.cpp=.o))

# First rule
all: $(TARGET) | $(TGTDIR)
	@mv $(TARGET) $(TGTDIR)/

# Linking
$(TARGET): $(OBJS) 
	$(LD) $(LDOPTS) -o $@ $^

# Compiling
$(OBJS): build/%.o : src/%.cpp | build
	$(CC) $(CCFLAGS) $(INCOPTS) -c -o $@ $?

# Objects directory
build:
	mkdir -p build

# Target directory
bin:
	mkdir -p bin

# Run with arguments
run:
	@./$(TGTDIR)/$(TARGET) $(ARGUMENTS)

# Clean objects and binaries
clean:
	rm -f $(OBJS)

# Clean and make again
again:
	@make clean
	@make

# Diagnostic to show files
show:
	@echo "Sources: $(SRCS)"
	@echo "Includes: $(INCS)"
	@echo "Objects: $(OBJS)" 

.PHONY: all run clean again show
