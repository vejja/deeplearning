# Compiler
CC = g++

# Compiler flags
CFLAGS = -std=c++14 -Wall

# Include directories for header files
INCLUDES = -I/opt/homebrew/Cellar/clblas/2.12_1/include/ -I/opt/homebrew/Cellar/sdl2/2.28.3/include/SDL2/

# Library directories
LIB_DIRS = -L/opt/homebrew/Cellar/clblas/2.12_1/lib/ -L/opt/homebrew/Cellar/sdl2/2.28.3/lib/

# Libraries to link
LIBS = -lclBLAS -lSDL2

# Source files
SRCS = main.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Executable name
EXEC = main

# Build target
all: $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $(EXEC) $(OBJS) $(LIB_DIRS) $(LIBS)

.cpp.o:
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJS) $(EXEC)
