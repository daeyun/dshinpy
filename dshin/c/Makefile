CC=gcc
CFLAGS=-O3 -fPIC -Wall
#CFLAGS=-Ofast -march=native -fPIC -Wall

INCLUDE_PATHS=-I/usr/local/include
LIB_PATHS=-L/usr/local/lib

UNAME := $(shell uname)
ifeq ($(UNAME), Linux)
TARGET=render_depth.linux.so
else
ifeq ($(UNAME), Darwin)
TARGET=render_depth.darwin.so
INCLUDE_PATHS=-I/opt/X11/include/
LIB_PATHS=-L/opt/X11/lib/
else
TARGET=render_depth.so
endif
endif

LIBS=-lOSMesa

default: $(TARGET)

%.darwin.so: %.c
	$(CC) $(CFLAGS) -shared $< -o $@ $(INCLUDE_PATHS) $(LIB_PATHS) $(LIBS) 

%.linux.so: %.c
	$(CC) $(CFLAGS) -shared $< -o $@ $(INCLUDE_PATHS) $(LIB_PATHS) $(LIBS) 

clean:
	rm -f $(TARGET) *.pyc

recompile:
	$(MAKE) clean
	$(MAKE)
