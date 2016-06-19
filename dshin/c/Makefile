CC=gcc
CFLAGS=-O3 -fPIC -Wall
#CFLAGS=-Ofast -march=native -fPIC -Wall

#INCLUDE_PATHS=-I/usr/local/mesa/include
#LIB_PATHS=-L/usr/local/mesa/lib
INCLUDE_PATHS=-I${HOME}/usr/include
LIB_PATHS=-L${HOME}/usr/lib

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

#LIBS=-lOSMesa
#LIBS=-l:/home/daeyun/usr/lib/libOSMesa.a -lstdc++ -lOSMesa
#LIBS=-L/home/daeyun/usr/lib/ -l:/home/daeyun/usr/lib/libOSMesa.so -lstdc++
LIBS=-l:/home/daeyun/usr/lib/libOSMesa.a -lstdc++ -lgcrypt -lncurses `llvm-config --cxxflags --ldflags --libs` `llvm-config --ldflags`

default: $(TARGET)

test: render_depth.c
	$(CC) $(CFLAGS) $< -o $@ $(INCLUDE_PATHS) $(LIB_PATHS) $(LIBS) 

%.darwin.so: %.c
	$(CC) $(CFLAGS) -shared $< -o $@ $(INCLUDE_PATHS) $(LIB_PATHS) $(LIBS) 

%.linux.so: %.c
	$(CC) $(CFLAGS) -shared $< -o $@ $(INCLUDE_PATHS) $(LIB_PATHS) $(LIBS) 

clean:
	rm -f $(TARGET) *.pyc

recompile:
	$(MAKE) clean
	$(MAKE)
