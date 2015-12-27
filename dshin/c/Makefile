CC=gcc
#CFLAGS=-O2 -fPIC -Wall
CFLAGS=-Ofast -march=native -fPIC -Wall
TARGET=render_depth.so
LIBS=-lOSMesa

default: $(TARGET)

%.so: %.c
	$(CC) $(CFLAGS) -shared $< -o $@ $(LIBS)

clean:
	rm -f $(TARGET) *.pyc

recompile:
	$(MAKE) clean
	$(MAKE)
