CC=gcc
CFLAGS=-O2 -fPIC
TARGET=render_depth.so
LIBS=-lOSMesa

default: $(TARGET)

$(TARGET): render_depth.c
	$(CC) $(CFLAGS) -shared -o $(TARGET) render_depth.c $(LIBS)

clean:
	rm -f $(TARGET) *.pyc

recompile:
	$(MAKE) clean
	$(MAKE)
