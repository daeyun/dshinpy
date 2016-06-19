/*
Off-screen depth rasterizer.

gcc -O2 -fPIC -shared -o render_depth.so render_depth.c -lOSMesa
*/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <GL/osmesa.h>

#define CHECK(cond, ...)            \
  if (!(cond)) {                    \
    fprintf(stderr, ##__VA_ARGS__); \
    return 1;                       \
  }

int render_depth(const double *vertices, const size_t num_points, const int w,
                 const int h, uint32_t *out) {
  CHECK(w > 0 && h > 0 && w < (1 << 15) && h < (1 << 15),
        "Invalid image size: %d, %d", w, h);

  CHECK(num_points % 3 == 0, "num_points must be a multiple of 3");

  OSMesaContext context = OSMesaCreateContextExt(OSMESA_RGBA, 32, 0, 0, NULL);

  void *buffer = malloc(w * h * 3 * sizeof(uint8_t));

  CHECK(OSMesaMakeCurrent(context, buffer, GL_UNSIGNED_BYTE, w, h),
        "Error in OSMesaMakeCurrent.");

  OSMesaPixelStore(OSMESA_Y_UP, 0);

  glEnable(GL_DEPTH_TEST);
  glDisable(GL_LIGHTING);
  glDisable(GL_CULL_FACE);

  glDepthFunc(GL_LESS);

  glViewport(0, 0, (GLsizei)w, (GLsizei)h);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glBegin(GL_TRIANGLES);
  size_t i;
  for (i = 0; i < num_points / 3; ++i) {
    glVertex3dv(vertices + i * 3);
  }
  glEnd();

  /*glFinish();*/

  GLint out_width, out_height, bytes_per_value;
  uint32_t *depth_buffer;
  CHECK(OSMesaGetDepthBuffer(context, &out_width, &out_height, &bytes_per_value,
                             (void **)&depth_buffer),
        "Error in OSMesaGetDepthBuffer.");

  CHECK(bytes_per_value == 4, "Bytes per values != 4");
  CHECK(out_width == w && out_height == h, "Unexpected output size: %d, %d",
        out_width, out_height);

  memcpy(out, depth_buffer, out_height * out_width * sizeof(uint32_t));
  int ii = 0;
  for(ii=0;ii<out_height * out_width;ii++) {
      printf("%d\n", depth_buffer[ii]);
  }

  OSMesaDestroyContext(context);
  free(buffer);

  return 0;
}
