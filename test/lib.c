#include <assert.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#define range(a, b) (int (a)=0; (a)<(b); ++(a))

typedef struct {
  int shape;
  float *map;
} bvec;

bvec new_bvec(int size0)
{
  bvec x;
  x.shape = size0;

  x.map = (float *)calloc(size0, sizeof(float));

  return x;
}

void load_bvec(bvec *x, char *path)
{
  FILE *fp;

  if ((fp = fopen(path, "r")) == NULL) {
    fprintf(stderr, "load_bvec failed: %s does't exist\n", path);
    exit(1);
  }

  for range(i, x->shape)
    fscanf(fp, "%f", &x->map[i]);

  fclose(fp);
}

void del_bvec(bvec *x)
{
  free(x->map);
}

typedef struct {
  int shape;
  float *map;
} bmap;

bmap new_bmap(int size0)
{
  bmap x;
  x.shape = size0;

  x.map = (float *)calloc(size0, sizeof(float));

  return x;
}

void load_bmap(bmap *x, char *path)
{
  FILE *fp;

  if ((fp = fopen(path, "r")) == NULL) {
    fprintf(stderr, "load_bmap failed: %s does't exist\n", path);
    exit(1);
  }

  for range(i, x->shape)
    fscanf(fp, "%f", &x->map[i]);

  fclose(fp);
}

void del_bmap(bmap *x)
{
  free(x->map);
}

typedef struct {
  int shape;
  float *map;
} ivec;

ivec new_ivec(int size0)
{
  ivec x;
  x.shape = size0;

  x.map = (float *)calloc(size0, sizeof(float));

  return x;
}

void load_ivec(ivec *x, char *path)
{
  FILE *fp;

  if ((fp = fopen(path, "r")) == NULL) {
    fprintf(stderr, "load_ivec failed: %s does't exist\n", path);
    exit(1);
  }

  for range(i, x->shape)
    fscanf(fp, "%f", &x->map[i]);

  fclose(fp);
}

void del_ivec(ivec *x)
{
  free(x->map);
}

typedef struct {
  int shape[2];
  float **map;
} wvec;

wvec new_wvec(int size0, int size1)
{
  wvec w;
  w.shape[0] = size0;
  w.shape[1] = size1;

  w.map = (float **)calloc(size0, sizeof(float *));

  for range(i, size0)
    w.map[i] = (float *)calloc(size1, sizeof(float));

  return w;
}

void load_wvec(wvec *w, char *path)
{
  FILE *fp;

  if ((fp = fopen(path, "r")) == NULL) {
    fprintf(stderr, "load_wvec failed: %s does't exist\n", path);
    exit(1);
  }

  for range(i, w->shape[0])
    for range(j, w->shape[1])
      fscanf(fp, "%f", &w->map[i][j]);

  fclose(fp);
}

void del_wvec(wvec *w)
{
  for range(i, w->shape[0])
      free(w->map[i]);

  free(w->map);
}

typedef struct {
  int shape[3];
  float ***map;
} imap;

imap new_imap(int size0, int size1, int size2)
{
  imap x;
  x.shape[0] = size0;
  x.shape[1] = size1;
  x.shape[2] = size2;

  x.map = (float ***)calloc(size0, sizeof(float **));

  for range(i, size0)
    x.map[i] = (float **)calloc(size1, sizeof(float *));

  for range(i, size0)
    for range(j, size1)
      x.map[i][j] = (float *)calloc(size2, sizeof(float));

  return x;
}

void load_imap(imap *x, char *path)
{
  FILE *fp;

  if ((fp = fopen(path, "r")) == NULL) {
    fprintf(stderr, "load_imap failed: %s does't exist\n", path);
    exit(1);
  }

  for range(i, x->shape[0])
    for range(j, x->shape[1])
      for range(k, x->shape[2])
        fscanf(fp, "%f", &x->map[i][j][k]);

  fclose(fp);
}

void del_imap(imap *x)
{
  for range(i, x->shape[0])
    for range(j, x->shape[1])
      free(x->map[i][j]);

  for range(i, x->shape[0])
      free(x->map[i]);

  free(x->map);
}

typedef struct {
  int shape[4];
  float ****map;
} wmap;

wmap new_wmap(int size0, int size1, int size2, int size3)
{
  wmap w;
  w.shape[0] = size0;
  w.shape[1] = size1;
  w.shape[2] = size2;
  w.shape[3] = size3;

  w.map = (float ****)calloc(size0, sizeof(float ***));

  for range(i, size0)
    w.map[i] = (float ***)calloc(size1, sizeof(float **));

  for range(i, size0)
    for range(j, size1)
      w.map[i][j] = (float **)calloc(size2, sizeof(float *));

  for range(i, size0)
    for range(j, size1)
      for range(k, size2)
        w.map[i][j][k] = (float *)calloc(size3, sizeof(float));

  return w;
}

void load_wmap(wmap *w, char *path)
{
  FILE *fp;

  if ((fp = fopen(path, "r")) == NULL) {
    fprintf(stderr, "load_wmap failed: %s does't exist\n", path);
    exit(1);
  }

  for range(i, w->shape[0])
    for range(j, w->shape[1])
      for range(k, w->shape[2])
        for range(l, w->shape[3])
          fscanf(fp, "%f", &w->map[i][j][k][l]);

  fclose(fp);
}

void del_wmap(wmap *w)
{
  for range(i, w->shape[0])
    for range(j, w->shape[1])
      for range(k, w->shape[2])
      free(w->map[i][j][k]);

  for range(i, w->shape[0])
    for range(j, w->shape[1])
      free(w->map[i][j]);

  for range(i, w->shape[0])
      free(w->map[i]);

  free(w->map);
}

void zero_pad(imap *x_pad, imap *x, wmap *w)
{
  assert(x_pad->shape[0] == x->shape[0]);
  assert(x_pad->shape[1] == x->shape[1] + w->shape[2] - 1);
  assert(x_pad->shape[2] == x->shape[2] + w->shape[3] - 1);

  int i0 = (w->shape[2]-1)/2;
  int j0 = (w->shape[3]-1)/2;

  for range(n, x->shape[0])
  for range(i, x->shape[1])
  for range(j, x->shape[2])
    x_pad->map[n][i+i0][j+j0] = x->map[n][i][j];
}

void convpad(imap *y, imap *x, wmap *w, bmap *b)
{
  // chainer encoding
  assert(b->shape == w->shape[0]);
  assert(x->shape[0] == w->shape[1]);
  assert(y->shape[0] == w->shape[0]);
  assert(y->shape[1] == x->shape[1]);
  assert(y->shape[2] == x->shape[2]);

  imap x_pad = new_imap(x->shape[0],
                        x->shape[1] + w->shape[2] - 1,
                        x->shape[2] + w->shape[3] - 1);

  zero_pad(&x_pad, x, w);

  for range(n, w->shape[0])
  for range(i, x->shape[1])
  for range(j, x->shape[2]) {
    float tmp = 0.0;
    for range(m, w->shape[1])
    for range(di, w->shape[2])
    for range(dj, w->shape[3])
      tmp += w->map[n][m][di][dj] * x_pad.map[m][i+di][j+dj];
    y->map[n][i][j] = tmp + b->map[n];
  }

  del_imap(&x_pad);
}

void relumap(imap *y, imap *x)
{
  assert(y->shape[0] == x->shape[0]);
  assert(y->shape[1] == x->shape[1]);
  assert(y->shape[2] == x->shape[2]);

  for range(i, x->shape[0])
  for range(j, x->shape[1])
  for range(k, x->shape[2])
    if (x->map[i][j][k] > 0.0)
      y->map[i][j][k] = x->map[i][j][k];
    else
      y->map[i][j][k] = 0.0;
}

void pool2x2(imap *y, imap *x)
{
  assert(y->shape[0] == x->shape[0]);
  assert(y->shape[1] == x->shape[1]/2);
  assert(y->shape[2] == x->shape[2]/2);

  for range(n, x->shape[0])
  for (int i = 0; i < x->shape[1]; i+=2)
  for (int j = 0; j < x->shape[2]; j+=2) {
    float tmp = -DBL_MAX;
    for range(di, 2)
    for range(dj, 2)
      if (x->map[n][i+di][j+dj] > tmp)
        tmp = x->map[n][i+di][j+dj];
    y->map[n][i/2][j/2] = tmp;
  }
}

void flatten(ivec *y, imap *x)
{
  assert(y->shape == x->shape[0]*x->shape[1]*x->shape[2]);

  for range(i, x->shape[0])
    for range(j, x->shape[1])
      for range(k, x->shape[2])
        y->map[x->shape[1]*x->shape[2]*i+x->shape[2]*j+k] = x->map[i][j][k];
}

void fullvec(ivec *y, ivec *x, wvec *w, bvec *b)
{
  assert(b->shape == w->shape[0]);
  assert(y->shape == w->shape[0]);
  assert(x->shape == w->shape[1]);

  for range(i, w->shape[0]) {
    float tmp = 0.0;
    for range(j, w->shape[1])
      tmp += w->map[i][j] * x->map[j];
    y->map[i] = tmp + b->map[i];
  }
}

void reluvec(ivec *y, ivec *x)
{
  assert(y->shape == x->shape);

  for range(i, x->shape)
    if (x->map[i] > 0.0)
      y->map[i] = x->map[i];
    else
      y->map[i] = 0.0;
}

void softmax(ivec *y, ivec *x)
{
  assert(y->shape == x->shape);

  float sum_exp_x = 0.0;
  for range(i, x->shape)
    sum_exp_x += exp(x->map[i]);

  for range(i, x->shape)
    y->map[i] = exp(x->map[i]) / sum_exp_x;
}


int argmax(ivec *x)
{
  int num = -1;
  float max_like = -FLT_MAX;

  for range(i, x->shape) {
    if (x->map[i] > max_like) {
      max_like = x->map[i];
      num = i;
    }
  }

  return num;
}
