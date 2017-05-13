#include <stdio.h>
#include "lib.c"

imap input;
wmap w_conv0;
bmap b_conv0;
imap conv0;
imap relu0;
imap pool0;
wmap w_conv1;
bmap b_conv1;
imap conv1;
imap relu1;
imap pool1;
ivec xflat;
wvec w_full2;
bvec b_full2;
ivec full2;
ivec relu2;
wvec w_full3;
bvec b_full3;
ivec full3;
ivec output;

void LeNet_init(void)
{
  input   = new_imap(1, 28, 28);
  w_conv0 = new_wmap(16, 1, 3, 3);
  b_conv0 = new_bmap(16);
  conv0   = new_imap(16, 28, 28);
  relu0   = new_imap(16, 28, 28);
  pool0   = new_imap(16, 14, 14);
  w_conv1 = new_wmap(32, 16, 3, 3);
  b_conv1 = new_bmap(32);
  conv1   = new_imap(32, 14, 14);
  relu1   = new_imap(32, 14, 14);
  pool1   = new_imap(32, 7, 7);
  xflat   = new_ivec(1568);
  w_full2 = new_wvec(256, 1568);
  b_full2 = new_bvec(256);
  full2   = new_ivec(256);
  relu2   = new_ivec(256);
  w_full3 = new_wvec(10, 256);
  b_full3 = new_bvec(10);
  full3   = new_ivec(10);
  output  = new_ivec(10);

  load_wmap(&w_conv0, "lenet/conv0/W.dat");
  load_bmap(&b_conv0, "lenet/conv0/b.dat");
  load_wmap(&w_conv1, "lenet/conv1/W.dat");
  load_bmap(&b_conv1, "lenet/conv0/b.dat");
  load_wvec(&w_full2, "lenet/full2/W.dat");
  load_bvec(&b_full2, "lenet/conv0/b.dat");
  load_wvec(&w_full3, "lenet/full3/W.dat");
  load_bvec(&b_full3, "lenet/conv0/b.dat");
}

int LeNet_eval(char *path)
{
  load_imap(&input, path);

  convpad(&conv0, &input, &w_conv0, &b_conv0);
  relumap(&relu0, &conv0);
  pool2x2(&pool0, &relu0);

  convpad(&conv1, &pool0, &w_conv1, &b_conv1);
  relumap(&relu1, &conv1);
  pool2x2(&pool1, &relu1);

  flatten(&xflat, &pool1);

  fullvec(&full2, &xflat, &w_full2, &b_full2);
  reluvec(&relu2, &full2);

  fullvec(&full3, &relu2, &w_full3, &b_full3);
  softmax(&output, &full3);

  return argmax(&output);
}

void LeNet_del(void)
{
  del_imap(&input);
  del_wmap(&w_conv0);
  del_imap(&conv0);
  del_imap(&relu0);
  del_imap(&pool0);
  del_wmap(&w_conv1);
  del_imap(&conv1);
  del_imap(&relu1);
  del_imap(&pool1);
  del_ivec(&xflat);
  del_wvec(&w_full2);
  del_ivec(&full2);
  del_ivec(&relu2);
  del_wvec(&w_full3);
  del_ivec(&full3);
  del_ivec(&output);
}

int main(void)
{
  int num;
  int answer = 0;
  int total = 0;
  char path[256];
  char filelist[] = "mnist_test.txt";
  FILE *fp;

  if ((fp = fopen(filelist, "r")) == NULL) {
    fprintf(stderr, "main failed: %s does't exist\n", filelist);
    exit(1);
  }

  LeNet_init();

  while (fscanf(fp, "%s %d", path, &num) != EOF) {
    int label = LeNet_eval(path);
    if (label == num) answer++;
    total++;
  }

  LeNet_del();

  fclose(fp);

  printf("Accuracy: %f\n", (float)answer/total);

  return 0;
}

