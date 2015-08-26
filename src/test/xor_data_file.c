#include "tnn.h"
int main(){
  srand(42);
  uint a[3]= { 2, 3, 1}, i;
  Net *n = tnn_init_net(3, a); 
  double in[2], label[1];
  double lrate = 20;
  FILE *fp;
  char *mode = "r", *in_file = "xor.data";
  uint dat_num, dat_in, dat_out;
  fp = fopen(in_file, mode);
  if (fp == NULL){
    fprintf(stderr, "Can't open file %s,\n",in_file);
    exit(1);
  }
  
  fscanf(fp,"%u %u %u",&dat_num,&dat_in,&dat_out);
  printf("%u is dat_num\n",dat_num);
  printf("%u is dat_num\n",dat_in);
  printf("%u is dat_num\n",dat_out);

  malloc();

  for(i=0;i<dat_num*2;i++){
    
  }
  
  fclose(fp);
  


  getchar();
  printf("\nTRAIN\n");
  
  for(i=0;i<35;i++){
	tnn_backprop(n, in, label, lrate);
  }
    
  tnn_destroy_net(n);
  printf("exiting gracefully \n");
  return 0;
}
