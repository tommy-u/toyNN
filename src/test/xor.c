#include "tnn.h"
int main(){

  srand(42);
  uint a[3]= { 2, 3, 1};
  Net *n = tnn_init_net(3, a); 
  tnn_print_net(n);

  double in[2];
  double label[1];

  /* XOR fn before training. */
  uint i;
  for(i=0;i<2;i++){
    uint j;
    for(j=0;j<2;j++){
      in[0]=i;in[1]=j; label[0]=i^j;
      tnn_print_output_activation(n,in);
    }
  }
  printf("\nTRAIN\n");
  
  double lrate = 20;
  for(i=0;i<35;i++){
    uint j;
    for(j=0;j<2;j++){ //only running 1 time
      uint k;
      for(k=0;k<2;k++){
	in[0]=j; in[1]=k; label[0]=j^k;
	//	printf("training in %u %u : %u\n",j,k,j^k);
	tnn_backprop(n, in, label, lrate);
      }
    }
  }
    
  printf("\nTEST\n");
  for(i=0;i<2;i++){
    uint j;
    for(j=0;j<2;j++){
      in[0]=i;in[1]=j; label[0]=i^j;
      printf("%u xor %u = %u",i,j,i^j);
      tnn_print_output_activation(n,in);
    }
  }

  //  tnn_print_net(n);
  tnn_destroy_net(n);
  printf("exiting gracefully \n");
  return 0;
}
