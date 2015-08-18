#include "tnn.h"
int main(){
  //    srand(time(NULL));
  srand(5);
  uint a[3]= { 1, 1, 1};
  //Net *n = tnn_init_net(2, a); 
  
  //uint a[2]= {1, 1};
  Net *n = tnn_init_net(3, a); 
  tnn_print_net(n);

  //Let's learn the copy function
  //  double in[1];
  //double label[1];


  /* XOR fn before training. */
  /*
  for(i=0;i<2;i++){
    uint j;
    for(j=0;j<2;j++){
      in[0]=i;in[1]=j; label[0]=i^j;
      printf("%u xor %u = %u",i,j,i^j);
      tnn_print_output_activation(n,in);
    }
  }
  printf("\nTRAIN\n");
  
  for(i=0;i<100;i++){
    uint j;
    for(j=0;j<2;j++){ //only running 1 time
      uint k;
      for(k=0;k<2;k++){
      in[0]=1;in[1]=1; label[0]=1;
      //printf("%u xor %u = %u\n",j,k,j^k);
      tnn_backprop(n, in, label, .1);
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
  */
  tnn_destroy_net(n);
  printf("exiting gracefully \n");
  return 0;
}
