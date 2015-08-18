#include "tnn.h"

int main(){
  srand(time(NULL));
  uint a[3]= { 1, 5, 2};
  Net *n = tnn_init_net(3, a); 
  //  tnn_print_net(n);

  double in[2];
  double label[2];

  in[0]=0; in[1]=0;
  tnn_print_output_activation(n,in);  
  in[0]=1; in[1]=1;
  tnn_print_output_activation(n,in);  
  
  printf("\nTRAIN\n\n");
  uint i;

  double lrate = .1;
  for(i=0;i<10000;i++){
    in[0]=0; in[1]=0;
    label[0]=1; label[1]=1;
    tnn_backprop(n, in, label, lrate); 
    
    in[0]=1; in[1]=1;
    label[0]=0; label[1]=0;
    tnn_backprop(n, in, label, lrate); 
  }

  in[0]=0; in[1]=0;
  tnn_print_output_activation(n,in);

  in[0]=1; in[1]=1;
  tnn_print_output_activation(n,in);

  //tnn_print_net(n);
  tnn_destroy_net(n);
  return 0;
}
