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
  double in[1];
  double label[1];

  //  tnn_print_net(n);

  in[0]=0;// label[0]=1;
  tnn_print_output_activation(n,in);  
  in[0]=1;// label[0]=1;
  tnn_print_output_activation(n,in);  
  
  printf("\ntrain\n\n");

  uint i;

  double lrate = .1;
  for(i=0;i<100000;i++){

    in[0]=0; label[0]=1;
    tnn_backprop(n, in, label, lrate); 
    
    in[0]=1; label[0]=0;
    tnn_backprop(n, in, label, lrate); 
  
  }

  in[0]=0;
  tnn_print_output_activation(n,in);

  in[0]=1;
  tnn_print_output_activation(n,in);

  tnn_print_net(n);


  tnn_destroy_net(n);
  printf("exiting gracefully \n");
  return 0;
}
