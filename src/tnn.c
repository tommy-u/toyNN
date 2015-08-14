/* 
Very basic NN implementation. Feed forward & backprop SGD.
Draws on http://leenissen.dk/fann/wp/ and 
https://github.com/mnielsen/neural-networks-and-deep-learning.
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <time.h>
#include <math.h>

typedef unsigned int uint;

typedef struct {
  uint num_layers;
  /* Num neurons in each layer */
  uint *layers;

  /* Jagged 2d array containing bias values. Not needed for first layer. */
  double **biases;

  /* List of num_layers -1 many 2d rectangular arrays of connections
     between layer l and l-1. An inner loop over one of these 2d arrays
     iterates over the neurons in layer l-1. */
  double ***connections;
} Net;

Net * tnn_init_net(uint num_layers, uint *layers){
  /* Creates Net struct. Assumes fully connected. */
  uint i;
  double **biases;
  double ***connections;

  /* Allocate biases array. */
  biases = malloc( num_layers * sizeof(double *));
  if(biases == NULL){
    printf("allocation of biases in init_net failed \n");
    exit(1);
  }  
  
  /* Skip input layer. */
  for(i=0; i<num_layers-1; i++){
    biases[i] = malloc( layers[i+1] * sizeof(double));
    if(biases[i] == NULL){
      printf("allocation of biases[%u] in init_net failed \n",i);
      exit(1);
    }
  }
  
  /* Allocate connections. */
  connections = malloc( (num_layers - 1 ) * sizeof(double * ) );
  if(connections == NULL){
    printf("allocation of connections in init_net failed \n");
    exit(1);
  }

  /* Each connection matrix. */
  for(i=0; i<num_layers - 1; i++){
    connections[i] = malloc( layers[i+1] * sizeof(double * ) );
    if(biases[i] == NULL){
      printf("allocation of connections[%u] in init_net failed \n",i);
      exit(1);
    }
    /* Each neuron in the i+1th layer. */
    uint j;
    for(j=0; j<layers[i+1]; j++){
      connections[i][j] = malloc ( layers[i] * sizeof(double) );
      if(connections[i][j] == 0){
	printf("allocation of connections[%u][%u] in init_net failed \n",i,j);
	exit(1);
      }
    }
  }
  
  /* Create, initialize Net. */
  Net *n = malloc( sizeof(Net) );
  n->num_layers = num_layers;
  n->layers = layers;
  n->biases = biases;
  n->connections = connections;
  return n;
}



void tnn_destroy_net(Net *n){
  /* Properly free all heap  memory. */
  uint i;

  /* Free bias values. */
  for(i=0; i<n->num_layers; i++)
    free(n->biases[i]);
  free(n->biases);

  /* Free Connections. */
  for(i=0; i<n->num_layers-1; i++){
    uint j;
    for(j=0; j<n->layers[i+1]; j++){
      free(n->connections[i][j]);
    }
    free(n->connections[i]);
  }
  free(n->connections);
  free(n);
}

double tnn_sigmoid(double z){
  return 1 / ( 1 + exp( -1 * z ) );
 }

double tnn_sigmoid_prime(double z){
  return tnn_sigmoid(z) * (1 - tnn_sigmoid(z));
}

int main(){
  uint a[3]= {2, 3, 1};
  Net *n = tnn_init_net(3, a); 

  /* test */ 
  printf("%u is first connection \n", n->num_layers);
  uint i; 
  for(i=0; i<n->num_layers; i++)
    printf("%u \n",n->layers[i]);
  printf("%f sig \n",tnn_sigmoid(1));

  tnn_destroy_net(n);
  return 0;
}
