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
double tnn_rand_norm(const double mean, const double std)
{
  double u, v, s;
  do{
      u = (rand() / ((double) RAND_MAX)) * 2.0 - 1.0;
      v = (rand() / ((double) RAND_MAX)) * 2.0 - 1.0;
      s = u * u + v * v;
    }
  while( (s >= 1.0) || (s == 0.0) );

  s = sqrt(-2.0 * log(s) / s);
  return mean + std * u * s;
}

typedef unsigned int uint;

typedef struct {
  uint num_layers;
  /* Num neurons in each layer */
  uint *layers;

  /* Space allocated for the result of running the net */
  double *output;

  /* Jagged 2d array containing bias values. Not needed for first layer. */
  double **biases;

  /* For use in backprop. Store during feed forward. Use to compute 
    sigma_prime */
  double **pre_activations;

  /* List of num_layers -1 many 2d rectangular arrays of connections
     between layer l and l-1. An inner loop over one of these 2d arrays
     iterates over the neurons in layer l-1. */
  double ***connections;

} Net;

double * cost_derivative(Net *n, double *labels, double *output){
  /* Partial derivatives for output activation using squared 
    difference cost function */
  double *derivs = calloc(n->layers[n->num_layers-1], sizeof(double));
  uint i;
  for (i = 0; i < n->layers[n->num_layers-1]; i++){
    derivs[i] = output[i] - labels[i];
  }
  return derivs;
}

void tnn_init_connections(Net *n){
  
  uint i;
  for(i=0;i<n->num_layers-1;i++){
    uint j;
    for(j=0;j<n->layers[i+1];j++){
      uint k;
      for(k=0;k<n->layers[i];k++){
        n->connections[i][j][k] = tnn_rand_norm(0,1);
      }
    }
  }
}

void tnn_init_biases(Net *n){
  uint i;
  for(i=0;i<n->num_layers-1;i++){
    uint j;
    for(j=0;j<n->layers[i+1];j++){
      n->biases[i][j] = tnn_rand_norm(0,1);
    }
  }
}

Net * tnn_init_net(uint num_layers, uint *layers){
  /* Creates Net struct. Assumes fully connected. */
  uint i;
  double *output;
  double **biases, **pre_activations;
  double ***connections;

  /* Allocate output array. */
  output = malloc( layers[num_layers-1] * sizeof(double) );

  /* Allocate biases array. */
  biases = malloc( num_layers * sizeof(double *) );
  if(biases == NULL){
    printf("allocation of biases in init_net failed \n");
    exit(1);
  }  
  
  /* Skip input layer. */
  for(i=0; i<num_layers-1; i++){
    biases[i] = malloc( layers[i+1] * sizeof(double) );
    if(biases[i] == NULL){
      printf("allocation of biases[%u] in init_net failed \n",i);
      exit(1);
    }
  }

  /* Allocate pre_activations array */

  pre_activations = malloc( num_layers * sizeof(double*));
  if (pre_activations == NULL){
    printf("allocation of pre_activations in init_net failed \n");
    exit(1);
  } 

  /* Skip input layer. */
  for (i = 0; i < num_layers-1; i++){
    pre_activations[i] = malloc (layers[i+1] * sizeof(double));
    if (pre_activations[i] == NULL){
      printf("allocation of pre_activations in init_net failed \n");
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
  n->output = output;
  n->biases = biases;
  n->pre_activations = pre_activations;
  n->connections = connections;
  
  /* Warm up prg */
  for(i=0;i<10;i++)
    rand();
  tnn_init_connections(n);
  tnn_init_biases(n);

  return n;
}

void tnn_destroy_net(Net *n){
  /* Properly free all heap  memory. */
  uint i;

  /* Free output array. */
  free(n->output);

  /* Free bias values. -1 because none for input. */
  for(i=0; i<n->num_layers-1; i++)
    free(n->biases[i]);
  free(n->biases);

  /* Free pre_activations. */
  for(i=0; i<n->num_layers-1; i++)
    free(n->pre_activations[i]);
  free(n->pre_activations);

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
  /* Sigmoid Fn */
  return 1 / ( 1 + exp( -1 * z ) );
 }

double tnn_sigmoid_prime(double z){
  /* Derivative of Sigmoid */
  return tnn_sigmoid(z) * (1 - tnn_sigmoid(z));
}

double * tnn_feedforward(Net *n, double *input, uint train){
  /* i iterates over layers, j iterates over the output, 
     k iterates over the input.  
     Train is a flag specifying if the pre_activations should
     be recorded. */
  double *previous, *current;
  previous = input;
  
  uint i;
  /* Loop over connection matrices between layers. */
  for (i=0; i<n->num_layers-1; i++) {
    /* This will contain the activations. */
    current = calloc(n->layers[i+1], sizeof(double) );
    /* Temporary storage for layer activation. */
    uint j;

    /* Loop over output neurons (of this layer). */
    for(j=0; j<n->layers[i+1]; j++){
      uint k;
      /* Loop over incoming connection signals. This is the dot 
	 product of the input vector and the weight matrix. */
      for(k=0; k<n->layers[i]; k++){
	current[j] += previous[k] * n->connections[i][j][k];
      }
      /* Add the bias term, this is the preactivation.  */
      current[j] += n->biases[i][j];
      
      if(train)
        n->pre_activations[i][j] = current[j];

      /* This is the activation of the n->layers[i+1] neurons in
      	 the i+1st layer. */
      current[j] = tnn_sigmoid(current[j]);
    }
    
    if(i!=0)
      free(previous);

    previous = current;
  }

  /* Copy current into net's output array. */
  for (i = 0; i < n->layers[n->num_layers-1]; i++){
    n->output[i] = current[i];
  }
  free(current);
  //current needs to be freed elsewhere. 
  return current; 
}

void tnn_print_net(Net *n){
  
  printf("----------------------------------------------\n");
  
  printf("layer counts in->out: ");
  uint i;
  for(i=0;i<n->num_layers;i++){
    printf("%u",n->layers[i]);
    if (i!=n->num_layers-1)
      printf(" - ");
  }
  
  printf("\n");
  
  printf("\nBIAS, values starting with first hidden layer -> output\n");
  for(i=0;i<n->num_layers-1;i++){
    if (i!=0)
      printf("\n");

    printf("layer %u:\n",i+1);
    uint j;
    for(j=0;j<n->layers[i+1];j++){
      printf("n: %u \t v: %.3f \n",j,n->biases[i][j]);
    }
  }
  
  printf("\nCONNECTIONS, count on incoming connection. \n");
  for(i=0;i<n->num_layers-1;i++){
    if (i!=0)
      printf("\n");

    printf("layer %u:\n",i+1);

    uint j;
    for(j=0;j<n->layers[i+1];j++){
      uint k=0;
      for(k=0;k<n->layers[i];k++){
	printf("o: %u i: %u\t%.3f\n",j,k,n->connections[i][j][k]);
      }
    }
  }

  printf("----------------------------------------------\n");
}


void tnn_print_output_activation(Net *n, double *in){
  double *out = tnn_feedforward(n, in, 0);
  printf("out vec: \n");
  uint i;
  for(i=0;i<n->layers[n->num_layers-1]; i++)
    printf("%.3f \n",out[i]);
  free(out);
}

void tnn_print_pre_activations(Net *n){
  uint i;

  printf("pre_activations:\n");
  for (i = 0; i < n->num_layers-1; i++){
    if(i!=0)
      printf("\n");
    printf("layer %u\n", i);

    uint j;
    for (j = 0; j < n->layers[i+1]; j++){
      printf("l:%u n:%u v: %.3f\n",i,j,n->pre_activations[i][j]);
    }
    
  }
}

void tnn_backprop(){

}


int main(){
  srand(time(NULL));
  uint a[3]= {2, 3, 1};
  double in[2] = {1, 1};
  Net *n = tnn_init_net(3, a); 

  tnn_feedforward(n, in, 1);

  tnn_print_pre_activations(n);

  tnn_print_output_activation(n,in);

  tnn_destroy_net(n);
  printf("exiting gracefully \n");
  return 0;
}
