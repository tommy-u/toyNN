/* Very basic NN implementation. Feed forward & backprop SGD.
   Draws on http://leenissen.dk/fann/wp/ and 
   https://github.com/mnielsen/neural-networks-and-deep-learning.
   tommyu@bu.edu */

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <time.h>
#include <math.h>

typedef unsigned int uint;

typedef struct{
  uint num_layers;

  /* Num neurons in each layer */
  uint *layers;

  /* Space allocated for the result of running the net */
  double *output;

  /* Jagged 2d array containing bias values. Not needed for first layer. */
  double **biases;

  /* For use in backprop. Store during feed forward. Use to compute 
     sigma prime. */
  double **pre_activations;

  /* List of num_layers -1 many 2d rectangular arrays of connections 
     between layer l and l-1. An inner loop over one of these 2d arrays
     iterates over the neurons in layer l-1. */
  double ***connections;

} Net;

double tnn_rand_norm(const double mean, const double std){
  /* Provides the base case for backprop. */
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

void tnn_cost_derivative(Net *n, double *labels, double **err){
  /* Partial derivatives for output activation using squared difference 
     cost function: C = 1/2 (a - y)^2. */
  uint i;
  for (i = 0; i < n->layers[n->num_layers-1]; i++){
    err[n->num_layers-2][i] = n->output[i] - labels[i]; 
    //err[n->num_layers-2][i] = labels[i] - n->output[i];
  }
}

void tnn_init_connections(Net *n){
  /* A perhaps sub-optimal initialization scheme (high
     variance). */
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
  /* A perhaps sub-optimal initialization scheme (high
     variance). */
  uint i;
  for(i=0;i<n->num_layers-1;i++){
    uint j;
    for(j=0;j<n->layers[i+1];j++){
      n->biases[i][j] = tnn_rand_norm(0,1);
    }
  }
}

Net * tnn_init_net(uint num_layers, uint *layers){
  /* Creates Net. Assumes fully connected. */
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
  /* Sigmoid Fn. */
  return 1 / ( 1 + exp( -1 * z ) );
}

double tnn_sigmoid_prime(double z){
  /* Derivative of Sigmoid at pt. */
  return tnn_sigmoid(z) * (1 - tnn_sigmoid(z));
}

void tnn_feedforward(Net *n, double *input, uint train){
  /* Run the net in forward mode, generating output. */

  /* Train is a flag specifying if the pre_activations should
     be recorded (needed for backprop, otherwise unneeded. */
  
  double *previous, *current=NULL;
  previous = input;

  uint i;
  /* For each connection matrix. Ex: matrix 0 represents
     the connections between the input (0th layer) and first
     hidden layer (1st layer). */
  for (i=0; i<n->num_layers-1; i++){
    /* This will contain the input, intermediate, and final  
       activations sequentially. Temp storage. */
    current = calloc(n->layers[i+1], sizeof(double));

    uint j;
    /* Loop over output neurons (of ith layer). Ex: the neurons
       in the 1st layer when looking at matrix 0. */
    for(j=0; j<n->layers[i+1]; j++){
      uint k;
      /* Loop over incoming connection signals. Ex: This is the dot 
	 product of the input vector and the weight matrix for 
	 matrix 0, the activation of the 1st layer and weight 
	 matrix for matrix 1. */
      for(k=0; k<n->layers[i]; k++){
	current[j] += previous[k] * n->connections[i][j][k];
      }
      
      /* Add the bias term, we've calculated the preactivation. */
      current[j] += n->biases[i][j];

      /* If we're doing backprop, we will need to use these. */
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
  tnn_feedforward(n, in, 0);
  printf("out vec: \n");
  uint i;
  for(i=0;i<n->layers[n->num_layers-1]; i++)
    printf("n:%u v:%.3f \n",i,n->output[i]);
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

double ** tnn_allocate_error_arrays(Net *n){
  /* Allocate pointers to each layer after input. */
  double **err = malloc( (n->num_layers-1) * sizeof(double*) );
  if(err==NULL){
    printf("Failed to allocate error array in tnn_allocate_error_arrays\n");
    exit(1);
  }

  /* For each layer after the input, allocate doubles for each neuron. */
  uint i;
  for(i=0; i < n->num_layers-1; i++){
    err[i] = calloc( n->layers[i+1], sizeof(double) );
    if(err[i] == NULL){
      printf("Failed to allocate error[%u] array in tnn_allocate_error_arrays\n",i);
      exit(1);
    }
  }
  return err;
}

void tnn_generate_error(Net *n, double *labels, double **err){
  /* Backprop requires finding the derivative of the Cost fn WRT all
     the connections and biases in the net. This fn caluculates 
     derivitives WRT the pre_activation of each neuron. It is trivial
     to calculate the derivitive WRT the connections or biases once 
     the partials WRT the pre_activations are known. */
  int i;  
  
  for(i=n->num_layers-2; i>=0; i--){
    /* This calculates derivative of cost fn with respect to activation, then multiplies
       by derivitive of the activation with respect to the pre_activation per the chain 
       rule. There are 2 cases, the base case where we calculate the error of the output 
       layer, and the repeated case(s) of moving the error back through the hidden layers. */
    uint j;
    if(i==n->num_layers-2){
      tnn_cost_derivative(n,labels,err); //Output Layer

    }else {
      for(j=0; j<n->layers[i+1]; j++){ //Nodes of layer i
	uint k;
	for(k=0; k<n->layers[i+2]; k++){ //Nodes of layer i+1
	  err[i][j] = n->connections[i+1][k][j] * err[i+1][k];
	}
      }
    }
    /* Multiply by sigmoid_prime to finish error calculation. */
    for(j=0;j<n->layers[i+1]; j++){
      err[i][j] *= tnn_sigmoid_prime(n->pre_activations[i][j]);
    }
  }
}

void tnn_update_biases(Net *n, double **err, double lrate){
  /* The derivative of the cost with respect to the bias is 
     simply the error term per our definition. */
  uint i;
  for(i=0;i<n->num_layers-1;i++){ //loop over layers
    uint j;
    for(j=0;j<n->layers[i+1];j++){ //loop over current layer neurons
      n->biases[i][j] -= lrate * err[i][j];
    }
  }
}

void tnn_update_connections(Net *n, double **err, double *in, double lrate){
  /* The derivative of the cost with respect to the connections
     is simply the error times the activation. */
  uint i;
  for(i=0;i<n->num_layers-1;i++){ //loop over layers                   
    uint j;
    for(j=0;j<n->layers[i+1];j++){ //loop over next layer neurons      
      uint k;
      for(k=0;k<n->layers[i];k++){ //current layer neurons
	/* Consider building input vector into pre_activations */
	if(i==0)
	  n->connections[i][j][k] -= lrate * err[i][j] * in[k]; 
	else
	  n->connections[i][j][k] -= lrate * err[i][j] * tnn_sigmoid( n->pre_activations[i-1][k] );
      } 
    }
  }
}

void tnn_update_net_parameters(Net *n, double **err, double *in, double lrate){
  tnn_update_biases(n,err,lrate);
  tnn_update_connections(n,err,in,lrate);
}

void tnn_backprop(Net *n, double *in, double *labels, double lrate){
  /* Allocate error arrays. */
  double **err = tnn_allocate_error_arrays(n);

  /* Acquires output vec as well as pre_activations. */
  tnn_feedforward(n,in,1);

  /* Compute error at each node, chain rule. */
  tnn_generate_error(n,labels, err);

  /* Update the model. */
  tnn_update_net_parameters(n,err,in, lrate);
  
  /* Clean up. */
  uint i;
  for(i=0; i<n->num_layers-1; i++)
    free(err[i]);
  free(err);
}
