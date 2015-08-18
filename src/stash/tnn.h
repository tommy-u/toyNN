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

void tnn_cost_derivative(Net *n, double *labels, double **err);
void tnn_init_connections(Net *n);
void tnn_init_biases(Net *n);
void tnn_destroy_net(Net *n);
void tnn_feedforward(Net *n, double *input, uint train);
void tnn_print_net(Net *n);
void tnn_print_output_activation(Net *n, double *in);
void tnn_print_pre_activations(Net *n);
void tnn_generate_error(Net *n, double *labels, double **err);
void tnn_update_biases(Net *n, double **err, double lrate);
void tnn_update_connections(Net *n, double **err, double *in, double lrate);
void tnn_update_net_parameters(Net *n, double **err, double *in, double lrate);
void tnn_backprop(Net *n, double *in, double *labels, double lrate);
double tnn_rand_norm(const double mean, const double std);
double tnn_sigmoid(double z);
double tnn_sigmoid_prime(double z);
double ** tnn_allocate_error_arrays(Net *n);
Net * tnn_init_net(uint num_layers, uint *layers);
