#include <stdio.h>
#include <math.h>
#include <assert.h>
#define BPT 6
#define SZZ sizeof(int) * 8
typedef struct {
  int val;
  int pt;
  
}fxp;

void print_fxp(fxp f){
  int i;
  for(i= SZZ - 1; i>=0 ; i--){
    printf("%d",(f.val  >> i) & 1 );
    if(i==f.pt)
      printf(".");
  }
  printf("\n");
}

fxp double_to_fxp(double d){
  //assume no overflow
  // Forget about fractional component 
  //Largest value is 2^(32 - BPT - 1)
  //Smallest value is 2^(32 - BPT - 1)
  printf("min %f\n",pow(2, SZZ - BPT -1));
  printf("max %f\n",-pow(2, SZZ - BPT -1));
  assert(d < pow(2, SZZ - BPT -1));
  assert(d > -pow(2, SZZ - BPT -1));

  fxp f;
  f.pt = BPT;
  
  return f;
}

double fxp_to_double(fxp f){
  double a=0;
  int i;
  //  101.101
  for(i=SZZ - 1; i>=0 ;i--){
    int bit=(f.val  >> i) & 1;
    a += bit * pow(2, i - f.pt ) ;
  }
  return a;
}

int main(){
  
  fxp a;
  //  10.1 = 2.5
  //  5/1
  a.val=5;
  a.pt =1;
 
  fxp b;
  //  10.01 = 2.25
  //  9/2
  b.val=9;
  b.pt =2;

  fxp c;
  //  110.11
  c.val= a.val*b.val;
  c.pt=0;

  //  print_fxp(c);
  //  printf("%f \n",fxp_to_double(a));
  double_to_fxp(5);
  


  return 0;
}
