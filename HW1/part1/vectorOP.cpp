#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //
  int flag;
  __pp_vec_float x;
  __pp_vec_int y;
  __pp_vec_float result;
  __pp_vec_int count;
  __pp_vec_int zero = _pp_vset_int(0);
  __pp_vec_int one = _pp_vset_int(1);
  __pp_vec_float onef = _pp_vset_float(1.f);
  __pp_vec_float largest = _pp_vset_float(9.999999f);
  __pp_mask maskWrite, maskAll, maskIsEqual, maskIsNotEqual, maskIsCountGreater, maskIsResultGreater;

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskWrite = _pp_init_ones(N-i);
    maskAll = _pp_init_ones();

    // All zeros
    maskIsEqual = _pp_init_ones(0);
    maskIsCountGreater = _pp_init_ones(0);
    maskIsResultGreater = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Load vector of values from contiguous memory addresses
    _pp_vload_int(y, exponents + i, maskAll); // y = values[i];

    // Set mask according to predicate
    _pp_veq_int(maskIsEqual, y, zero, maskAll); // if (y == 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vmove_float(result, onef, maskIsEqual); //   output[i] = 1.f;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotEqual = _pp_mask_not(maskIsEqual); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotEqual); //   result = x; 

    // Execute instruction using mask ("if" clause)
    _pp_vsub_int(count, y, one, maskIsNotEqual); //   count = y - 1;

    _pp_vgt_int(maskIsCountGreater, count, zero, maskIsNotEqual);  // while (count > 0){

    flag = _pp_cntbits(maskIsCountGreater);

    while(flag){

      _pp_vmult_float(result, result, x, maskIsCountGreater); //  result *= x;

      _pp_vsub_int(count, count, one, maskIsCountGreater); //  count--;

      _pp_vgt_int(maskIsCountGreater, count, zero, maskIsNotEqual);  

      flag = _pp_cntbits(maskIsCountGreater);

    }

    // Set mask according to predicate
    _pp_vgt_float(maskIsResultGreater, result, largest, maskIsNotEqual); // if (result > 9.999999f){

    // Execute instruction using mask ("if" clause)
    _pp_vmove_float(result, largest, maskIsResultGreater); //  result = 9.999999f;}

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskWrite);

  }


}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{

  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //
  float *output = new float[N+VECTOR_WIDTH];
  int NN=N;  
  __pp_vec_float x;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_vec_float addResult;
  __pp_vec_float interleaveResult;
  __pp_mask maskAll;

  // All ones
  maskAll = _pp_init_ones();

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    _pp_vload_float(x, values + i, maskAll); 

    _pp_hadd_float(addResult, x);
    _pp_interleave_float(interleaveResult, addResult);

    _pp_vstore_float(output + i/2, interleaveResult, maskAll);

  }


  NN=ceil((float)NN/2);
  _pp_vstore_float(output + NN, zero, maskAll);

  while(NN!=1){
    for (int i = 0; i < NN; i += VECTOR_WIDTH)
    {

      _pp_vload_float(x, output + i, maskAll); 

      _pp_hadd_float(addResult, x);
      _pp_interleave_float(interleaveResult, addResult);

      _pp_vstore_float(output + i/2, interleaveResult, maskAll);

    }

    NN=ceil((float)NN/2);
    _pp_vstore_float(output + NN, zero, maskAll);

  }

  return output[0];
}
