__kernel void add_numbers(__global float4* data, 
      __local float* local_result, __global float* group_result) {

   float sum;
   float4 input1, input2, sum_vector; /* float4 can be operated in the same clock cycle on supported devices */
   uint global_addr, local_addr;

   global_addr = get_global_id(0) * 2;
   input1 = data[global_addr]; /* Loads 4 floats */
   input2 = data[global_addr+1]; /* Loads another 4 floats */
   sum_vector = input1 + input2;

   local_addr = get_local_id(0); /*Store site for the sum of values */
   local_result[local_addr] = sum_vector.s0 + sum_vector.s1 + 
                              sum_vector.s2 + sum_vector.s3; 
   barrier(CLK_LOCAL_MEM_FENCE); /* All work-items in a work-group executing the kernel on a processor must execute this function before any are allowed to continue execution beyond the barrier. */


   /* One Work item sum all the results */
   if(get_local_id(0) == 0) {
      sum = 0.0f;
      for(int i=0; i<get_local_size(0); i++) {
         sum += local_result[i];
      }
      group_result[get_group_id(0)] = sum;
   }
}
