#include <Arduino.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "sin_predictor.h"


#define INPUT_BUFFER_SIZE 64
#define OUTPUT_BUFFER_SIZE 64
#define INT_ARRAY_SIZE 8

// put function declarations here:
int string_to_array(char *in_str, int *int_array);
void print_int_array(int *int_array, int array_len);
int sum_array(int *int_array, int array_len);


char received_char = (char)NULL;              
int chars_avail = 0;                    // input present on terminal
char out_str_buff[OUTPUT_BUFFER_SIZE];  // strings to print to terminal
char in_str_buff[INPUT_BUFFER_SIZE];    // stores input from terminal
int input_array[INT_ARRAY_SIZE];        // array of integers input by user

int in_buff_idx=0; // tracks current input location in input buffer
int array_length=0;
int array_sum=0;

// Define tensor arena
namespace {
  constexpr int kTensorArenaSize = 2 * 1024; // You can adjust this value as needed.
  uint8_t tensor_arena[kTensorArenaSize];
}

// Global pointers for the interpreter and input tensor:
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;

void setup() {
  // put your setup code here, to run once:
  delay(5000);
  // Arduino does not have a stdout, so printf does not work easily
  // So to print fixed messages (without variables), use 
  // Serial.println() (appends new-line)  or Serial.print() (no added new-line)
  Serial.println("Test Project waking up");
  memset(in_str_buff, (char)0, INPUT_BUFFER_SIZE*sizeof(char)); 

  // Set up TFLM error reporter.
  static tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  // Map the model from the byte array
  const tflite::Model* model = tflite::GetModel(sin_predictor_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model schema version mismatch!");
    while (1);
  }

  // Create an op resolver that includes all the ops your model might need.
  static tflite::AllOpsResolver resolver;

  // Build the interpreter.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    while (1);
  }

  // Get a pointer to the model's input tensor.
  input_tensor = interpreter->input(0);
}

void loop() {
  // put your main code here, to run repeatedly:

  // check if characters are avialble on the terminal input
  chars_avail = Serial.available(); 
  if (chars_avail > 0) {
    received_char = Serial.read(); // get the typed character and 
    Serial.print(received_char);   // echo to the terminal
    in_str_buff[in_buff_idx++] = received_char; // add it to the buffer

    if (received_char == 13) { // 13 decimal = newline character
      // 'enter' was pressed, process the line.
      Serial.print("About to process line: ");
      Serial.println(in_str_buff);

      // Process and print out the array
      array_length = string_to_array(in_str_buff, input_array);

      // Check if inp 7 int
      if (array_length != 7){
        Serial.println("Warning: Please enter exactly 7 integers for the sine predictor.");
      }
      else{
        // Measure the time for printing a test statement.
        unsigned long t0 = micros();
        Serial.println("Starting inference...");
        unsigned long t1 = micros();
        
        // Copy the 7 integers into the input tensor.
        // The TFLM model is expecting int8_t values.
        for (int i = 0; i < 7; i++) {
          // You may need to scale or adjust these numbers if your model expects values in a specific range.
          input_tensor->data.int8[i] = (int8_t)input_array[i];
        }
      // Run inference.
      if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("Error during inference.");
      } else {
        // Get the prediction from the output tensor.
        TfLiteTensor* output_tensor = interpreter->output(0);
        int8_t prediction = output_tensor->data.int8[0];
        unsigned long t2 = micros();
        
        // Calculate elapsed times.
        unsigned long t_print = t1 - t0;
        unsigned long t_infer = t2 - t1;

        // Print out the results.
        Serial.print("Prediction: ");
        Serial.println(prediction);
        Serial.print("Printing time = ");
        Serial.print(t_print);
        Serial.print(" us. Inference time = ");
        Serial.print(t_infer);
        Serial.println(" us.");
      }
    }
      // Now clear the input buffer and reset the index to 0
      memset(in_str_buff, 0, INPUT_BUFFER_SIZE); 
      in_buff_idx = 0;
    }
    else if (in_buff_idx >= INPUT_BUFFER_SIZE) {
      memset(in_str_buff, 0, INPUT_BUFFER_SIZE); 
      in_buff_idx = 0;
    }    
  }
}


int string_to_array(char *in_str, int *int_array) {
  int num_integers=0;
  char *token = strtok(in_str, ",");
  
  while (token != NULL) {
    int_array[num_integers++] = atoi(token);
    token = strtok(NULL, ",");
    if (num_integers >= INT_ARRAY_SIZE) {
      break;
    }
  }
  
  return num_integers;
}

void print_int_array(int *int_array, int array_len) {
  int curr_pos = 0; // track where in the output buffer we're writing

  sprintf(out_str_buff, "Integers: [");
  curr_pos = strlen(out_str_buff); // so the next write adds to the end
  for(int i=0;i<array_len;i++) {
    // sprintf returns number of char's written. use it to update current position
    curr_pos += sprintf(out_str_buff+curr_pos, "%d, ", int_array[i]);
  }
  sprintf(out_str_buff+curr_pos, "]\r\n");
  Serial.print(out_str_buff);
}

int sum_array(int *int_array, int array_len) {
  int curr_sum = 0; // running sum of the array

  for(int i=0;i<array_len;i++) {
    curr_sum += int_array[i];
  }
  return curr_sum;
}
