
```
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/public/session.h>

using namespace tensorflow;

int main() {
  // Create a new TensorFlow session
  Session* session;
  NewSession(SessionOptions(), &session);

  // Define the neural network graph
  Scope root = Scope::NewRootScope();
  auto x = ops::Placeholder(root, DT_FLOAT);
  auto y = ops::Placeholder(root, DT_FLOAT);
  auto w = ops::Variable(root, {1}, DT_FLOAT, ops::Constant(root, {0.5}));
  auto b = ops::Variable(root, {1}, DT_FLOAT, ops::Constant(root, {0.5}));
  auto output = ops::Add(root, ops::Mul(root, x, w), b);

  // Create a client session to run the graph
  ClientSession session(root);

  // Define the input and output tensors
  Tensor x_tensor(DT_FLOAT, TensorShape({1}));
  Tensor y_tensor(DT_FLOAT, TensorShape({1}));
  Tensor output_tensor(DT_FLOAT, TensorShape({1}));

  // Run the graph
  std::vector<Tensor> outputs;
  session.Run({{x, x_tensor}, {y, y_tensor}}, {output}, {}, &outputs);

  // Get the output tensor
  output_tensor = outputs[0];

  // Print the output
  std::cout << "Output: " << output_tensor.flat<float>()(0) << std::endl;

  return 0;
}
