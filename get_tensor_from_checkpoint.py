#!/usr/bin/env python
# -*- coding:utf-8 -*-
#auther:jf183
#datetime:2019/1/3 19:43

from tensorflow.python import pywrap_tensorflow
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--checkpoint", help="the path of your downloaded checkpoint file.")

def return_tensors_in_checkpoint_file(file_name, tensor_name=None, all_tensors=False,
                                     all_tensor_names=False):
  """Prints tensors in a checkpoint file.
  If no `tensor_name` is provided, prints the tensor names and shapes
  in the checkpoint file.
  If `tensor_name` is provided, prints the content of the tensor.
  Args:
    file_name: Name of the checkpoint file.
    tensor_name: Name of the tensor in the checkpoint file to print.
    all_tensors: Boolean indicating whether to print all tensors.
    all_tensor_names: Boolean indicating whether to print all tensor names.
  """
  try:
      reader = pywrap_tensorflow.NewCheckpointReader(file_name)

      if all_tensors or all_tensor_names:
          var_to_shape_map = reader.get_variable_to_shape_map()
          #print('The type of var_to_shape_map:', type(var_to_shape_map))

          if all_tensors:
              tensor_dict = dict()
              for key in sorted(var_to_shape_map):
                  tensor_dict[key] = reader.get_tensor(key)
              return tensor_dict

          return var_to_shape_map
      elif not tensor_name:
          print(reader.debug_string().decode("utf-8"))
      else:
          #print("tensor_name: ", tensor_name)
          tensor_var = reader.get_tensor(tensor_name)
          return tensor_var
  except Exception as e:  # pylint: disable=broad-except
      print(str(e))
      if "corrupted compressed block contents" in str(e):
        print("It's likely that your checkpoint file has been compressed with SNAPPY.")
      if ("Data loss" in str(e) and any(e in file_name for e in [".index", ".meta", ".data"])):
          proposed_file = ".".join(file_name.split(".")[0:-1])
          v2_file_error_template = """
It's likely that this is a V2 checkpoint and you need to provide the filename
*prefix*.  Try removing the '.' and extension.  Try:
inspect checkpoint --file_name = {}"""
          print(v2_file_error_template.format(proposed_file))

def _main_(args):
    checkpoint_path = args.checkpoint
    # tensor_name = 'MobilenetV1/Logits/Conv2d_1c_1x1/biases'
    print("Here are the tensors in this checkpoint file.")
    tensor_var = return_tensors_in_checkpoint_file(checkpoint_path, all_tensor_names=True)
    for key in tensor_var:
        print(key)

if __name__ == '__main__':
    args = parser.parse_args()
    _main_(args)

