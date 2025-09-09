# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""TensorFlow Lite frontend for Relax."""

import itertools
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tvm
from tvm import relax, tir
from tvm.ir import IRModule
from tvm.ir.supply import NameSupply

__all__ = ["from_tflite"]


class TensorWrapper:
    """Tensor wrapper for TFLite Tensor"""

    def __init__(self, tensor_idx, tensor, buffer, qnn_params=None):
        self.tensor_idx = tensor_idx
        self.tensor = tensor
        self.buffer = buffer
        self.qnn_params = qnn_params


def get_scalar_from_constant(expr):
    """Returns scalar value from Relax constant scalar."""
    if isinstance(expr, relax.Constant):
        value = expr.data.numpy()
        return value.item() if value.shape == () else value.item(0)
    elif isinstance(expr, relax.PrimValue):
        return expr.value.value if hasattr(expr.value, 'value') else expr.value
    else:
        raise ValueError(f"Expected Constant or PrimValue, got {type(expr)}")


def get_tensor_from_constant(expr):
    """Returns tensor of values from Relax constant node."""
    if isinstance(expr, relax.Constant):
        return expr.data.numpy()
    else:
        raise ValueError(f"Expected Constant, got {type(expr)}")


def build_str_map(obj):
    """Build string map of TFLite enum int value"""
    ret = {}
    for field_name in dir(obj):
        if not field_name.startswith("_"):
            field_value = getattr(obj, field_name)
            if isinstance(field_value, int):
                ret[field_value] = field_name
    return ret


def get_pad_value(data, kernel, stride):
    """Get the pad tuple of value for SAME padding"""
    out = int(math.ceil(float(data) / float(stride)))
    pad = max(0, (out - 1) * stride + kernel - data)
    pad_before = pad // 2
    pad_after = pad - pad_before
    return pad_before, pad_after


def get_tensor_name(subgraph, tensor_idx):
    """Get the tensor name."""
    tensor_name = subgraph.Tensors(tensor_idx).Name()
    if tensor_name is not None:
        tensor_name = tensor_name.decode("utf-8")
    else:
        tensor_name = "tvmgen_tensor_" + str(tensor_idx)
    return tensor_name


def to_int_list(arr):
    """Convert numpy array or list to python int list"""
    if isinstance(arr, np.ndarray):
        return arr.astype(int).tolist()
    elif isinstance(arr, (list, tuple)):
        return [int(x) for x in arr]
    else:
        return [int(arr)]


class TFLiteGraphImporter:
    """A helper class for handling Relax expression conversion from TFLite model."""

    def __init__(
        self,
        shape_dict: Optional[Dict[str, List]] = None,
        dtype_dict: Optional[Union[str, Dict[str, str]]] = "float32",
    ):
        self._nodes: Dict[str, relax.Expr] = {}
        self._inputs: Dict[str, relax.Var] = {}
        self._num_input: int = 0
        self._shape = shape_dict.copy() if shape_dict else {}
        self._dtype = dtype_dict
        self._name_supply = NameSupply()
        self.bb: relax.BlockBuilder = relax.BlockBuilder()
        self._params = {}
        self._prefetched_nodes = {}

        # Build operator maps
        try:
            from tflite.ActivationFunctionType import ActivationFunctionType
            from tflite.BuiltinOperator import BuiltinOperator
            from tflite.BuiltinOptions import BuiltinOptions
        except ImportError:
            raise ImportError("The tflite package must be installed")

        self.builtin_op_code = build_str_map(BuiltinOperator())
        self.activation_fn_type = build_str_map(ActivationFunctionType())
        self.builtin_options = build_str_map(BuiltinOptions())

        # Initialize converter map
        self._init_convert_map()

    def _init_convert_map(self):
        """Initialize the operator conversion map."""
        self.convert_map = {
            "ABS": self.convert_abs,
            "ADD": self.convert_add,
            "AVERAGE_POOL_2D": self.convert_average_pool2d,
            "CAST": self.convert_cast,
            "CONCATENATION": self.convert_concatenation,
            "CONV_2D": self.convert_conv2d,
            "DEPTHWISE_CONV_2D": self.convert_depthwise_conv2d,
            "DEQUANTIZE": self.convert_dequantize,
            "DIV": self.convert_div,
            "EQUAL": self.convert_equal,
            "EXP": self.convert_exp,
            "FULLY_CONNECTED": self.convert_fully_connected,
            "GATHER": self.convert_gather,
            "GREATER": self.convert_greater,
            "LESS": self.convert_less,
            "LOGISTIC": self.convert_logistic,
            "MAX_POOL_2D": self.convert_max_pool2d,
            "MAXIMUM": self.convert_maximum,
            "MEAN": self.convert_reduce_mean,
            "MINIMUM": self.convert_minimum,
            "MUL": self.convert_mul,
            "NEG": self.convert_neg,
            "RELU": self.convert_relu,
            "RELU6": self.convert_relu6,
            "RESHAPE": self.convert_reshape,
            "SOFTMAX": self.convert_softmax,
            "SQUEEZE": self.convert_squeeze,
            "SUB": self.convert_sub,
            "TANH": self.convert_tanh,
            "TRANSPOSE": self.convert_transpose,
        }

    def from_tflite(self, model) -> Tuple[IRModule, Dict[str, tvm.nd.NDArray]]:
        """Construct Relax expressions from the TFLite model."""
        # Store model reference for use in other methods
        self.current_model = model

        with self.bb.function("main"):
            with self.bb.dataflow() as df:
                self._parse_model_inputs(model)
                self._convert_operators(model)

                # Get outputs
                subgraph = model.Subgraphs(0)
                model_outputs = subgraph.OutputsAsNumpy()
                outputs = [self._nodes[get_tensor_name(subgraph, i)] for i in model_outputs]
                outputs = outputs[0] if len(outputs) == 1 else relax.Tuple(outputs)

                output_var = self.bb.emit_output(outputs)

            # Create function
            input_list = [var for var in self._inputs.values() if isinstance(var, relax.Var)]
            self.bb.emit_func_output(output_var, params=input_list)

        return self.bb.get(), self._params

    def _parse_model_inputs(self, model):
        """Parse model inputs and create Relax variables."""
        subgraph = model.Subgraphs(0)
        model_inputs = subgraph.InputsAsNumpy()

        for model_input in model_inputs:
            input_name = get_tensor_name(subgraph, model_input)
            tensor = subgraph.Tensors(model_input)
            
            # Get shape and dtype
            if input_name in self._shape:
                shape = self._shape[input_name]
            else:
                shape = tuple(tensor.ShapeAsNumpy()) if tensor.ShapeLength() > 0 else ()
            
            if isinstance(self._dtype, dict) and input_name in self._dtype:
                dtype = self._dtype[input_name]
            elif isinstance(self._dtype, str):
                dtype = self._dtype
            else:
                dtype = self._get_tensor_type_str(tensor.Type())

            # Create Relax variable
            input_var = relax.Var(
                name_hint=input_name,
                struct_info=relax.TensorStructInfo(shape=shape, dtype=dtype)
            )
            
            self._nodes[input_name] = input_var
            self._inputs[input_name] = input_var
            self._num_input += 1

    def _convert_operators(self, model):
        """Convert TFLite operators to Relax expressions."""
        subgraph = model.Subgraphs(0)
        
        for op_idx in range(subgraph.OperatorsLength()):
            op = subgraph.Operators(op_idx)
            op_code_str = self._get_op_code_str(model, op)
            
            if op_code_str not in self.convert_map:
                raise NotImplementedError(f"Operator {op_code_str} is not supported yet")

            # Convert operator
            result = self.convert_map[op_code_str](subgraph, op)
            
            if result is not None:
                # Get output tensors
                output_tensors = self._get_output_tensors(subgraph, op)
                
                if len(output_tensors) == 1:
                    output_name = get_tensor_name(subgraph, output_tensors[0].tensor_idx)
                    self._nodes[output_name] = result
                else:
                    for i, output_tensor in enumerate(output_tensors):
                        output_name = get_tensor_name(subgraph, output_tensor.tensor_idx)
                        self._nodes[output_name] = result[i]

    def _get_op_code_str(self, model, op):
        """Get TFLite operator code string."""
        try:
            from tflite.BuiltinOperator import BuiltinOperator
        except ImportError:
            raise ImportError("The tflite package must be installed")

        op_code_list_idx = op.OpcodeIndex()
        op_c = model.OperatorCodes(op_code_list_idx)
        
        # Handle different TFLite versions
        try:
            opc = max(op_c.DeprecatedBuiltinCode(), op_c.BuiltinCode())
        except AttributeError:
            opc = op_c.BuiltinCode()

        try:
            op_code_str = self.builtin_op_code[opc]
        except KeyError:
            raise NotImplementedError(f"TFLite operator with code {opc} is not supported")
            
        if opc == BuiltinOperator.CUSTOM:
            raise NotImplementedError("Custom operators are not supported yet")
            
        return op_code_str

    def _get_input_tensors(self, subgraph, op):
        """Get input tensors for an operator."""
        operator_inputs = op.InputsAsNumpy()
        return self._get_tensors(subgraph, operator_inputs)

    def _get_output_tensors(self, subgraph, op):
        """Get output tensors for an operator."""
        operator_outputs = op.OutputsAsNumpy()
        return self._get_tensors(subgraph, operator_outputs)

    def _get_tensors(self, subgraph, tensor_indices):
        """Get tensor wrappers from tensor indices."""
        return_list = []
        for tensor_idx in tensor_indices:
            if tensor_idx < 0:
                return_list.append(TensorWrapper(tensor_idx, None, None))
                continue

            tensor = subgraph.Tensors(tensor_idx)
            buffer_idx = tensor.Buffer()
            # The `subgraph.Model()` method is not standard, so we access it from the main model
            buffer = self.current_model.Buffers(buffer_idx) if buffer_idx < self.current_model.BuffersLength() else None
            
            # Handle quantization parameters
            qnn_params = self._parse_qnn_params(tensor)
            
            return_list.append(TensorWrapper(tensor_idx, tensor, buffer, qnn_params))
        
        return return_list

    def _parse_qnn_params(self, tensor):
        """Parse quantization parameters from tensor."""
        qnn_params = None
        tflite_qnn_params = tensor.Quantization()
        
        if tflite_qnn_params is not None:
            tflite_scale = tflite_qnn_params.ScaleAsNumpy()
            tflite_zero_point = tflite_qnn_params.ZeroPointAsNumpy()
            
            if isinstance(tflite_scale, np.ndarray) and tflite_scale.size > 0:
                if tflite_scale.size == 1 and tflite_zero_point.size == 1:
                    scale = float(tflite_scale[0])
                    zero_point = int(tflite_zero_point[0])
                    
                    if scale != 0 or zero_point != 0:
                        qnn_params = {
                            "scale": relax.const(scale, "float32"),
                            "zero_point": relax.const(zero_point, "int32")
                        }
                        
        return qnn_params

    def _get_tensor_type_str(self, tensor_type):
        """Get tensor type string from TFLite tensor type."""
        try:
            from tflite.TensorType import TensorType
        except ImportError:
            raise ImportError("The tflite package must be installed")

        type_map = {
            TensorType.INT8: "int8",
            TensorType.INT16: "int16",
            TensorType.UINT8: "uint8",
            TensorType.FLOAT16: "float16",
            TensorType.FLOAT32: "float32",
            TensorType.INT32: "int32",
            TensorType.INT64: "int64",
            TensorType.BOOL: "bool",
        }
        
        if tensor_type in type_map:
            return type_map[tensor_type]
        else:
            raise NotImplementedError(f"Tensor type {tensor_type} is not supported")

    def _has_tensor_value(self, tensor_wrapper):
        """Check if a tensor has a constant value."""
        return tensor_wrapper.buffer is not None and tensor_wrapper.buffer.DataLength() > 0

    def _get_tensor_value(self, tensor_wrapper):
        """Get tensor buffer value from tensor wrapper."""
        if not self._has_tensor_value(tensor_wrapper):
            return None
            
        dtype = self._get_numpy_dtype(tensor_wrapper.tensor.Type())
        data = tensor_wrapper.buffer.DataAsNumpy()
        
        if tensor_wrapper.tensor.ShapeLength() != 0:
            shape = to_int_list(tensor_wrapper.tensor.ShapeAsNumpy())
        else:
            shape = []

        return np.frombuffer(data, dtype=dtype).reshape(shape)

    def _get_numpy_dtype(self, tensor_type):
        """Get numpy dtype from TFLite tensor type."""
        try:
            from tflite.TensorType import TensorType
            
            type_map = {
                TensorType.UINT8: np.uint8,
                TensorType.INT8: np.int8,
                TensorType.INT16: np.int16,
                TensorType.FLOAT16: np.float16,
                TensorType.FLOAT32: np.float32,
                TensorType.INT32: np.int32,
                TensorType.INT64: np.int64,
                TensorType.BOOL: np.bool_,
            }
            
            return type_map[tensor_type]
        except KeyError:
            raise NotImplementedError(f"Tensor type {tensor_type} is not supported")

    def _get_tensor_expr(self, tensor_wrapper):
        """Get Relax expression for a tensor."""
        tensor_name = get_tensor_name(self.current_subgraph, tensor_wrapper.tensor_idx)

        if tensor_name in self._nodes:
            return self._nodes[tensor_name]
        else:
            # Create constant and treat it as a parameter
            if not self._has_tensor_value(tensor_wrapper):
                raise ValueError(f"Tensor '{tensor_name}' is not an input and has no constant value.")

            value = self._get_tensor_value(tensor_wrapper)
            dtype = self._get_tensor_type_str(tensor_wrapper.tensor.Type())

            param_var = relax.Var(tensor_name, relax.TensorStructInfo(value.shape, dtype))
            self._nodes[tensor_name] = param_var
            self._params[tensor_name] = tvm.nd.array(value)
            return param_var

    # Operator conversion methods
    def convert_abs(self, subgraph, op):
        """Convert TFLite ABS operator."""
        self.current_subgraph = subgraph
        input_tensors = self._get_input_tensors(subgraph, op)
        assert len(input_tensors) == 1
        
        input_expr = self._get_tensor_expr(input_tensors[0])
        return self.bb.normalize(relax.op.abs(input_expr))

    def convert_add(self, subgraph, op):
        """Convert TFLite ADD operator."""
        self.current_subgraph = subgraph
        input_tensors = self._get_input_tensors(subgraph, op)
        assert len(input_tensors) == 2
        
        lhs = self._get_tensor_expr(input_tensors[0])
        rhs = self._get_tensor_expr(input_tensors[1])
        return self.bb.normalize(relax.op.add(lhs, rhs))

    def convert_sub(self, subgraph, op):
        """Convert TFLite SUB operator."""
        self.current_subgraph = subgraph
        input_tensors = self._get_input_tensors(subgraph, op)
        assert len(input_tensors) == 2
        
        lhs = self._get_tensor_expr(input_tensors[0])
        rhs = self._get_tensor_expr(input_tensors[1])
        return self.bb.normalize(relax.op.subtract(lhs, rhs))

    def convert_mul(self, subgraph, op):
        """Convert TFLite MUL operator."""
        self.current_subgraph = subgraph
        input_tensors = self._get_input_tensors(subgraph, op)
        assert len(input_tensors) == 2
        
        lhs = self._get_tensor_expr(input_tensors[0])
        rhs = self._get_tensor_expr(input_tensors[1])
        return self.bb.normalize(relax.op.multiply(lhs, rhs))

    def convert_div(self, subgraph, op):
        """Convert TFLite DIV operator."""
        self.current_subgraph = subgraph
        input_tensors = self._get_input_tensors(subgraph, op)
        assert len(input_tensors) == 2
        
        lhs = self._get_tensor_expr(input_tensors[0])
        rhs = self._get_tensor_expr(input_tensors[1])
        return self.bb.normalize(relax.op.divide(lhs, rhs))

    def convert_maximum(self, subgraph, op):
        """Convert TFLite MAXIMUM operator."""
        self.current_subgraph = subgraph
        input_tensors = self._get_input_tensors(subgraph, op)
        assert len(input_tensors) == 2
        
        lhs = self._get_tensor_expr(input_tensors[0])
        rhs = self._get_tensor_expr(input_tensors[1])
        return self.bb.normalize(relax.op.maximum(lhs, rhs))

    def convert_minimum(self, subgraph, op):
        """Convert TFLite MINIMUM operator."""
        self.current_subgraph = subgraph
        input_tensors = self._get_input_tensors(subgraph, op)
        assert len(input_tensors) == 2
        
        lhs = self._get_tensor_expr(input_tensors[0])
        rhs = self._get_tensor_expr(input_tensors[1])
        return self.bb.normalize(relax.op.minimum(lhs, rhs))

    def convert_equal(self, subgraph, op):
        """Convert TFLite EQUAL operator."""
        self.current_subgraph = subgraph
        input_tensors = self._get_input_tensors(subgraph, op)
        assert len(input_tensors) == 2
        
        lhs = self._get_tensor_expr(input_tensors[0])
        rhs = self._get_tensor_expr(input_tensors[1])
        return self.bb.normalize(relax.op.equal(lhs, rhs))

    def convert_greater(self, subgraph, op):
        """Convert TFLite GREATER operator."""
        self.current_subgraph = subgraph
        input_tensors = self._get_input_tensors(subgraph, op)
        assert len(input_tensors) == 2
        
        lhs = self._get_tensor_expr(input_tensors[0])
        rhs = self._get_tensor_expr(input_tensors[1])
        return self.bb.normalize(relax.op.greater(lhs, rhs))

    def convert_less(self, subgraph, op):
        """Convert TFLite LESS operator."""
        self.current_subgraph = subgraph
        input_tensors = self._get_input_tensors(subgraph, op)
        assert len(input_tensors) == 2
        
        lhs = self._get_tensor_expr(input_tensors[0])
        rhs = self._get_tensor_expr(input_tensors[1])
        return self.bb.normalize(relax.op.less(lhs, rhs))

    def convert_neg(self, subgraph, op):
        """Convert TFLite NEG operator."""
        self.current_subgraph = subgraph
        input_tensors = self._get_input_tensors(subgraph, op)
        assert len(input_tensors) == 1
        
        input_expr = self._get_tensor_expr(input_tensors[0])
        return self.bb.normalize(relax.op.negative(input_expr))

    def convert_exp(self, subgraph, op):
        """Convert TFLite EXP operator."""
        self.current_subgraph = subgraph
        input_tensors = self._get_input_tensors(subgraph, op)
        assert len(input_tensors) == 1
        
        input_expr = self._get_tensor_expr(input_tensors[0])
        return self.bb.normalize(relax.op.exp(input_expr))

    def convert_tanh(self, subgraph, op):
        """Convert TFLite TANH operator."""
        self.current_subgraph = subgraph
        input_tensors = self._get_input_tensors(subgraph, op)
        assert len(input_tensors) == 1
        
        input_expr = self._get_tensor_expr(input_tensors[0])
        return self.bb.normalize(relax.op.tanh(input_expr))

    def convert_relu(self, subgraph, op):
        """Convert TFLite RELU operator."""
        self.current_subgraph = subgraph
        input_tensors = self._get_input_tensors(subgraph, op)
        assert len(input_tensors) == 1
        
        input_expr = self._get_tensor_expr(input_tensors[0])
        return self.bb.normalize(relax.op.nn.relu(input_expr))

    def convert_relu6(self, subgraph, op):
        """Convert TFLite RELU6 operator."""
        self.current_subgraph = subgraph
        input_tensors = self._get_input_tensors(subgraph, op)
        assert len(input_tensors) == 1
        
        input_expr = self._get_tensor_expr(input_tensors[0])
        return self.bb.normalize(relax.op.clip(input_expr, 0, 6))

    def convert_logistic(self, subgraph, op):
        """Convert TFLite LOGISTIC operator."""
        self.current_subgraph = subgraph
        input_tensors = self._get_input_tensors(subgraph, op)
        assert len(input_tensors) == 1
        
        input_expr = self._get_tensor_expr(input_tensors[0])
        return self.bb.normalize(relax.op.sigmoid(input_expr))

    def convert_softmax(self, subgraph, op):
        """Convert TFLite SOFTMAX operator."""
        self.current_subgraph = subgraph
        input_tensors = self._get_input_tensors(subgraph, op)
        assert len(input_tensors) == 1
        
        input_expr = self._get_tensor_expr(input_tensors[0])
        return self.bb.normalize(relax.op.nn.softmax(input_expr, axis=-1))

    def convert_reshape(self, subgraph, op):
        """Convert TFLite RESHAPE operator."""
        self.current_subgraph = subgraph
        input_tensors = self._get_input_tensors(subgraph, op)
        
        data_expr = self._get_tensor_expr(input_tensors[0])

        # The new shape can be provided as a second input tensor or in the options
        if len(input_tensors) == 2 and self._has_tensor_value(input_tensors[1]):
            shape_tensor = input_tensors[1]
            new_shape = self._get_tensor_value(shape_tensor).tolist()
        else:
            # Get shape from operator options
            try:
                from tflite.BuiltinOptions import BuiltinOptions
                from tflite.ReshapeOptions import ReshapeOptions
            except ImportError:
                raise ImportError("The tflite package must be installed")
                
            assert op.BuiltinOptionsType() == BuiltinOptions.ReshapeOptions
            op_options = op.BuiltinOptions()
            reshape_options = ReshapeOptions()
            reshape_options.Init(op_options.Bytes, op_options.Pos)
            new_shape = to_int_list(reshape_options.NewShapeAsNumpy())

        return self.bb.normalize(relax.op.reshape(data_expr, new_shape))

    def convert_transpose(self, subgraph, op):
        """Convert TFLite TRANSPOSE operator."""
        self.current_subgraph = subgraph
        input_tensors = self._get_input_tensors(subgraph, op)
        assert len(input_tensors) == 2
        
        data_expr = self._get_tensor_expr(input_tensors[0])
        perm = self._get_tensor_value(input_tensors[1])
        if perm is not None:
            perm = tuple(perm.tolist())
        
        return self.bb.normalize(relax.op.permute_dims(data_expr, perm))

    def convert_squeeze(self, subgraph, op):
        """Convert TFLite SQUEEZE operator."""
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.SqueezeOptions import SqueezeOptions
        except ImportError:
            raise ImportError("The tflite package must be installed")
            
        self.current_subgraph = subgraph
        input_tensors = self._get_input_tensors(subgraph, op)
        assert len(input_tensors) == 1
        
        data_expr = self._get_tensor_expr(input_tensors[0])
        
        assert op.BuiltinOptionsType() == BuiltinOptions.SqueezeOptions
        op_options = op.BuiltinOptions()
        squeeze_options = SqueezeOptions()
        squeeze_options.Init(op_options.Bytes, op_options.Pos)
        squeeze_axes = squeeze_options.SqueezeDimsAsNumpy()
        
        if len(squeeze_axes) > 0:
            axes = tuple(squeeze_axes.tolist())
        else:
            axes = None
            
        return self.bb.normalize(relax.op.squeeze(data_expr, axis=axes))

    def convert_concatenation(self, subgraph, op):
        """Convert TFLite CONCATENATION operator."""
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.ConcatenationOptions import ConcatenationOptions
        except ImportError:
            raise ImportError("The tflite package must be installed")
            
        self.current_subgraph = subgraph
        input_tensors = self._get_input_tensors(subgraph, op)
        
        input_exprs = [self._get_tensor_expr(t) for t in input_tensors]
        
        assert op.BuiltinOptionsType() == BuiltinOptions.ConcatenationOptions
        op_options = op.BuiltinOptions()
        concat_options = ConcatenationOptions()
        concat_options.Init(op_options.Bytes, op_options.Pos)
        axis = concat_options.Axis()
        
        return self.bb.normalize(relax.op.concat(input_exprs, axis=axis))

    def convert_gather(self, subgraph, op):
        """Convert TFLite GATHER operator."""
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.GatherOptions import GatherOptions
        except ImportError:
            raise ImportError("The tflite package must be installed")
            
        self.current_subgraph = subgraph
        input_tensors = self._get_input_tensors(subgraph, op)
        assert len(input_tensors) == 2
        
        data_expr = self._get_tensor_expr(input_tensors[0])
        indices_expr = self._get_tensor_expr(input_tensors[1])
        
        assert op.BuiltinOptionsType() == BuiltinOptions.GatherOptions
        op_options = op.BuiltinOptions()
        gather_options = GatherOptions()
        gather_options.Init(op_options.Bytes, op_options.Pos)
        axis = gather_options.Axis()
        
        return self.bb.normalize(relax.op.take(data_expr, indices_expr, axis=axis))

    def convert_cast(self, subgraph, op):
        """Convert TFLite CAST operator."""
        self.current_subgraph = subgraph
        input_tensors = self._get_input_tensors(subgraph, op)
        assert len(input_tensors) == 1
        
        input_expr = self._get_tensor_expr(input_tensors[0])
        
        # Get output type from output tensor
        output_tensors = self._get_output_tensors(subgraph, op)
        output_dtype = self._get_tensor_type_str(output_tensors[0].tensor.Type())
        
        return self.bb.normalize(relax.op.astype(input_expr, output_dtype))

    def convert_reduce_mean(self, subgraph, op):
        """Convert TFLite MEAN operator."""
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.ReducerOptions import ReducerOptions
        except ImportError:
            raise ImportError("The tflite package must be installed")
            
        self.current_subgraph = subgraph
        input_tensors = self._get_input_tensors(subgraph, op)
        assert len(input_tensors) == 2
        
        data_expr = self._get_tensor_expr(input_tensors[0])
        
        # Handle axes tensor
        axes_tensor = input_tensors[1]
        if self._has_tensor_value(axes_tensor):
            axes_value = self._get_tensor_value(axes_tensor)
            if axes_value is not None:
                axes = tuple(axes_value.tolist()) if axes_value.size > 0 else None
            else:
                axes = None  # Use default behavior
        else:
            axes = None  # Use default behavior for dynamic case
        
        keepdims = False
        if op.BuiltinOptionsType() == BuiltinOptions.ReducerOptions:
            op_options = op.BuiltinOptions()
            reducer_options = ReducerOptions()
            reducer_options.Init(op_options.Bytes, op_options.Pos)
            keepdims = reducer_options.KeepDims()
        
        return self.bb.normalize(relax.op.mean(data_expr, axis=axes, keepdims=keepdims))

    # Pool operators
    def convert_average_pool2d(self, subgraph, op):
        """Convert TFLite AVERAGE_POOL_2D operator."""
        return self._convert_pool2d(subgraph, op, "avg")

    def convert_max_pool2d(self, subgraph, op):
        """Convert TFLite MAX_POOL_2D operator."""
        return self._convert_pool2d(subgraph, op, "max")

    def _convert_pool2d(self, subgraph, op, pool_type):
        """Generic pool2d conversion."""
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.Pool2DOptions import Pool2DOptions
            from tflite.Padding import Padding
        except ImportError:
            raise ImportError("The tflite package must be installed")
            
        self.current_subgraph = subgraph
        input_tensors = self._get_input_tensors(subgraph, op)
        assert len(input_tensors) == 1
        
        data_expr = self._get_tensor_expr(input_tensors[0])
        
        assert op.BuiltinOptionsType() == BuiltinOptions.Pool2DOptions
        op_options = op.BuiltinOptions()
        pool_options = Pool2DOptions()
        pool_options.Init(op_options.Bytes, op_options.Pos)
        
        kernel_h = pool_options.FilterHeight()
        kernel_w = pool_options.FilterWidth()
        stride_h = pool_options.StrideH()
        stride_w = pool_options.StrideW()
        padding = pool_options.Padding()
        
        pool_size = [kernel_h, kernel_w]
        strides = [stride_h, stride_w]
        
        # Handle padding
        if padding == Padding.VALID:
            padding_val = [0, 0, 0, 0]
        elif padding == Padding.SAME:
            # Calculate SAME padding - simplified version
            padding_val = [0, 0, 0, 0]  # TODO: implement proper SAME padding calculation
        else:
            raise ValueError(f"Unsupported padding type: {padding}")
        
        if pool_type == "avg":
            return self.bb.normalize(
                relax.op.nn.avg_pool2d(
                    data_expr,
                    pool_size=pool_size,
                    strides=strides,
                    padding=padding_val,
                    layout="NHWC"
                )
            )
        elif pool_type == "max":
            return self.bb.normalize(
                relax.op.nn.max_pool2d(
                    data_expr,
                    pool_size=pool_size,
                    strides=strides,
                    padding=padding_val,
                    layout="NHWC"
                )
            )
        else:
            raise ValueError(f"Unsupported pool type: {pool_type}")

    # Convolution operators
    def convert_conv2d(self, subgraph, op):
        """Convert TFLite CONV_2D operator."""
        return self._convert_conv(subgraph, op, "conv2d")

    def convert_depthwise_conv2d(self, subgraph, op):
        """Convert TFLite DEPTHWISE_CONV_2D operator."""
        return self._convert_conv(subgraph, op, "depthwise")

    def _convert_conv(self, subgraph, op, conv_type):
        """Generic convolution conversion."""
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.Conv2DOptions import Conv2DOptions
            from tflite.DepthwiseConv2DOptions import DepthwiseConv2DOptions
            from tflite.Padding import Padding
        except ImportError:
            raise ImportError("The tflite package must be installed")
            
        self.current_subgraph = subgraph
        input_tensors = self._get_input_tensors(subgraph, op)
        assert len(input_tensors) >= 2
        
        data_expr = self._get_tensor_expr(input_tensors[0])
        weight_expr = self._get_tensor_expr(input_tensors[1])
        
        # Get convolution options
        if conv_type == "conv2d":
            assert op.BuiltinOptionsType() == BuiltinOptions.Conv2DOptions
            op_options = op.BuiltinOptions()
            conv_options = Conv2DOptions()
            conv_options.Init(op_options.Bytes, op_options.Pos)
        else:  # depthwise
            assert op.BuiltinOptionsType() == BuiltinOptions.DepthwiseConv2DOptions
            op_options = op.BuiltinOptions()
            conv_options = DepthwiseConv2DOptions()
            conv_options.Init(op_options.Bytes, op_options.Pos)
        
        stride_h = conv_options.StrideH()
        stride_w = conv_options.StrideW()
        dilation_h = conv_options.DilationHFactor()
        dilation_w = conv_options.DilationWFactor()
        padding = conv_options.Padding()
        
        strides = [stride_h, stride_w]
        dilation = [dilation_h, dilation_w]
        
        # Handle padding
        if padding == Padding.VALID:
            padding_val = [0, 0, 0, 0]
        elif padding == Padding.SAME:
            # Calculate SAME padding - simplified version
            padding_val = [0, 0, 0, 0]  # TODO: implement proper SAME padding calculation
        else:
            raise ValueError(f"Unsupported padding type: {padding}")
        
        # Convert weight layout from TFLite to Relax format
        # TFLite: OHWI or 1HWO (depthwise) -> Relax: OIHW or OIHW
        if conv_type == "conv2d":
            # TFLite OHWI -> Relax OIHW  
            weight_expr = self.bb.normalize(relax.op.permute_dims(weight_expr, [0, 3, 1, 2]))
        else:
            # TFLite 1HWO -> Relax OIHW (need to reshape and permute)
            # This is simplified - actual depthwise handling is more complex
            pass
        
        if conv_type == "conv2d":
            result = relax.op.nn.conv2d(
                data_expr,
                weight_expr,
                strides=strides,
                padding=padding_val,
                dilation=dilation,
                data_layout="NHWC",
                kernel_layout="OIHW"
            )
        else:
            # For depthwise conv, we need different handling
            result = relax.op.nn.conv2d(
                data_expr,
                weight_expr,
                strides=strides,
                padding=padding_val,
                dilation=dilation,
                data_layout="NHWC",
                kernel_layout="OIHW",
                groups=data_expr.struct_info.shape[-1] # This needs proper group calculation for depthwise
            )
        
        result = self.bb.normalize(result)
        
        # Add bias if present
        if len(input_tensors) == 3:
            bias_expr = self._get_tensor_expr(input_tensors[2])
            result = self.bb.normalize(relax.op.nn.bias_add(result, bias_expr, axis=3))
        
        return result

    def convert_fully_connected(self, subgraph, op):
        """Convert TFLite FULLY_CONNECTED operator."""
        self.current_subgraph = subgraph
        input_tensors = self._get_input_tensors(subgraph, op)
        assert len(input_tensors) >= 2
        
        data_expr = self._get_tensor_expr(input_tensors[0])
        weight_expr = self._get_tensor_expr(input_tensors[1])
        
        # Flatten input data to 2D for matrix multiplication if necessary
        if len(data_expr.struct_info.shape) > 2:
            data_expr = self.bb.normalize(relax.op.reshape(data_expr, [-1, data_expr.struct_info.shape[-1]]))
        
        # TFLite weight layout is [out_features, in_features], which is W.
        # We need to compute data @ W.T. So we transpose W.
        weight_expr = self.bb.normalize(relax.op.permute_dims(weight_expr, [1, 0]))
        
        # The `dense` operator is not in relax, we use matmul.
        result = self.bb.normalize(relax.op.matmul(data_expr, weight_expr))
        
        # Add bias if present
        if len(input_tensors) > 2 and self._has_tensor_value(input_tensors[2]):
            bias_expr = self._get_tensor_expr(input_tensors[2])
            result = self.bb.normalize(relax.op.nn.bias_add(result, bias_expr))
        
        return result

    def convert_dequantize(self, subgraph, op):
        """Convert TFLite DEQUANTIZE operator."""
        self.current_subgraph = subgraph
        input_tensors = self._get_input_tensors(subgraph, op)
        assert len(input_tensors) == 1
        
        input_expr = self._get_tensor_expr(input_tensors[0])
        
        # For now, just cast to float32 (simplified dequantization)
        return self.bb.normalize(relax.op.astype(input_expr, "float32"))


def _decode_type(n):
    """Decode TFLite tensor type to string."""
    _tflite_m = {
        0: "float32",
        1: "float16", 
        2: "int32",
        3: "uint8",
        4: "int64",
        5: "string",
        6: "bool",
        7: "int16",
        8: "complex64",
        9: "int8",
    }
    return _tflite_m.get(n, "float32")


def _input_type(model):
    """Extract input shapes and types from TFLite model."""
    subgraph_count = model.SubgraphsLength()
    assert subgraph_count > 0
    shape_dict = {}
    dtype_dict = {}
    
    for subgraph_index in range(subgraph_count):
        subgraph = model.Subgraphs(subgraph_index)
        inputs_count = subgraph.InputsLength()
        
        for input_index in range(inputs_count):
            input_idx = subgraph.Inputs(input_index)
            tensor = subgraph.Tensors(input_idx)
            input_shape = tuple(tensor.ShapeAsNumpy()) if tensor.ShapeLength() > 0 else ()
            tensor_type = tensor.Type()
            input_name = get_tensor_name(subgraph, input_idx)
            
            shape_dict[input_name] = input_shape
            dtype_dict[input_name] = _decode_type(tensor_type)
    
    return shape_dict, dtype_dict


def from_tflite(
    model,
    shape_dict: Optional[Dict[str, List]] = None,
    dtype_dict: Optional[Union[str, Dict[str, str]]] = "float32",
) -> Tuple[IRModule, Dict[str, tvm.nd.NDArray]]:
    """Convert from TFLite model into compatible Relax IRModule.

    Parameters
    ----------
    model : tflite.Model
        The TFLite model to convert.

    shape_dict : dict of str to list/tuple, optional
        Input shapes of the model.

    dtype_dict : str or dict of str to str, optional
        Input types of the model.

    Returns
    -------
    mod : tvm.IRModule
        The Relax module for compilation.
    params : Dict[str, tvm.nd.NDArray]
        The parameters of the model.
    """
    try:
        # The tflite.Model.Model is the actual class for a model object
        # loaded from a buffer.
        from tflite.Model import Model as TFLiteModelClass
    except ImportError:
        raise ImportError("Could not import TFLite Model class. Is the tflite package installed correctly?")

    if not isinstance(model, TFLiteModelClass):
        raise TypeError(
            "The provided model is not a valid tflite.Model.Model object. "
            f"Expected type {TFLiteModelClass}, but got {type(model)}."
        )

    _shape_dict, _dtype_dict = _input_type(model)
    if shape_dict is not None:
        _shape_dict.update(shape_dict)
    if dtype_dict is not None:
        if isinstance(dtype_dict, str):
            _dtype_dict = {k: dtype_dict for k in _dtype_dict.keys()}
        else:
            _dtype_dict.update(dtype_dict)

    # Only support single subgraph for now
    assert model.SubgraphsLength() == 1, "Only single subgraph models are supported"

    # Create importer and convert
    importer = TFLiteGraphImporter(shape_dict=_shape_dict, dtype_dict=_dtype_dict)
    mod, params = importer.from_tflite(model)
    return mod, params


