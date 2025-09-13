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
import os
import logging


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
        return expr.value.value if hasattr(expr.value, "value") else expr.value
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

def debug_ir_variables(mod):
    """Debug function to print all variables in the IR module."""
    print("\n=== Debugging IR Variables ===")
    
    try:
        main_func = mod["main"]
        print(f"Function parameters ({len(main_func.params)}):")
        for i, param in enumerate(main_func.params):
            print(f"  {i}: {param.name_hint} (struct_info: {param.struct_info})")
        
        # Try to find all variable references in the function body
        print(f"\nAnalyzing function body...")
        
        # Print the IR for inspection
        print(f"\nFull IR for inspection:")
        print(main_func.script(show_meta=True))
        
    except Exception as e:
        print(f"Error during IR analysis: {e}")

def sanitize_tensor_name(name):
    """Sanitize tensor names to be valid TVM variable names."""
    # Replace problematic characters with underscores
    sanitized = name.replace(".", "_").replace("/", "_").replace(":", "_")
    # Remove any double underscores
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    # Ensure it doesn't start with a number
    if sanitized and sanitized[0].isdigit():
        sanitized = "tensor_" + sanitized
    return sanitized

def get_tensor_name(subgraph, tensor_idx):
    """Get the tensor name - FIXED for consistent naming."""
    tensor_name = subgraph.Tensors(tensor_idx).Name()
    if tensor_name is not None:
        tensor_name = tensor_name.decode("utf-8")
    else:
        tensor_name = "tvmgen_tensor_" + str(tensor_idx)
    
    # CRITICAL: Apply consistent name sanitization
    return sanitize_tensor_name(tensor_name)

def to_int_list(arr):
    """Convert numpy array or list to python int list"""
    if isinstance(arr, np.ndarray):
        return arr.astype(int).tolist()
    elif isinstance(arr, (list, tuple)):
        return [int(x) for x in arr]
    else:
        return [int(arr)]


def _extract_int_value(value):
    """Extract integer value from TVM IntImm or regular int."""
    if hasattr(value, 'value'):
        return int(value.value)
    elif isinstance(value, tir.IntImm):
        return int(value)
    else:
        return int(value)


class TFLiteGraphImporter:
    """A helper class for handling Relax expression conversion from TFLite model."""

    def __init__(
        self,
        shape_dict: Optional[Dict[str, List]] = None,
        dtype_dict: Optional[Union[str, Dict[str, str]]] = "float32",
        keep_params_in_input: bool = True,
    ):
        self._nodes: Dict[str, relax.Expr] = {}
        self._inputs: Dict[str, relax.Var] = {}
        self._num_input: int = 0
        self._shape = shape_dict.copy() if shape_dict else {}
        self._dtype = dtype_dict
        self._name_supply = NameSupply()
        self.bb: relax.BlockBuilder = relax.BlockBuilder()
        self._params = {}  # Store (var, value) tuples like ONNX
        self._keep_params_in_input = keep_params_in_input
        self._prefetched_nodes = {}
        
        # Setup debug logging
        self.debug_enabled = os.environ.get('TFLITE_DEBUG', '0') == '1'
        if self.debug_enabled:
            logging.basicConfig(level=logging.DEBUG, 
                              format='[TFLITE_DEBUG] %(message)s')
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = None

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

    def _debug_log(self, message):
        """Log debug message if debugging is enabled."""
        if self.debug_enabled and self.logger:
            self.logger.debug(message)

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
            "PACK": self.convert_pack,
            "RELU": self.convert_relu,
            "RELU6": self.convert_relu6,
            "RESHAPE": self.convert_reshape,
            "SHAPE": self.convert_shape,
            "SOFTMAX": self.convert_softmax,
            "SQUEEZE": self.convert_squeeze,
            "STRIDED_SLICE": self.convert_strided_slice,
            "SUB": self.convert_sub,
            "TANH": self.convert_tanh,
            "TRANSPOSE": self.convert_transpose,
        }

    
    def convert_resize_bilinear(self, subgraph, op):
        """Convert TFLite RESIZE_BILINEAR operator."""
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.ResizeBilinearOptions import ResizeBilinearOptions
        except ImportError:
            raise ImportError("The tflite package must be installed")

        self._debug_log("Converting RESIZE_BILINEAR")
        self.current_subgraph = subgraph
        input_tensors = self._get_input_tensors(subgraph, op)
        assert len(input_tensors) == 2, "RESIZE_BILINEAR expects 2 inputs (data, size)"

        data_expr = self._get_tensor_expr(input_tensors[0])
        size_tensor = input_tensors[1]
        
        input_shape = data_expr.struct_info.shape
        self._debug_log(f"Input shape: {input_shape}")

        # Get the new size from the second input tensor
        if not self._has_tensor_value(size_tensor):
            raise NotImplementedError(
                "Dynamic resize size is not supported. Size tensor must be constant."
            )
        
        new_size = self._get_tensor_value(size_tensor)
        if len(new_size) != 2:
            raise ValueError(f"Expected size tensor with 2 elements, got {len(new_size)}")
        
        new_height, new_width = int(new_size[0]), int(new_size[1])
        self._debug_log(f"Target size: H={new_height}, W={new_width}")

        # Get resize options
        align_corners = False
        half_pixel_centers = False
        
        if op.BuiltinOptionsType() == BuiltinOptions.ResizeBilinearOptions:
            op_options = op.BuiltinOptions()
            resize_options = ResizeBilinearOptions()
            resize_options.Init(op_options.Bytes, op_options.Pos)
            align_corners = bool(resize_options.AlignCorners())
            half_pixel_centers = bool(resize_options.HalfPixelCenters())
        
        self._debug_log(f"Resize options - align_corners: {align_corners}, half_pixel_centers: {half_pixel_centers}")

        # Convert to Relax resize operation
        try:
            # Relax resize expects (new_height, new_width) as size
            size = [new_height, new_width]
            
            # Map TFLite options to Relax coordinate transformation mode
            if half_pixel_centers:
                coordinate_transformation_mode = "half_pixel"
            elif align_corners:
                coordinate_transformation_mode = "align_corners"
            else:
                coordinate_transformation_mode = "asymmetric"
            
            self._debug_log(f"Using coordinate_transformation_mode: {coordinate_transformation_mode}")
            
            # Use image.resize for bilinear interpolation
            result = relax.op.image.resize2d(
                data_expr,
                size=size,
                layout="NHWC",
                method="linear",  # bilinear interpolation
                coordinate_transformation_mode=coordinate_transformation_mode,
            )
            
            self._debug_log("Created resize_bilinear operation successfully")
            result = self.bb.normalize(result)
            self._debug_log(f"Normalized resize operation, output shape: {result.struct_info.shape}")
            
            return result
            
        except Exception as e:
            self._debug_log(f"ERROR in resize_bilinear operation: {e}")
            # Fallback: try using the generic resize if image.resize2d is not available
            try:
                self._debug_log("Trying fallback resize method...")
                
                # Alternative approach using nn.upsampling
                # Calculate scale factors
                if len(input_shape) == 4:  # NHWC
                    input_h = input_shape[1]
                    input_w = input_shape[2]
                    
                    if isinstance(input_h, (tvm.tir.IntImm, int)) and isinstance(input_w, (tvm.tir.IntImm, int)):
                        input_h_val = int(input_h) if hasattr(input_h, 'value') else int(input_h)
                        input_w_val = int(input_w) if hasattr(input_w, 'value') else int(input_w)
                        
                        scale_h = new_height / input_h_val
                        scale_w = new_width / input_w_val
                        
                        self._debug_log(f"Calculated scales: H={scale_h}, W={scale_w}")
                        
                        # Use upsampling with computed scales
                        result = relax.op.nn.upsampling(
                            data_expr,
                            scale_h=scale_h,
                            scale_w=scale_w,
                            layout="NHWC",
                            method="linear"
                        )
                        
                        result = self.bb.normalize(result)
                        self._debug_log("Fallback upsampling successful")
                        return result
                    else:
                        raise ValueError("Cannot compute scales for symbolic input dimensions")
                else:
                    raise ValueError(f"Expected 4D input, got {len(input_shape)}D")
                    
            except Exception as fallback_e:
                self._debug_log(f"Fallback method also failed: {fallback_e}")
                raise RuntimeError(f"Both resize methods failed. Primary: {e}, Fallback: {fallback_e}")

    def convert_resize_nearest_neighbor(self, subgraph, op):
        """Convert TFLite RESIZE_NEAREST_NEIGHBOR operator."""
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.ResizeNearestNeighborOptions import ResizeNearestNeighborOptions
        except ImportError:
            raise ImportError("The tflite package must be installed")

        self._debug_log("Converting RESIZE_NEAREST_NEIGHBOR")
        self.current_subgraph = subgraph
        input_tensors = self._get_input_tensors(subgraph, op)
        assert len(input_tensors) == 2, "RESIZE_NEAREST_NEIGHBOR expects 2 inputs (data, size)"

        data_expr = self._get_tensor_expr(input_tensors[0])
        size_tensor = input_tensors[1]
        
        input_shape = data_expr.struct_info.shape
        self._debug_log(f"Input shape: {input_shape}")

        # Get the new size from the second input tensor
        if not self._has_tensor_value(size_tensor):
            raise NotImplementedError(
                "Dynamic resize size is not supported. Size tensor must be constant."
            )
        
        new_size = self._get_tensor_value(size_tensor)
        if len(new_size) != 2:
            raise ValueError(f"Expected size tensor with 2 elements, got {len(new_size)}")
        
        new_height, new_width = int(new_size[0]), int(new_size[1])
        self._debug_log(f"Target size: H={new_height}, W={new_width}")

        # Get resize options
        align_corners = False
        half_pixel_centers = False
        
        if op.BuiltinOptionsType() == BuiltinOptions.ResizeNearestNeighborOptions:
            op_options = op.BuiltinOptions()
            resize_options = ResizeNearestNeighborOptions()
            resize_options.Init(op_options.Bytes, op_options.Pos)
            align_corners = bool(resize_options.AlignCorners())
            half_pixel_centers = bool(resize_options.HalfPixelCenters())
        
        self._debug_log(f"Resize options - align_corners: {align_corners}, half_pixel_centers: {half_pixel_centers}")

        try:
            # Use image.resize for nearest neighbor interpolation
            size = [new_height, new_width]
            
            # Map TFLite options to Relax coordinate transformation mode
            if half_pixel_centers:
                coordinate_transformation_mode = "half_pixel"
            elif align_corners:
                coordinate_transformation_mode = "align_corners"
            else:
                coordinate_transformation_mode = "asymmetric"
            
            result = relax.op.image.resize2d(
                data_expr,
                size=size,
                layout="NHWC",
                method="nearest_neighbor",
                coordinate_transformation_mode=coordinate_transformation_mode,
            )
            
            self._debug_log("Created resize_nearest_neighbor operation successfully")
            result = self.bb.normalize(result)
            self._debug_log(f"Normalized resize operation, output shape: {result.struct_info.shape}")
            
            return result
            
        except Exception as e:
            self._debug_log(f"ERROR in resize_nearest_neighbor operation: {e}")
            raise

    
    def _validate_pool_params(self, data_shape, kernel_size, strides, padding):
        """Validate pool parameters before creating the operation."""
        self._debug_log("Validating pool parameters...")
        
        if len(data_shape) != 4:
            raise ValueError(f"Expected 4D input, got {len(data_shape)}D: {data_shape}")
        
        batch, height, width, channels = data_shape
        kernel_h, kernel_w = kernel_size
        stride_h, stride_w = strides
        pad_top, pad_left, pad_bottom, pad_right = padding
        
        # Convert symbolic dimensions to concrete values for validation
        try:
            h_val = int(height) if hasattr(height, 'value') else int(height)
            w_val = int(width) if hasattr(width, 'value') else int(width)
        except (TypeError, AttributeError):
            self._debug_log("Symbolic dimensions detected, skipping strict validation")
            return
        
        # Check for invalid parameters
        if kernel_h <= 0 or kernel_w <= 0:
            raise ValueError(f"Invalid kernel size: [{kernel_h}, {kernel_w}]")
        
        if stride_h <= 0 or stride_w <= 0:
            raise ValueError(f"Invalid stride: [{stride_h}, {stride_w}]")
        
        if any(p < 0 for p in padding):
            raise ValueError(f"Negative padding not allowed: {padding}")
        
        # Check if kernel is larger than input + padding
        effective_h = h_val + pad_top + pad_bottom
        effective_w = w_val + pad_left + pad_right
        
        if kernel_h > effective_h:
            raise ValueError(f"Kernel height {kernel_h} > effective input height {effective_h}")
        
        if kernel_w > effective_w:
            raise ValueError(f"Kernel width {kernel_w} > effective input width {effective_w}")
        
        # Calculate expected output shape
        out_h = (effective_h - kernel_h) // stride_h + 1
        out_w = (effective_w - kernel_w) // stride_w + 1
        
        self._debug_log(f"Validation passed - expected output: [{batch}, {out_h}, {out_w}, {channels}]")

    def _safe_int_extract(self, value, name="value"):
        """Safely extract integer from TVM expressions."""
        try:
            if isinstance(value, int):
                return value
            elif hasattr(value, 'value'):
                return int(value.value)
            elif isinstance(value, tvm.tir.IntImm):
                return int(value)
            else:
                self._debug_log(f"Could not extract {name}: {type(value)}, returning default")
                return 224  # Safe default
        except Exception as e:
            self._debug_log(f"Error extracting {name}: {e}, returning default")
            return 224

    def _convert_pool2d_safe(self, subgraph, op, pool_type):
        """Safe pool2d conversion with extensive validation."""
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.Padding import Padding
            from tflite.Pool2DOptions import Pool2DOptions
        except ImportError:
            raise ImportError("The tflite package must be installed")

        self._debug_log(f"=== SAFE Converting {pool_type}_pool2d ===")
        self.current_subgraph = subgraph
        
        try:
            input_tensors = self._get_input_tensors(subgraph, op)
            assert len(input_tensors) == 1
            self._debug_log("✓ Got input tensors")

            data_expr = self._get_tensor_expr(input_tensors[0])
            input_shape = data_expr.struct_info.shape
            self._debug_log(f"✓ Input shape: {input_shape}")

            # Extract pool options
            assert op.BuiltinOptionsType() == BuiltinOptions.Pool2DOptions
            op_options = op.BuiltinOptions()
            pool_options = Pool2DOptions()
            pool_options.Init(op_options.Bytes, op_options.Pos)

            kernel_h = pool_options.FilterHeight()
            kernel_w = pool_options.FilterWidth()
            stride_h = pool_options.StrideH()
            stride_w = pool_options.StrideW()
            padding = pool_options.Padding()
            
            self._debug_log(f"✓ Raw params - kernel: [{kernel_h}, {kernel_w}], stride: [{stride_h}, {stride_w}]")

            # Safe dimension extraction
            if len(input_shape) != 4:
                raise ValueError(f"Expected 4D input for pool2d, got {len(input_shape)}D")
            
            input_h = self._safe_int_extract(input_shape[1], "input_h")
            input_w = self._safe_int_extract(input_shape[2], "input_w")
            
            self._debug_log(f"✓ Extracted dimensions - H: {input_h}, W: {input_w}")

            # Calculate padding
            if padding == Padding.VALID:
                padding_val = [0, 0, 0, 0]
                self._debug_log("✓ Using VALID padding")
            elif padding == Padding.SAME:
                # Safe SAME padding calculation
                def safe_same_padding(input_size, kernel_size, stride):
                    output_size = (input_size + stride - 1) // stride
                    total_pad = max(0, (output_size - 1) * stride + kernel_size - input_size)
                    pad_before = total_pad // 2
                    pad_after = total_pad - pad_before
                    return pad_before, pad_after

                pad_top, pad_bottom = safe_same_padding(input_h, kernel_h, stride_h)
                pad_left, pad_right = safe_same_padding(input_w, kernel_w, stride_w)
                padding_val = [pad_top, pad_left, pad_bottom, pad_right]
                self._debug_log(f"✓ SAME padding: {padding_val}")
            else:
                raise ValueError(f"Unsupported padding type: {padding}")

            # Validate all parameters
            self._validate_pool_params(input_shape, [kernel_h, kernel_w], [stride_h, stride_w], padding_val)

            # Create operation with error handling
            pool_size = [kernel_h, kernel_w]
            strides = [stride_h, stride_w]
            
            self._debug_log(f"✓ Creating {pool_type}_pool2d with validated params")
            
            if pool_type == "avg":
                op_func = relax.op.nn.avg_pool2d
            elif pool_type == "max":
                op_func = relax.op.nn.max_pool2d
            else:
                raise ValueError(f"Unsupported pool type: {pool_type}")

            try:
                result = op_func(
                    data_expr,
                    pool_size=pool_size,
                    strides=strides,
                    padding=padding_val,
                    layout="NHWC",
                )
                self._debug_log(f"✓ Created {pool_type}_pool2d operation")
                
                result = self.bb.normalize(result)
                self._debug_log(f"✓ Normalized {pool_type}_pool2d operation")
                
                return result
                
            except Exception as op_error:
                self._debug_log(f"✗ Operation creation failed: {op_error}")
                self._debug_log(f"  Final params: pool_size={pool_size}, strides={strides}, padding={padding_val}")
                raise
                
        except Exception as e:
            self._debug_log(f"✗ FATAL ERROR in {pool_type}_pool2d: {e}")
            self._debug_log(f"  Exception type: {type(e)}")
            import traceback
            self._debug_log(f"  Traceback: {traceback.format_exc()}")
            raise

    def from_tflite(self, model) -> Tuple[IRModule, Dict[str, np.ndarray]]:
        """Construct Relax expressions from the TFLite model - Fixed to ensure parameters exist."""
        self._debug_log("Starting TFLite to Relax conversion")
        
        # Store model reference for use in other methods
        self.current_model = model

        # CRITICAL: Parse inputs BEFORE creating function to ensure we have input parameters
        self._parse_model_inputs(model)
        self._debug_log(f"Parsed {len(self._inputs)} model inputs")
        
        # Ensure we have at least one input parameter
        input_list = [value for value in self._inputs.values() if isinstance(value, relax.Var)]
        if not input_list:
            raise RuntimeError("No input parameters found - TVM Relax functions must have parameters")
        
        self._debug_log(f"Input list has {len(input_list)} parameters: {[v.name_hint for v in input_list]}")

        # Create function with input parameters
        with self.bb.function("main", input_list):  # Pass inputs to function signature
            with self.bb.dataflow():
                self._debug_log("Starting dataflow block")
                
                self._convert_operators(model)
                self._debug_log(f"Converted operators")

                # Get outputs
                subgraph = model.Subgraphs(0)
                model_outputs = subgraph.OutputsAsNumpy()
                self._debug_log(f"Model has {len(model_outputs)} outputs")
                
                # old version
                #outputs = [self._nodes[get_tensor_name(subgraph, i)] for i in model_outputs]
                #outputs = outputs[0] if len(outputs) == 1 else relax.Tuple(outputs)

                #output_var = self.bb.emit_output(outputs)
                #self._debug_log("Emitted output variable")
                outputs = [self._nodes[get_tensor_name(subgraph, i)] for i in model_outputs]
    
                # CRITICAL FIX: Make ALL outputs contiguous (even passthrough inputs)
                def ensure_contiguous(tensor_expr):
                    """Force contiguous layout using flatten-reshape."""
                    original_shape = tensor_expr.struct_info.shape
                    flattened = self.bb.normalize(relax.op.reshape(tensor_expr, [-1]))
                    contiguous = self.bb.normalize(relax.op.reshape(flattened, original_shape))
                    return contiguous
                
                contiguous_outputs = []
                for i, output in enumerate(outputs):
                    self._debug_log(f"Making output {i} contiguous")
                    contiguous_output = ensure_contiguous(output)
                    contiguous_outputs.append(contiguous_output)
                
                final_outputs = contiguous_outputs[0] if len(contiguous_outputs) == 1 else relax.Tuple(contiguous_outputs)
                output_var = self.bb.emit_output(final_outputs)

            # emit_func_output without additional params since they're already in function signature
            self.bb.emit_func_output(output_var)
            self._debug_log("Emitted function output")

        # Get the module
        self._debug_log("Building module")
        relax_mod = self.bb.get()
        
        # Add function attributes
        func_attrs = {"num_input": len(input_list)}
        try:
            main_func = relax_mod["main"]
            relax_mod["main"] = main_func.with_attrs(func_attrs)
            self._debug_log("Attached function attributes")
        except Exception as attr_e:
            self._debug_log(f"Error attaching attributes: {attr_e}")
            raise       
        
        self._debug_log("Conversion complete")
        print(f"Function created with {len(input_list)} input parameters")
        return relax_mod, {}
    

    def long_from_tflite(self, model) -> Tuple[IRModule, Dict[str, np.ndarray]]:
        """Construct Relax expressions from the TFLite model - Corrected TVM pattern."""
        self._debug_log("Starting TFLite to Relax conversion")
        
        # Store model reference for use in other methods
        self.current_model = model

        # First, prepare all inputs and parameters outside the function scope
        self._parse_model_inputs(model)
        self._debug_log(f"Parsed {len(self._inputs)} model inputs")
        
        # Create function signature with all variables
        func_attrs = {"num_input": self._num_input}
        input_list = [value for value in self._inputs.values() if isinstance(value, relax.Var)]
        
        # Handle parameters following ONNX pattern  
        if self._keep_params_in_input and self._params:
            param_var_list, param_value_list = map(list, zip(*self._params.values()))
            
            # Ensure all arrays are writable
            writable_param_list = []
            for param_array in param_value_list:
                if hasattr(param_array, 'flags') and not param_array.flags.writeable:
                    writable_array = np.array(param_array, copy=True)
                    writable_param_list.append(writable_array)
                else:
                    writable_param_list.append(param_array)
            
            input_list = input_list + param_var_list
            func_attrs["params"] = writable_param_list
            self._debug_log(f"Added {len(param_var_list)} parameters to function signature")

        # CRITICAL: Pass parameters to function signature, not to emit_func_output
        with self.bb.function("main", input_list):  # Parameters go HERE
            with self.bb.dataflow():
                self._debug_log("Starting dataflow block")
                
                self._convert_operators(model)
                self._debug_log(f"Converted operators")

                # Get outputs
                subgraph = model.Subgraphs(0)
                model_outputs = subgraph.OutputsAsNumpy()
                self._debug_log(f"Model has {len(model_outputs)} outputs")
                
                outputs = [self._nodes[get_tensor_name(subgraph, i)] for i in model_outputs]
                outputs = outputs[0] if len(outputs) == 1 else relax.Tuple(outputs)

                output_var = self.bb.emit_output(outputs)
                self._debug_log("Emitted output variable")

            # CRITICAL: emit_func_output WITHOUT params parameter
            self.bb.emit_func_output(output_var)  # NO params here!
            self._debug_log("Emitted function output")

        # Get the module (after function scope ends)
        self._debug_log("Building module")
        relax_mod = self.bb.get()
        
        # Apply attributes to the main function
        try:
            main_func = relax_mod["main"]
            relax_mod["main"] = main_func.with_attrs(func_attrs)
            self._debug_log("Attached function attributes")
        except Exception as attr_e:
            self._debug_log(f"Error attaching attributes: {attr_e}")
            raise       
        
        # Return numpy arrays for backward compatibility
        np_params = {}
        for name, (var, numpy_array) in self._params.items():
            if hasattr(numpy_array, 'flags') and not numpy_array.flags.writeable:
                np_params[name] = np.array(numpy_array, copy=True)
            else:
                np_params[name] = numpy_array

        self._debug_log("Conversion complete")
        print(f"Final parameters dict has {len(np_params)} entries")
        # DEBUG: Print parameter names to check for consistency
        print("\n=== Parameter Name Check ===")
        for name, arr in np_params.items():
            print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}")
        
        # DEBUG: Analyze the IR before returning
        debug_ir_variables(relax_mod)
        return relax_mod, np_params

    def _convert_pool2d_safe(self, subgraph, op, pool_type):
        """Safe pool2d conversion with extensive validation."""
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.Padding import Padding
            from tflite.Pool2DOptions import Pool2DOptions
        except ImportError:
            raise ImportError("The tflite package must be installed")

        self._debug_log(f"=== SAFE Converting {pool_type}_pool2d ===")
        self.current_subgraph = subgraph
        
        try:
            input_tensors = self._get_input_tensors(subgraph, op)
            assert len(input_tensors) == 1
            self._debug_log("✓ Got input tensors")

            data_expr = self._get_tensor_expr(input_tensors[0])
            input_shape = data_expr.struct_info.shape
            self._debug_log(f"✓ Input shape: {input_shape}")

            # Extract pool options
            assert op.BuiltinOptionsType() == BuiltinOptions.Pool2DOptions
            op_options = op.BuiltinOptions()
            pool_options = Pool2DOptions()
            pool_options.Init(op_options.Bytes, op_options.Pos)

            kernel_h = pool_options.FilterHeight()
            kernel_w = pool_options.FilterWidth()
            stride_h = pool_options.StrideH()
            stride_w = pool_options.StrideW()
            padding = pool_options.Padding()
            
            self._debug_log(f"✓ Raw params - kernel: [{kernel_h}, {kernel_w}], stride: [{stride_h}, {stride_w}]")

            # Safe dimension extraction
            if len(input_shape) != 4:
                raise ValueError(f"Expected 4D input for pool2d, got {len(input_shape)}D")
            
            input_h = self._safe_int_extract(input_shape[1], "input_h")
            input_w = self._safe_int_extract(input_shape[2], "input_w")
            
            self._debug_log(f"✓ Extracted dimensions - H: {input_h}, W: {input_w}")

            # Calculate padding
            if padding == Padding.VALID:
                padding_val = [0, 0, 0, 0]
                self._debug_log("✓ Using VALID padding")
            elif padding == Padding.SAME:
                # Safe SAME padding calculation
                def safe_same_padding(input_size, kernel_size, stride):
                    output_size = (input_size + stride - 1) // stride
                    total_pad = max(0, (output_size - 1) * stride + kernel_size - input_size)
                    pad_before = total_pad // 2
                    pad_after = total_pad - pad_before
                    return pad_before, pad_after

                pad_top, pad_bottom = safe_same_padding(input_h, kernel_h, stride_h)
                pad_left, pad_right = safe_same_padding(input_w, kernel_w, stride_w)
                padding_val = [pad_top, pad_left, pad_bottom, pad_right]
                self._debug_log(f"✓ SAME padding: {padding_val}")
            else:
                raise ValueError(f"Unsupported padding type: {padding}")

            # Validate all parameters
            self._validate_pool_params(input_shape, [kernel_h, kernel_w], [stride_h, stride_w], padding_val)

            # Create operation with error handling
            pool_size = [kernel_h, kernel_w]
            strides = [stride_h, stride_w]
            
            self._debug_log(f"✓ Creating {pool_type}_pool2d with validated params")
            
            if pool_type == "avg":
                op_func = relax.op.nn.avg_pool2d
            elif pool_type == "max":
                op_func = relax.op.nn.max_pool2d
            else:
                raise ValueError(f"Unsupported pool type: {pool_type}")

            try:
                result = op_func(
                    data_expr,
                    pool_size=pool_size,
                    strides=strides,
                    padding=padding_val,
                    layout="NHWC",
                )
                self._debug_log(f"✓ Created {pool_type}_pool2d operation")
                
                result = self.bb.normalize(result)
                self._debug_log(f"✓ Normalized {pool_type}_pool2d operation")
                
                return result
                
            except Exception as op_error:
                self._debug_log(f"✗ Operation creation failed: {op_error}")
                self._debug_log(f"  Final params: pool_size={pool_size}, strides={strides}, padding={padding_val}")
                raise
                
        except Exception as e:
            self._debug_log(f"✗ FATAL ERROR in {pool_type}_pool2d: {e}")
            self._debug_log(f"  Exception type: {type(e)}")
            import traceback
            self._debug_log(f"  Traceback: {traceback.format_exc()}")
            raise

    # Usage: Replace your _convert_pool2d calls with _convert_pool2d_safe
    # self.convert_map = {
    #     "AVERAGE_POOL_2D": lambda s, o: self._convert_pool2d_safe(s, o, "avg"),
    #     "MAX_POOL_2D": lambda s, o: self._convert_pool2d_safe(s, o, "max"),
    #     ...
    # }

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
            "PACK": self.convert_pack,
            "RELU": self.convert_relu,
            "RELU6": self.convert_relu6,
            "RESHAPE": self.convert_reshape,            
            "RESIZE_BILINEAR": self.convert_resize_bilinear,  
            "RESIZE_NEAREST_NEIGHBOR": self.convert_resize_nearest_neighbor,   
            "SHAPE": self.convert_shape,
            "SOFTMAX": self.convert_softmax,
            "SQUEEZE": self.convert_squeeze,
            "STRIDED_SLICE": self.convert_strided_slice,
            "SUB": self.convert_sub,
            "TANH": self.convert_tanh,
            "TRANSPOSE": self.convert_transpose,
        }

    def _parse_model_inputs(self, model):
        """Parse model inputs and create Relax variables - Debug version."""
        subgraph = model.Subgraphs(0)
        model_inputs = subgraph.InputsAsNumpy()
        
        print(f"DEBUG: Found {len(model_inputs)} model inputs: {model_inputs}")

        for model_input in model_inputs:
            input_name = get_tensor_name(subgraph, model_input)
            tensor = subgraph.Tensors(model_input)
            
            print(f"DEBUG: Processing input {model_input}: {input_name}")

            # Get shape and dtype
            if input_name in self._shape:
                shape = self._shape[input_name]
            else:
                if tensor.ShapeLength() > 0:
                    raw_shape = tensor.ShapeAsNumpy()
                    shape = tuple(int(dim) for dim in raw_shape)
                else:
                    shape = ()

            if isinstance(self._dtype, dict) and input_name in self._dtype:
                dtype = self._dtype[input_name]
            elif isinstance(self._dtype, str):
                dtype = self._dtype
            else:
                dtype = self._get_tensor_type_str(tensor.Type())

            # Create Relax variable
            input_var = relax.Var(
                name_hint=input_name,
                struct_info=relax.TensorStructInfo(shape=shape, dtype=dtype),
            )

            self._nodes[input_name] = input_var
            self._inputs[input_name] = input_var
            self._num_input += 1
            
            print(f"DEBUG: Created input var {input_name}: shape={shape}, dtype={dtype}")

        print(f"DEBUG: Total inputs created: {len(self._inputs)}")
        print(f"DEBUG: Input names: {list(self._inputs.keys())}")

    def old_parse_model_inputs(self, model):
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
                if tensor.ShapeLength() > 0:
                    # Simple conversion: numpy int32 -> Python int
                    raw_shape = tensor.ShapeAsNumpy()
                    shape = tuple(int(dim) for dim in raw_shape)
                else:
                    shape = ()

            if isinstance(self._dtype, dict) and input_name in self._dtype:
                dtype = self._dtype[input_name]
            elif isinstance(self._dtype, str):
                dtype = self._dtype
            else:
                dtype = self._get_tensor_type_str(tensor.Type())

            # Dont turn shape into list, keep it as tuple
            # shape=list(shape)
            # Create Relax variable
            input_var = relax.Var(
                name_hint=input_name,
                struct_info=relax.TensorStructInfo(shape=shape, dtype=dtype),
            )

            self._nodes[input_name] = input_var
            print("VAR", input_name, shape, dtype)
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
            buffer = (
                self.current_model.Buffers(buffer_idx)
                if buffer_idx < self.current_model.BuffersLength()
                else None
            )

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
                            "zero_point": relax.const(zero_point, "int32"),
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
        """Get tensor buffer value from tensor wrapper - Fixed for readonly arrays."""
        if not self._has_tensor_value(tensor_wrapper):
            return None

        dtype = self._get_numpy_dtype(tensor_wrapper.tensor.Type())
        data = tensor_wrapper.buffer.DataAsNumpy()

        if tensor_wrapper.tensor.ShapeLength() != 0:
            shape = to_int_list(tensor_wrapper.tensor.ShapeAsNumpy())
        else:
            shape = []

        # Create array from buffer and make a writable copy to avoid DLPack readonly issues
        array = np.frombuffer(data, dtype=dtype).reshape(shape)
        
        # CRITICAL FIX: Make a writable copy to avoid DLPack readonly error
        writable_array = np.array(array, copy=True)

        array = np.frombuffer(data, dtype=dtype).reshape(shape)
    
        # DEBUG: Check if buffer data is actually zeros
        print(f"First 10 values: {array.flatten()[:10]}")
        
        writable_array = np.array(array, copy=True)
        return writable_array
        

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
        """Get Relax expression for a tensor - Using constants for weights."""
        tensor_name = get_tensor_name(self.current_subgraph, tensor_wrapper.tensor_idx)
        self._debug_log(f"Getting tensor expression for: {tensor_name} (idx: {tensor_wrapper.tensor_idx})")

        if tensor_name in self._nodes:
            self._debug_log(f"Found existing node for: {tensor_name}")
            return self._nodes[tensor_name]
        else:
            # Check if this tensor has constant data (weights/biases)
            if not self._has_tensor_value(tensor_wrapper):
                raise ValueError(f"Tensor '{tensor_name}' is not an input and has no constant value.")

            # Get the tensor value as numpy array
            value = self._get_tensor_value(tensor_wrapper)
            dtype = self._get_tensor_type_str(tensor_wrapper.tensor.Type())
            
            self._debug_log(f"Creating CONSTANT: {tensor_name}, shape: {value.shape}, dtype: {dtype}")

            # Use constant directly - this avoids variable reference issues
            const_expr = relax.const(value, dtype)
            self._nodes[tensor_name] = const_expr
            self._debug_log(f"Added constant {tensor_name} to nodes")
            return const_expr
        
    def complex_get_tensor_expr(self, tensor_wrapper):
        """Get Relax expression for a tensor - FIXED for consistent naming."""
        tensor_name = get_tensor_name(self.current_subgraph, tensor_wrapper.tensor_idx)
        self._debug_log(f"Getting tensor expression for: {tensor_name} (idx: {tensor_wrapper.tensor_idx})")

        if tensor_name in self._nodes:
            self._debug_log(f"Found existing node for: {tensor_name}")
            return self._nodes[tensor_name]
        else:
            # Check if this tensor has constant data (weights/biases)
            if not self._has_tensor_value(tensor_wrapper):
                raise ValueError(f"Tensor '{tensor_name}' is not an input and has no constant value.")

            # Get the tensor value as numpy array
            value = self._get_tensor_value(tensor_wrapper)
            dtype = self._get_tensor_type_str(tensor_wrapper.tensor.Type())
            
            self._debug_log(f"Creating parameter: {tensor_name}, shape: {value.shape}, dtype: {dtype}")

            # Follow ONNX pattern for parameter handling
            if self._keep_params_in_input:
                # CRITICAL: Use the sanitized name consistently
                param_var = relax.Var(tensor_name, relax.TensorStructInfo(value.shape, dtype))
                self._nodes[tensor_name] = param_var
                # Store as (var, numpy_array) tuple like ONNX does
                self._params[tensor_name] = (param_var, value)  # value is already numpy array
                self._debug_log(f"Added parameter {tensor_name} to params dict")
                return param_var
            else:
                # Use constant directly
                const_expr = relax.const(value)
                self._nodes[tensor_name] = const_expr
                return const_expr

        
    def old_get_tensor_expr(self, tensor_wrapper):
        """Get Relax expression for a tensor - Fixed to follow ONNX pattern."""
        tensor_name = get_tensor_name(self.current_subgraph, tensor_wrapper.tensor_idx)
        self._debug_log(f"Getting tensor expression for: {tensor_name} (idx: {tensor_wrapper.tensor_idx})")

        if tensor_name in self._nodes:
            self._debug_log(f"Found existing node for: {tensor_name}")
            return self._nodes[tensor_name]
        else:
            # Check if this tensor has constant data (weights/biases)
            if not self._has_tensor_value(tensor_wrapper):
                raise ValueError(f"Tensor '{tensor_name}' is not an input and has no constant value.")

            # Get the tensor value as numpy array
            value = self._get_tensor_value(tensor_wrapper)
            dtype = self._get_tensor_type_str(tensor_wrapper.tensor.Type())
            
            self._debug_log(f"Creating parameter: {tensor_name}, shape: {value.shape}, dtype: {dtype}")

            # Follow ONNX pattern for parameter handling
            if self._keep_params_in_input:
                # Create variable for the parameter
                param_var = relax.Var(tensor_name, relax.TensorStructInfo(value.shape, dtype))
                self._nodes[tensor_name] = param_var
                # Store as (var, numpy_array) tuple like ONNX does
                self._params[tensor_name] = (param_var, value)  # value is already numpy array
                self._debug_log(f"Added parameter {tensor_name} to params dict")
                return param_var
            else:
                # Use constant directly
                const_expr = relax.const(value)
                self._nodes[tensor_name] = const_expr
                return const_expr

    def _new_var(self, var_name: str, shape: List, dtype: str = "float32"):
        """Creates a new Relax variable - same as ONNX version."""
        return relax.Var(
            name_hint=var_name, struct_info=relax.TensorStructInfo(shape=shape, dtype=dtype)
        )        
    
    def bad_get_tensor_expr(self, tensor_wrapper):
        """Get Relax expression for a tensor."""
        tensor_name = get_tensor_name(self.current_subgraph, tensor_wrapper.tensor_idx)
        
        if tensor_name in self._nodes:
            return self._nodes[tensor_name]
        else:
            if not self._has_tensor_value(tensor_wrapper):
                raise ValueError(f"Tensor '{tensor_name}' is not an input and has no constant value.")

            value = self._get_tensor_value(tensor_wrapper)
            dtype = self._get_tensor_type_str(tensor_wrapper.tensor.Type())
            
            # Create constant directly instead of parameter
            const_expr = relax.const(value, dtype)
            self._nodes[tensor_name] = const_expr
            return const_expr

    def complex_get_tensor_expr(self, tensor_wrapper):
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
            # Store as (var, value) tuple like ONNX does - consistent approach
            self._params[tensor_name] = (param_var, np.array(value))
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

        # The new shape can be provided as a second input tensor or in the options.
        if len(input_tensors) == 2:
            shape_tensor = input_tensors[1]
            if self._has_tensor_value(shape_tensor):
                # Shape is provided as a constant tensor.
                new_shape = self._get_tensor_value(shape_tensor).tolist()
                return self.bb.normalize(relax.op.reshape(data_expr, new_shape))
            else:
                # Shape is provided as a dynamic tensor from another operator.
                shape_input_expr = self._get_tensor_expr(shape_tensor)
                
                # For dynamic reshape, we need to convert the tensor to a shape
                # First ensure the shape tensor is int64 (required for shape operations)
                if hasattr(shape_input_expr.struct_info, 'dtype') and shape_input_expr.struct_info.dtype != 'int64':
                    shape_input_expr = self.bb.normalize(relax.op.astype(shape_input_expr, 'int64'))
                
                # Convert tensor to shape using tensor_to_shape
                shape_expr = self.bb.normalize(relax.op.tensor_to_shape(shape_input_expr))
                
                # Now use the shape expression for reshape
                return self.bb.normalize(relax.op.reshape(data_expr, shape_expr))
        else:
            # Shape is provided in the operator options.
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
        
    def convert_shape(self, subgraph, op):
        """Convert TFLite SHAPE operator."""
        self.current_subgraph = subgraph
        input_tensors = self._get_input_tensors(subgraph, op)
        assert len(input_tensors) == 1

        input_expr = self._get_tensor_expr(input_tensors[0])

        # In Relax, shape_of returns a ShapeExpr. We need to convert it to a tensor
        # for it to be used by other operators as a tensor input.
        shape_expr = relax.op.shape_of(input_expr)

        # The output tensor is typically int32, but we let TVM infer it.
        return self.bb.normalize(relax.op.shape_to_tensor(shape_expr))

    def convert_pack(self, subgraph, op):
        """Convert TFLite PACK operator."""
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.PackOptions import PackOptions
        except ImportError:
            raise ImportError("The tflite package must be installed")

        self.current_subgraph = subgraph
        input_tensors = self._get_input_tensors(subgraph, op)

        input_exprs = [self._get_tensor_expr(t) for t in input_tensors]

        assert op.BuiltinOptionsType() == BuiltinOptions.PackOptions
        op_options = op.BuiltinOptions()
        pack_options = PackOptions()
        pack_options.Init(op_options.Bytes, op_options.Pos)
        axis = pack_options.Axis()

        # Determine the common dtype for all inputs
        dtypes = []
        for expr in input_exprs:
            if hasattr(expr.struct_info, 'dtype') and expr.struct_info.dtype is not None:
                dtypes.append(expr.struct_info.dtype)
        
        if dtypes:
            # Find a common dtype - prefer int64 over int32, float32 over integer types
            unique_dtypes = list(set(dtypes))
            if len(unique_dtypes) > 1:
                # Define dtype precedence
                dtype_precedence = {
                    'bool': 0,
                    'int8': 1,
                    'uint8': 2,
                    'int16': 3,
                    'int32': 4,
                    'int64': 5,
                    'float16': 6,
                    'float32': 7,
                    'float64': 8
                }
                
                # Choose the dtype with highest precedence
                common_dtype = max(unique_dtypes, 
                                   key=lambda x: dtype_precedence.get(x, 0))
                
                # Cast all inputs to common dtype
                casted_exprs = []
                for expr in input_exprs:
                    if hasattr(expr.struct_info, 'dtype') and expr.struct_info.dtype != common_dtype:
                        casted_expr = self.bb.normalize(relax.op.astype(expr, common_dtype))
                        casted_exprs.append(casted_expr)
                    else:
                        casted_exprs.append(expr)
                
                input_exprs = casted_exprs

        # The stack operator might not be available in older TVM versions.
        # Emulate it with expand_dims and concat.
        expanded_exprs = []
        for tensor_expr in input_exprs:
            expanded = self.bb.normalize(relax.op.expand_dims(tensor_expr, axis=axis))
            expanded_exprs.append(expanded)

        return self.bb.normalize(relax.op.concat(expanded_exprs, axis=axis))

    def convert_strided_slice(self, subgraph, op):
        """Convert TFLite STRIDED_SLICE operator."""
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.StridedSliceOptions import StridedSliceOptions
        except ImportError:
            raise ImportError("The tflite package must be installed")

        self.current_subgraph = subgraph
        input_tensors = self._get_input_tensors(subgraph, op)
        assert len(input_tensors) >= 3, "strided_slice requires at least 3 inputs (data, begin, end)"

        data_expr = self._get_tensor_expr(input_tensors[0])

        # Get TFLite strided slice options and masks
        begin_mask = 0
        end_mask = 0
        ellipsis_mask = 0
        new_axis_mask = 0
        shrink_axis_mask = 0

        if op.BuiltinOptionsType() == BuiltinOptions.StridedSliceOptions:
            op_options = op.BuiltinOptions()
            slice_options = StridedSliceOptions()
            slice_options.Init(op_options.Bytes, op_options.Pos)

            begin_mask = slice_options.BeginMask()
            end_mask = slice_options.EndMask()
            ellipsis_mask = slice_options.EllipsisMask()
            new_axis_mask = slice_options.NewAxisMask()
            shrink_axis_mask = slice_options.ShrinkAxisMask()

        # Extract begin values
        begin_tensor = input_tensors[1]
        if not self._has_tensor_value(begin_tensor):
            raise NotImplementedError(
                "Dynamic begin values for strided_slice are not fully supported. "
                "The begin tensor must be a constant."
            )

        begin_values = self._get_tensor_value(begin_tensor).flatten()

        # Extract end values
        end_tensor = input_tensors[2]
        if not self._has_tensor_value(end_tensor):
            raise NotImplementedError(
                "Dynamic end values for strided_slice are not fully supported. "
                "The end tensor must be a constant."
            )

        end_values = self._get_tensor_value(end_tensor).flatten()

        # Extract strides values (optional)
        strides_values = None
        if len(input_tensors) > 3 and input_tensors[3].tensor_idx != -1:
            strides_tensor = input_tensors[3]
            if self._has_tensor_value(strides_tensor):
                strides_values = self._get_tensor_value(strides_tensor).flatten()
            else:
                # Default strides of 1 for all dimensions
                strides_values = np.ones(len(begin_values), dtype=np.int32)
        else:
            # Default strides of 1 for all dimensions
            strides_values = np.ones(len(begin_values), dtype=np.int32)

        # Get data shape for mask processing
        data_shape = data_expr.struct_info.shape
        ndim = len(data_shape)

        # Process masks to modify begin/end values
        processed_begin = []
        processed_end = []
        processed_strides = []
        axes = []

        for i in range(len(begin_values)):
            # Handle begin_mask: if bit i is set, use 0 instead of begin[i]
            if begin_mask & (1 << i):
                begin_val = 0
            else:
                begin_val = int(begin_values[i])

            # Handle end_mask: if bit i is set, use dimension size instead of end[i]
            if end_mask & (1 << i):
                if i < ndim:
                    # Use the actual dimension size
                    if isinstance(data_shape[i], tir.IntImm):
                        end_val = int(data_shape[i])
                    elif hasattr(data_shape[i], "value"):
                        end_val = int(data_shape[i].value)
                    else:
                        # For symbolic shapes, we'll use a large number and rely on runtime bounds checking
                        end_val = 2147483647  # Max int32
                else:
                    end_val = 2147483647
            else:
                end_val = int(end_values[i])

            stride_val = int(strides_values[i])

            # Handle negative indices by converting to positive
            if begin_val < 0 and i < ndim:
                if isinstance(data_shape[i], tir.IntImm):
                    begin_val += int(data_shape[i])
                elif hasattr(data_shape[i], "value"):
                    begin_val += int(data_shape[i].value)
                # For symbolic shapes, leave negative indices as-is

            if end_val < 0 and i < ndim:
                if isinstance(data_shape[i], tir.IntImm):
                    end_val += int(data_shape[i])
                elif hasattr(data_shape[i], "value"):
                    end_val += int(data_shape[i].value)
                # For symbolic shapes, leave negative indices as-is

            processed_begin.append(begin_val)
            processed_end.append(end_val)
            processed_strides.append(stride_val)
            axes.append(i)

        # Warn about unsupported mask features
        if ellipsis_mask != 0:
            print(f"Warning: ellipsis_mask ({ellipsis_mask}) in strided_slice is not fully supported")

        if new_axis_mask != 0:
            print(f"Warning: new_axis_mask ({new_axis_mask}) in strided_slice is not fully supported")

        # Convert to PrimValue tuples as required by Relax
        begin_tuple = tuple(relax.PrimValue(b) for b in processed_begin)
        end_tuple = tuple(relax.PrimValue(e) for e in processed_end)
        strides_tuple = tuple(relax.PrimValue(s) for s in processed_strides)

        # Perform the strided slice operation
        result = self.bb.normalize(
            relax.op.strided_slice(
                data_expr,
                begin=begin_tuple,
                end=end_tuple,
                strides=strides_tuple,
                axes=axes,
                assume_inbound=False,
            )
        )

        # Handle shrink_axis_mask: remove dimensions where the mask bit is set
        if shrink_axis_mask != 0:
            squeeze_axes = []
            for i in range(len(begin_values)):
                if shrink_axis_mask & (1 << i):
                    squeeze_axes.append(i)

            if squeeze_axes:
                # Apply squeeze to remove the specified dimensions
                result = self.bb.normalize(relax.op.squeeze(result, axis=squeeze_axes))

        return result

    def convert_transpose(self, subgraph, op):
        """
        Convert the TFLite TRANSPOSE operator to a Relax permute_dims operator,
        ensuring the output is contiguous for the runtime.
        """
        self.current_subgraph = subgraph
        
        # Get the input tensors: the data and the permutation axes.
        input_tensors = self._get_input_tensors(subgraph, op)
        assert len(input_tensors) == 2, "TRANSPOSE operator expects 2 inputs: data and permutation."

        data_expr = self._get_tensor_expr(input_tensors[0])
        
        perm_values = self._get_tensor_value(input_tensors[1])
        if perm_values is None:
            raise ValueError(
                "Dynamic (non-constant) permutation axes for the TRANSPOSE operator are not supported."
            )

        axes = tuple(perm_values.tolist())
        
        # 1. Perform the transpose. The result is potentially non-contiguous.
        permuted_expr = self.bb.normalize(relax.op.permute_dims(data_expr, axes=axes))
        
        # 2. FIX: Force a contiguous layout using a flatten-and-reshape trick.
        # A simple reshape(x, x.shape) can be optimized away. By first reshaping
        # to a 1D tensor and then back to the target shape, we force the compiler
        # to create a new, contiguous tensor in memory.
        original_shape = permuted_expr.struct_info.shape
        flattened_expr = self.bb.normalize(relax.op.reshape(permuted_expr, [-1]))
        contiguous_result = self.bb.normalize(relax.op.reshape(flattened_expr, original_shape))

        self._debug_log(
            f"Transpose (via reshape flatten): {data_expr.struct_info.shape} -> {contiguous_result.struct_info.shape}"
        )
        
        return contiguous_result


     
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
        from tflite.BuiltinOptions import BuiltinOptions
        from tflite.ReducerOptions import ReducerOptions

        self.current_subgraph = subgraph
        input_tensors = self._get_input_tensors(subgraph, op)
        assert len(input_tensors) == 2

        data_expr = self._get_tensor_expr(input_tensors[0])
        data_sinfo = data_expr.struct_info   # ✅ not relax.analysis.get_static_type
        rank = len(getattr(data_sinfo, "shape", [])) if hasattr(data_sinfo, "shape") else None

        # Parse axes tensor (2nd input)
        axes_tensor = input_tensors[1]
        axes = None
        if self._has_tensor_value(axes_tensor):
            axes_value = self._get_tensor_value(axes_tensor)  # numpy array or None
            if axes_value is not None:
                axes_list = axes_value.tolist()
                if len(axes_list) == 0:
                    axes = []  # explicit empty => no-op reduction
                else:
                    # normalize negatives and dedup
                    if rank is not None:
                        norm = []
                        for a in axes_list:
                            a = int(a)
                            if a < 0:
                                a += rank
                            norm.append(a)
                        axes = sorted(set(norm))
                    else:
                        # fallback: pass as-is; Relax will handle at runtime
                        axes = tuple(int(a) for a in axes_list)

        # KeepDims from ReducerOptions
        keepdims = False
        if op.BuiltinOptionsType() == BuiltinOptions.ReducerOptions:
            opts = op.BuiltinOptions()
            ro = ReducerOptions()
            ro.Init(opts.Bytes, opts.Pos)
            keepdims = bool(ro.KeepDims())

        # Decide axis argument:
        # - axes == None  -> reduce all dims (TF default when axes missing)
        # - axes == []    -> no-op reduction (identity), so just return data_expr
        if axes == []:
            return data_expr  # identity, regardless of keepdims

        return self.bb.normalize(relax.op.mean(data_expr, axis=axes, keepdims=keepdims))

    # Pool operators
    def convert_average_pool2d(self, subgraph, op):
        """Convert TFLite AVERAGE_POOL_2D operator."""
        return self._convert_pool2d(subgraph, op, "avg")

    def convert_max_pool2d(self, subgraph, op):
        """Convert TFLite MAX_POOL_2D operator."""
        return self._convert_pool2d(subgraph, op, "max")

    def _convert_pool2d(self, subgraph, op, pool_type):
        """Generic pool2d conversion with proper SAME padding."""
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.Padding import Padding
            from tflite.Pool2DOptions import Pool2DOptions
        except ImportError:
            raise ImportError("The tflite package must be installed")

        self._debug_log(f"Converting {pool_type}_pool2d")
        self.current_subgraph = subgraph
        input_tensors = self._get_input_tensors(subgraph, op)
        assert len(input_tensors) == 1

        data_expr = self._get_tensor_expr(input_tensors[0])
        self._debug_log(f"Input shape: {data_expr.struct_info.shape}")

        assert op.BuiltinOptionsType() == BuiltinOptions.Pool2DOptions
        op_options = op.BuiltinOptions()
        pool_options = Pool2DOptions()
        pool_options.Init(op_options.Bytes, op_options.Pos)

        kernel_h = pool_options.FilterHeight()
        kernel_w = pool_options.FilterWidth()
        stride_h = pool_options.StrideH()
        stride_w = pool_options.StrideW()
        padding = pool_options.Padding()

        self._debug_log(f"Pool params - kernel: [{kernel_h}, {kernel_w}], stride: [{stride_h}, {stride_w}], padding: {padding}")

        pool_size = [kernel_h, kernel_w]
        strides = [stride_h, stride_w]

        # Get input shape for SAME padding calculation
        input_shape = data_expr.struct_info.shape
        if len(input_shape) == 4:  # NHWC format
            input_h = input_shape[1]
            input_w = input_shape[2]
            self._debug_log(f"Input spatial dims: H={input_h}, W={input_w}")
        else:
            raise ValueError(f"Expected 4D input for pool2d, got {len(input_shape)}D")

        # Handle padding
        if padding == Padding.VALID:
            padding_val = [0, 0, 0, 0]
            self._debug_log("Using VALID padding: [0, 0, 0, 0]")
        elif padding == Padding.SAME:
            # Calculate SAME padding properly
            def calculate_same_padding(input_size, kernel_size, stride):
                """Calculate SAME padding for a single dimension."""
                if isinstance(input_size, (tvm.tir.IntImm, int)):
                    input_size_val = int(input_size) if hasattr(input_size, 'value') else int(input_size)
                elif hasattr(input_size, 'value'):
                    input_size_val = int(input_size.value)
                else:
                    # For symbolic shapes, assume worst case
                    input_size_val = 224  # reasonable default
                    self._debug_log(f"Using default input size {input_size_val} for symbolic shape")
                
                output_size = (input_size_val + stride - 1) // stride
                total_pad = max(0, (output_size - 1) * stride + kernel_size - input_size_val)
                pad_before = total_pad // 2
                pad_after = total_pad - pad_before
                self._debug_log(f"Padding calc: input={input_size_val}, kernel={kernel_size}, stride={stride} -> total_pad={total_pad}, before={pad_before}, after={pad_after}")
                return pad_before, pad_after

            pad_top, pad_bottom = calculate_same_padding(input_h, kernel_h, stride_h)
            pad_left, pad_right = calculate_same_padding(input_w, kernel_w, stride_w)
            
            padding_val = [pad_top, pad_left, pad_bottom, pad_right]
            self._debug_log(f"SAME padding calculated: {padding_val}")
        else:
            raise ValueError(f"Unsupported padding type: {padding}")

        self._debug_log(f"Final pool params - size: {pool_size}, strides: {strides}, padding: {padding_val}")

        try:
            if pool_type == "avg":
                result = relax.op.nn.avg_pool2d(
                    data_expr,
                    pool_size=pool_size,
                    strides=strides,
                    padding=padding_val,
                    layout="NHWC",
                )
            elif pool_type == "max":
                result = relax.op.nn.max_pool2d(
                    data_expr,
                    pool_size=pool_size,
                    strides=strides,
                    padding=padding_val,
                    layout="NHWC",
                )
            else:
                raise ValueError(f"Unsupported pool type: {pool_type}")
            
            self._debug_log(f"Created {pool_type}_pool2d operation successfully")
            result = self.bb.normalize(result)
            self._debug_log(f"Normalized {pool_type}_pool2d operation successfully")
            return result
            
        except Exception as e:
            self._debug_log(f"ERROR in {pool_type}_pool2d operation: {e}")
            self._debug_log(f"  data_expr type: {type(data_expr)}")
            self._debug_log(f"  data_expr struct_info: {data_expr.struct_info}")
            raise

    # Pool operators
    def convert_average_pool2d(self, subgraph, op):
        """Convert TFLite AVERAGE_POOL_2D operator."""
        return self._convert_pool2d(subgraph, op, "avg")

    def convert_max_pool2d(self, subgraph, op):
        """Convert TFLite MAX_POOL_2D operator."""
        return self._convert_pool2d(subgraph, op, "max")
        
    # Convolution operators
    def convert_conv2d(self, subgraph, op):
        """Convert TFLite CONV_2D operator."""
        return self._convert_conv(subgraph, op, "conv2d")

    def convert_depthwise_conv2d(self, subgraph, op):
        """Convert TFLite DEPTHWISE_CONV_2D operator."""
        return self._convert_conv(subgraph, op, "depthwise")

    def _convert_conv(self, subgraph, op, conv_type):
        """Generic convolution conversion with proper SAME padding."""
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.Conv2DOptions import Conv2DOptions
            from tflite.DepthwiseConv2DOptions import DepthwiseConv2DOptions
            from tflite.Padding import Padding
        except ImportError:
            raise ImportError("The tflite package must be installed")

        self._debug_log(f"Converting {conv_type}")
        self.current_subgraph = subgraph
        input_tensors = self._get_input_tensors(subgraph, op)
        assert len(input_tensors) >= 2

        data_expr = self._get_tensor_expr(input_tensors[0])
        weight_expr = self._get_tensor_expr(input_tensors[1])
        
        self._debug_log(f"Input shape: {data_expr.struct_info.shape}")
        self._debug_log(f"Weight shape: {weight_expr.struct_info.shape}")

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

        self._debug_log(f"Conv params - stride: [{stride_h}, {stride_w}], dilation: [{dilation_h}, {dilation_w}], padding: {padding}")

        strides = [stride_h, stride_w]
        dilation = [dilation_h, dilation_w]

        # Get input and kernel dimensions for SAME padding calculation
        input_shape = data_expr.struct_info.shape
        weight_shape = weight_expr.struct_info.shape
        
        if len(input_shape) == 4:  # NHWC format
            input_h = input_shape[1]
            input_w = input_shape[2]
            self._debug_log(f"Input spatial dims: H={input_h}, W={input_w}")
        else:
            raise ValueError(f"Expected 4D input for conv2d, got {len(input_shape)}D")

        # Get kernel dimensions (TFLite format depends on conv type)
        if conv_type == "conv2d":
            # TFLite Conv2D weight format: [out_channels, height, width, in_channels] (OHWI)
            kernel_h = weight_shape[1] 
            kernel_w = weight_shape[2]
        else:  # depthwise
            # TFLite DepthwiseConv2D weight format: [1, height, width, out_channels]
            kernel_h = weight_shape[1]
            kernel_w = weight_shape[2]
        
        self._debug_log(f"Kernel dims: H={kernel_h}, W={kernel_w}")

        # Handle padding with proper SAME calculation
        if padding == Padding.VALID:
            padding_val = [0, 0, 0, 0]
            self._debug_log("Using VALID padding: [0, 0, 0, 0]")
        elif padding == Padding.SAME:
            # Proper SAME padding calculation for convolution
            def calculate_conv_same_padding(input_size, kernel_size, stride, dilation):
                """Calculate SAME padding for convolution."""
                if isinstance(input_size, (tvm.tir.IntImm, int)):
                    input_size_val = int(input_size) if hasattr(input_size, 'value') else int(input_size)
                elif hasattr(input_size, 'value'):
                    input_size_val = int(input_size.value)
                else:
                    # For symbolic shapes, assume reasonable default
                    input_size_val = 224
                    self._debug_log(f"Using default input size {input_size_val} for symbolic shape")
                
                if isinstance(kernel_size, (tvm.tir.IntImm, int)):
                    kernel_size_val = int(kernel_size) if hasattr(kernel_size, 'value') else int(kernel_size)
                elif hasattr(kernel_size, 'value'):
                    kernel_size_val = int(kernel_size.value)
                else:
                    kernel_size_val = int(kernel_size)
                
                # Calculate effective kernel size with dilation
                effective_kernel_size = kernel_size_val + (kernel_size_val - 1) * (dilation - 1)
                
                # Calculate output size (same as input for SAME padding)
                output_size = (input_size_val + stride - 1) // stride
                
                # Calculate total padding needed
                total_pad = max(0, (output_size - 1) * stride + effective_kernel_size - input_size_val)
                pad_before = total_pad // 2
                pad_after = total_pad - pad_before
                
                self._debug_log(f"SAME padding calc: input={input_size_val}, kernel={kernel_size_val}, "
                            f"effective_kernel={effective_kernel_size}, stride={stride}, "
                            f"total_pad={total_pad}, before={pad_before}, after={pad_after}")
                
                return pad_before, pad_after

            pad_top, pad_bottom = calculate_conv_same_padding(input_h, kernel_h, stride_h, dilation_h)
            pad_left, pad_right = calculate_conv_same_padding(input_w, kernel_w, stride_w, dilation_w)
            
            padding_val = [pad_top, pad_left, pad_bottom, pad_right]
            self._debug_log(f"SAME padding calculated: {padding_val}")
        else:
            raise ValueError(f"Unsupported padding type: {padding}")

        # Convert weight layout from TFLite to Relax format
        if conv_type == "conv2d":
            # TFLite OHWI -> Relax OIHW
            weight_expr = self.bb.normalize(relax.op.permute_dims(weight_expr, [0, 3, 1, 2]))
            self._debug_log("Converted Conv2D weights from OHWI to OIHW")
        else: # depthwise
            # TFLite [1, H, W, C_out] -> Relax [C_out, 1, H, W] (OIHW)
            weight_expr = self.bb.normalize(relax.op.permute_dims(weight_expr, [3, 0, 1, 2]))
            self._debug_log("Converted DepthwiseConv2D weights from 1HWO to OIHW")

        try:
            if conv_type == "conv2d":
                result = relax.op.nn.conv2d(
                    data_expr,
                    weight_expr,
                    strides=strides,
                    padding=padding_val,
                    dilation=dilation,
                    data_layout="NHWC",
                    kernel_layout="OIHW",
                )
            else:
                # For depthwise conv, we need the groups parameter
                groups_value = data_expr.struct_info.shape[-1]
                groups = _extract_int_value(groups_value)
                
                result = relax.op.nn.conv2d(
                    data_expr,
                    weight_expr,
                    strides=strides,
                    padding=padding_val,
                    dilation=dilation,
                    data_layout="NHWC",
                    kernel_layout="OIHW",
                    groups=groups,
                )

            self._debug_log(f"Created {conv_type} operation successfully")
            result = self.bb.normalize(result)
            self._debug_log(f"Normalized {conv_type} operation successfully")

            # Add bias if present
            if len(input_tensors) > 2 and self._has_tensor_value(input_tensors[2]):
                bias_expr = self._get_tensor_expr(input_tensors[2])
                result = self.bb.normalize(relax.op.add(result, bias_expr))
                self._debug_log("Added bias to convolution result")

            # Log output shape for debugging
            self._debug_log(f"Conv output shape: {result.struct_info.shape}")
            return result
            
        except Exception as e:
            self._debug_log(f"ERROR in {conv_type} operation: {e}")
            raise
        
    def convert_fully_connected(self, subgraph, op):
        """Convert TFLite FULLY_CONNECTED operator."""
        self.current_subgraph = subgraph
        input_tensors = self._get_input_tensors(subgraph, op)
        assert len(input_tensors) >= 2

        data_expr = self._get_tensor_expr(input_tensors[0])
        weight_expr = self._get_tensor_expr(input_tensors[1])

        # Flatten input data to 2D for matrix multiplication if necessary
        if len(data_expr.struct_info.shape) > 2:
            data_expr = self.bb.normalize(
                relax.op.reshape(data_expr, [-1, data_expr.struct_info.shape[-1]])
            )

        # TFLite weight layout is [out_features, in_features], which is W.
        # We need to compute data @ W.T. So we transpose W.
        weight_expr = self.bb.normalize(relax.op.permute_dims(weight_expr, [1, 0]))

        # The `dense` operator is not in relax, we use matmul.
        result = self.bb.normalize(relax.op.matmul(data_expr, weight_expr))

        # Add bias if present
        if len(input_tensors) > 2 and self._has_tensor_value(input_tensors[2]):
            bias_expr = self._get_tensor_expr(input_tensors[2])
            result = self.bb.normalize(relax.op.add(result, bias_expr))

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
    keep_params_in_input: bool = True,
) -> Tuple[IRModule, Dict[str, np.ndarray]]:
    """Construct Relax expressions from the TFLite model.
    
    Parameters
    ----------
    model : tflite.Model
        The TFLite model to convert
    shape_dict : Optional[Dict[str, List]]
        Dictionary mapping input names to their shapes
    dtype_dict : Optional[Union[str, Dict[str, str]]]
        Dictionary mapping input names to their dtypes, or a single dtype string
    keep_params_in_input : bool
        Whether to keep parameters as function inputs (default: True)
        
    Returns
    -------
    mod : IRModule
        The converted Relax module
    params : Dict[str, np.ndarray]
        The model parameters as numpy arrays
    """
    importer = TFLiteGraphImporter(shape_dict, dtype_dict, keep_params_in_input)
    return importer.from_tflite(model)


def simple_from_tflite(self, model):
    """Simplified TFLite conversion."""
    self.current_model = model

    with self.bb.function("main"):
        with self.bb.dataflow():
            self._parse_model_inputs(model)
            self._convert_operators(model)

            # Get outputs
            subgraph = model.Subgraphs(0)
            model_outputs = subgraph.OutputsAsNumpy()
            outputs = [self._nodes[get_tensor_name(subgraph, i)] for i in model_outputs]
            outputs = outputs[0] if len(outputs) == 1 else relax.Tuple(outputs)
            
            output_var = self.bb.emit_output(outputs)

        # Simple function signature with just inputs
        input_list = list(self._inputs.values())
        self.bb.emit_func_output(output_var, params=input_list)

    # Get module
    relax_mod = self.bb.get()
    
    # Return empty params dict since we're using constants
    return relax_mod, {}