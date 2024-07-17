import numpy as np
import tensorrt as trt
import torch
from collections import namedtuple


class LoadLSTMEngine:
    def __init__(self, engine_path, seq_len=5):
        self.seq_len = seq_len

        logger = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(logger)
        with open(engine_path, 'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())

        context = engine.create_execution_context()

        dtype = trt.float32
        input_shape = (1, seq_len, 39)  # 2维坐标
        output_shape = 3
        hidden_size = 32

        input_binding = {
            'input': (input_shape, dtype),
            'hidden_state': ((2, 1, hidden_size), dtype),
            'cell_state': ((2, 1, hidden_size), dtype),
            'output': (output_shape, dtype),
        }

        bindings = []
        bindings.append(engine.get_binding_index('input'))
        bindings.append(engine.get_binding_index('hidden_state'))
        bindings.append(engine.get_binding_index('cell_state'))
        bindings.append(engine.get_binding_index('output'))

        binding_addrs = []
        for binding in bindings:
            size = trt.volume(engine.get_binding_shape(binding)) * dtype.itemsize
            device_mem = torch.empty(size, dtype=torch.uint8, device=torch.device('cuda'))
            binding_addrs.append(int(device_mem.data_ptr()))

        self.context = context
        self.bindings = input_binding
        self.binding_addrs = binding_addrs

    def infer(self, input_data):
        self.binding_addrs[0] = int(input_data.data_ptr())

        self.context.execute_v2(self.binding_addrs)

        output = torch.from_numpy(np.empty(self.bindings['output'][0]))
        output.data_ptr = self.binding_addrs[-1]
        return output