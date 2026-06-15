import sys
import types

# Create a mock api_implementation module to avoid Python 3.14 Protobuf C-extension crash
if 'google.protobuf.internal.api_implementation' not in sys.modules:
    api_impl = types.ModuleType('google.protobuf.internal.api_implementation')
    api_impl.Type = lambda: 'python'
    api_impl.Version = lambda: 2
    api_impl.IsPythonDefaultSerializationDeterministic = lambda: False
    api_impl._implementation_type = 'python'
    api_impl._c_module = None
    sys.modules['google.protobuf.internal.api_implementation'] = api_impl
