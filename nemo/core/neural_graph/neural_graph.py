# -*- coding: utf-8 -*-

# =============================================================================
# Copyright (c) 2020 NVIDIA. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

__all__ = [
    'NeuralGraph',
]

from collections import OrderedDict, namedtuple
from typing import Dict, Optional

from nemo.core import OperationMode
from nemo.core.neural_graph.graph_inputs import GraphInputs
from nemo.core.neural_graph.graph_outputs import GraphOutputs
from nemo.core.neural_interface import NeuralInterface
from nemo.core.neural_modules import NeuralModule
from nemo.core.neural_types import NeuralPortNameMismatchError, NeuralType, NmTensor
from nemo.package_info import __version__ as nemo_version
from nemo.utils import logging
from nemo.utils.module_port import Connection, ModulePort


class NeuralGraph(NeuralInterface):
    """
        Neural Graph class stores dynamically defined graphs of connected Neural Modules.
    """

    def __init__(self, operation_mode=OperationMode.both, name=None):
        """
            Constructor. Initializes graph variables.

            Args:
                operation_mode: Graph operation mode, that will be propagated along modules during graph creation.
                [training | eval | both] (DEFAULT: both)
                name: Name of the graph (optional)
        """
        # Initialize the inferface.
        super().__init__(name)

        # Register graph.
        self._name = self._app_state.register_graph(self, name)

        # Store name and operation mode.
        self._operation_mode = operation_mode

        # "Modules" - list of modules constituting "nodes" in a given graph.
        self._modules = {}

        # All tensors produced within this graph (dict of dicts).
        # This stores  "all output tensors" dictionary, where the key is the name of "producer" module,
        # and the value contains a dictionary of all tensors produced by it.
        self._all_tensors = {}

        # "Steps": order of the  execution of modules in a graph.
        self._steps = OrderedDict()

        # Bound inputs.
        self._inputs = GraphInputs()

        # Bound outputs.
        self._outputs = GraphOutputs(self._all_tensors)

        # Flag indicating whether the "default" output ports/tensors will be automatically bound.
        self.default_output_binding = True

    def __call__(self, **kwargs):
        """
            This method "nests" one existing neural graph into another one.
            Also checks if all inputs were provided and properly connects them.

            Args:
                kwargs: keyword arguments containing dictionary of (input_port_name, port_content).

        """
        # Test operation modes of the nested graphs.
        outer_mode = self._app_state.active_graph.operation_mode
        inner_mode = self.operation_mode

        if inner_mode == OperationMode.inference and outer_mode == OperationMode.training:
            raise TypeError("Cannot nest 'inference' graph into 'training'")

        if inner_mode == OperationMode.training and outer_mode == OperationMode.inference:
            raise TypeError("Cannot nest 'training' graph into 'inference'")

        if inner_mode == OperationMode.training and outer_mode == OperationMode.both:
            raise TypeError("Cannot nest 'training' graph into 'both'")

        if inner_mode == OperationMode.inference and outer_mode == OperationMode.both:
            raise TypeError("Cannot nest 'inference' graph into 'both'")

        # Check inputs: iterate through all inputs passed to the "self".
        for port_name, port_content in kwargs.items():
            # Make sure that passed arguments correspond to input port names.
            if port_name not in self.input_ports.keys():
                raise NeuralPortNameMismatchError(port_name)

        # "Nest" this graph into an active graph.
        results = self._app_state.active_graph.nest(self, kwargs)

        # Return output tensors.
        return results

    def nest(self, inner_graph, inner_graph_args):
        """
            Method nests (copies) a graph: modules, steps, topology (tensors).

            Args:
                inner_graph: Graph to be copied (will be "nested" in this (self) graph).
                inner_graph_args: inputs passed to the graph call.
        """
        # "Copy" the modules from nested graph.
        for key, module in inner_graph.modules.items():
            # Check if module with that name already exists.
            # TODO: Uncomment when we will refactor all examples so training/validation graphs won't be added
            # to the "default" graph.
            # if key in self._modules.keys():
            #    raise KeyError("Neural Graph already contains a module named {}".format(module.name))
            self._modules[key] = module

        # Next we should copy the topography - i.e. produce "real" copies of tensors.
        # In fact, instead of copying, we will produce them, following:
        # - the execution order defined in "steps"
        # - connectivity defined in tensor' consumers-ports
        # (so the same logic that will be used in graph deserialization)

        # So let us first serialize the connections of the nested graph.
        # Create a list: (producer.port -> consumer.port)
        inner_connections = []
        for tensors in inner_graph.tensors.values():
            for t in tensors.values():
                inner_connections.extend(t.connections())

        # We need to disable the binding of "defeault" ports on per module basis - we will "manually" produce
        # them only for ports that are already indicated as the "bound" ones in the inner graph.
        self.default_output_binding = False

        # Now "copy" graph execution order and topology by actually executing each step of the nested graph.
        for _, module_name in inner_graph.steps.items():
            # Both module and step will be added by the modules' call().

            # Get the module.
            module = inner_graph._modules[module_name]

            # Produce list of arguments that will be passed to a given modules.
            module_args = {}
            # Do it by:
            # - harvesing input port names of a given module,
            # - checking if the input was not bound (in the inner graph),
            # - checking if we have already tensors leading to that input (in outer graph).
            for input_port_name in module.input_ports.keys():
                # Check if this port was bound in the inner graph.
                key = inner_graph.inputs.has_binding(module_name, input_port_name)
                # If so, then we must pass whatever was passed to that port in the list of arguments.
                if key is not None:
                    module_args[input_port_name] = inner_graph_args[key]
                    # As a result, the "module" call() will bind this input!
                    continue

                # Else: find a tensor that should be passed to the given module's input.
                # Search for producer/port that we should use.
                for connection in inner_connections:
                    if (
                        connection.consumer.module_name == module_name
                        and connection.consumer.port_name == input_port_name
                    ):
                        # Got the connection!
                        producer_name = connection.producer.module_name
                        producer_port_name = connection.producer.port_name
                        break
                # Now, the tensor is already produced in outer (i.e. this) graph!
                module_args[input_port_name] = self.tensors[producer_name][producer_port_name]

            # import pdb;pdb.set_trace()
            # Ok, now we have all keyword arguments. We can call() the module.
            # This will collect all the produced output tensors and add them to this graph.
            module(**module_args)

        # At that point we have all modules, steps and tensors added to outer (self) graph.
        # Now we have to prepare the outputs.

        # This part is different from Neural Module.
        # Now the goal is NOT to create NEW "tensors", but to return the BOUND ones!
        # Still, those must be bound in the outer (active) graph, but using port names from the inner (nested) graph.

        # Get list of "the adequate output tensors".
        output_tensors = {}
        # Iterate through outputs of the inner graph.
        for key, tensor in inner_graph.output_tensors.items():
            # Find the tensors within this (outer) graph that are outpus by the same producer-port.
            producer_name = tensor.producer_name
            producer_port_name = tensor.name
            # Get adequate tensor from "outer graph" (self).
            output_tensors[key] = self.tensors[producer_name][producer_port_name]

        if len(output_tensors) == 1:
            # Return a single tensor.
            key = list(output_tensors)[0]
            results = output_tensors[key]

            # Bind the "default" output ports of the inner graph as "default" output ports of this graph.
            # Call the bind() method of bound_outputs directly, as we already have the tensors in our graph.
            # But: Use output port name of the inner graph!
            self.outputs.bind([results], [key])

        else:
            # Create a named tuple type enabling to access outputs by attributes (e.g. out.x).
            output_class_name = f'{self.__class__.__name__}Output'
            result_type = namedtuple(typename=output_class_name, field_names=output_tensors.keys())

            # Return the bound output tensors.
            results = result_type(*output_tensors.values())

            # Bind the "default" output ports of the inner graph as "default" output ports of this graph.
            # Call the bind() method of bound_outputs directly, as we already have the tensors in our graph.
            # But: Use output port name of the inner graph!
            self.outputs.bind(output_tensors.values(), output_tensors.keys())

        # Ok, now we can turn automatic binding on.
        self.default_output_binding = True

        # Return the results.
        return results

    def record_step(self, module):
        """
            Records the operation (module plus passed inputs) on a list.

            Args:
                module: Neural modules added to a given graph.
        """
        # Check if module with that name already exists - to avoid potential loops (DAG).
        # TODO: Uncomment after we will refactor all the examples, so training/validation graphs won't be added
        # to the "default" graph.
        # if module.name in self._modules.keys() and self._modules[module.name] is not module:
        #    raise KeyError("Neural Graph already contains a module named {}".format(module.name))

        # Add module to list of modules.
        self._modules[module.name] = module

        # Add step - store the module name.
        self._steps[len(self._steps)] = module.name

    def bind_outputs(self, tensors_list):
        """ Binds the output tensors.

            Args:
                tensors_list: A single tensor OR a List of tensors to be bound.
        """
        # Handle both single port and lists of ports to be bound.
        if type(tensors_list) is not list:
            tensors_list = [tensors_list]
        # Add tensors list to of tensors.
        for tensor in tensors_list:
            # Add tensor to "all" tensors dictionary.
            producer_name = tensor.producer_name
            if producer_name not in self._all_tensors.keys():
                self._all_tensors[producer_name] = {}

            port_name = tensor.name
            # Add tensor.
            self._all_tensors[producer_name][port_name] = tensor

        # Bind the tensors as graph outputs.
        if self.default_output_binding:
            self.outputs.bind(tensors_list)

    @property
    def inputs(self):
        """
            Returns graph inputs.

        Returns:
            A graph input.
        """
        return self._inputs

    @property
    def input_ports(self) -> Optional[Dict[str, NeuralType]]:
        """
            Returns definitions of graph input ports (dict of Neural Types).

        .. note::
            This method actually returns a dictionary with definitions (like Neural Modules).
            In order to get access to actual graph inputs please call the inputs() method.

        Returns:
            A graph input ports definitions.
        """
        return self._inputs.definitions

    @property
    def outputs(self):
        """
            Returns graph outputs.

        Returns:
            A graph outputs.
        """
        return self._outputs

    @property
    def output_ports(self) -> Optional[Dict[str, NeuralType]]:
        """
            Returns definitions of module output ports (dict of Neural Types).

        .. note::
            This method actually returns a dictionary with definitions (like Neural Modules).
            In order to get access to actual graph outpus please call the outputs() method.

        Returns:
            A graph output ports definitions.
            
        """
        return self._outputs.definitions

    @property
    def output_tensors(self):
        """
            Returns graph output tensors.

        Returns:
            A graph output tensors.
        """
        return self._outputs.tensors

    @property
    def modules(self):
        """ Returns modules. """
        return self._modules

    def __getitem__(self, key):
        """ Returns module given its name (name of the variable).

            Args:
                key: Name of the variable.
        """
        if key not in self._modules.keys():
            raise KeyError("Neural Graph doesn't contain a module named {}".format(key))
        return self._modules[key]

    def __len__(self):
        """ Returns number of modules (vertices) in a given graph. """
        return len(self._modules)

    @property
    def steps(self):
        """ Returns steps. """
        return self._steps

    @property
    def tensors(self):
        """ Returns the (double) dictionary of all output tensors, aggregated by modules (key) and (output) port name. """
        return self._all_tensors

    @property
    def operation_mode(self):
        """ Returns operation mode. """
        return self._operation_mode

    def __enter__(self):
        """ 
            Activates this graph.
        
            Returns:
                The graph object.
        """
        self._app_state.active_graph = self
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """
            Deactivates the current graph.
        """
        self._app_state.active_graph = None

    def activate(self):
        """ 
            Activates this graph.
        """
        self._app_state.active_graph = self

    def deactivate(self):
        """
            Deactivates the current graph.
        """
        self._app_state.active_graph = None

    def export_to_config(self, config_file):
        """ Exports the neural graph to a file.
        
            Args:
                config_file: Name (and path) of the config file (YML) to be written to.
        """
        # Create a dictionary where we will add the whole information.
        config = {self.name: {}}
        # Get shortcut.
        graph = config[self.name]
        # Serialize modules.
        graph["modules"] = self.__serialize_modules()

    def serialize(self):
        """ Method serializes the whole graph.

            Returns:
                Dictionary containing description of the whole graph.
        """
        # Create a dictionary representing the serialized object.
        serialized_graph = {}

        # Add "header" with module "specification".
        serialized_graph["header"] = self.__serialize_header()

        # Add modules.
        serialized_graph["modules"] = self.__serialize_modules()

        # Add steps.
        serialized_graph["steps"] = self.__serialize_steps()

        # Add connectinos.
        serialized_graph["connections"] = self.__serialize_connections()

        # Serialize graph (bound) inputs.
        serialized_graph["inputs"] = self._inputs.serialize()

        # Serialize graph (bound) outputs.
        serialized_graph["outputs"] = self._outputs.serialize()

        # Return the dictionary.
        return serialized_graph

    def __serialize_header(self):
        """ Private method responsible for serializing the graph header.

            Returns:
                Dictionary containing description of the whole graph.        
        """
        # Generate full_spec of the class.
        full_spec = str(self.__module__) + "." + str(self.__class__.__qualname__)
        header = {"nemo_core_version": nemo_version, "full_spec": full_spec}
        # Add operation mode.
        if self._operation_mode == OperationMode.training:
            header["operation_mode"] = "training"
        elif self._operation_mode == OperationMode.inference:
            header["operation_mode"] = "inference"
        else:
            header["operation_mode"] = "both"
        # Return header.
        return header

    def __serialize_modules(self):
        """ Private method responsible for serializing the modules present in the graph.

            Returns:
                Dictionary containing description of all graph modules.
        """
        serialized_modules = {}
        for name, module in self._modules.items():
            serialized_modules[name] = module.serialize()
        return serialized_modules

    def __serialize_steps(self):
        """ Private method responsible for serializing the steps (order of module executions).

            Returns:
                Dictionary containing description of the steps.
        """
        serialized_steps = {}
        for no, module_name in self._steps.items():
            serialized_steps[no] = module_name
        return serialized_steps

    def __serialize_connections(self):
        """ Private method responsible for serializing the connections in the graph.

            Returns:
                List containing "connections" between modules.
        """
        serialized_connections = []
        # Iterate through "tensor modules".
        for tensors in self._all_tensors.values():
            # Iterate through "tensor output ports".
            for tensor in tensors.values():
                # "Transform" tensor to the list of connections.
                for c in tensor.connections():
                    # Serialize!
                    source = c.producer.module_name + "." + c.producer.port_name
                    target = c.consumer.module_name + "." + c.consumer.port_name
                    serialized_connections.append(source + "->" + target)
        return serialized_connections

    @classmethod
    def import_from_config(cls, config_file, reuse_existing_modules=False, overwrite_params={}, name=None):
        """
            Class method importing the neural graph from the configuration file.
            Raises an ImportError exception when config file is invalid.

            Args:
                config_file: path (absolute or relative) and name of the config file (YML)

                reuse_existing_modules: If the modules with (name, type, init_params) are already created, import will
                connect to them instead of creating new instances.

                overwrite_params: Dictionary containing parameters that will be added to or overwrite (!) the default
                parameters loaded from the configuration file

                name: Name of the new graph (optional, DEFAULT: NONE)

            Returns:
                Instance of the created NeuralGraph object.
        """
        logging.info("Loading configuration of a new Neural Graph from the `{}` file".format(config_file))

        # TODO: validate the content of the configuration file (its header).
        loaded_config = []  # cls.__validate_config_file(config_file, section_name)
        # TODO: overwrite params.

        # "Deserialize" the graph.
        new_graph = cls.deserialize(loaded_config, reuse_existing_modules, name)

        return new_graph

    @classmethod
    def deserialize(cls, configuration, reuse_existing_modules=False, name=None):
        """
            Class method creating a graph instance by deserializing the provided configuratino.

            Args:
                configuration: Dictionary containing serialized graph.

                reuse_existing_modules: If the modules with (name, type, init_params) are already created, import will
                connect to them instead of creating new instances.

            Returns:
                Instance of the created NeuralGraph object.
        """
        # Deserialize header and get object class.
        operation_mode = cls.__deserialize_header(configuration["header"])

        # Create the graph instance.
        new_graph = NeuralGraph(operation_mode=operation_mode, name=name)
        logging.info(
            "Instantiated a new Neural Graph named `{}` with mode `{}`".format(
                new_graph.name, new_graph.operation_mode
            )
        )
        # Deserialize modules.
        modules = new_graph.__deserialize_modules(configuration["modules"], reuse_existing_modules)

        # Deserialize steps.
        steps = new_graph.__deserialize_steps(configuration["steps"])

        # Deserialize the connections between modules.
        connections = new_graph.__deserialize_connections(configuration["connections"])

        # Deserialize input bindings - return it in an external entity.
        inputs = GraphInputs.deserialize(configuration["inputs"], modules)

        # Deserialize "manual" output bindings.
        new_graph._outputs.deserialize(configuration["outputs"], modules)

        # Now we have to execute the graph, following the steps and connections.
        new_graph.__execute_and_create_tensors(steps, modules, connections, inputs)

        # Return the graph instance.
        return new_graph

    @classmethod
    def __deserialize_header(cls, serialized_header):
        """ Private class method deserializing the header and extracts the general information.
            
            Args:
                serialized_header: Dictionary containing graph header.

            Returns:
                Operation mode.
        """
        # Parse the "full specification" - do not need that now.
        # spec_list = serialized_header["full_spec"].split(".")

        # Get operation mode.
        if serialized_header["operation_mode"] == "training":
            operation_mode = OperationMode.training
        elif serialized_header["operation_mode"] == "inference":
            operation_mode = OperationMode.inference
        else:
            operation_mode = OperationMode.both

        # Return the mode.
        return operation_mode

    def __deserialize_modules(self, serialized_modules, reuse_existing_modules):
        """ Private method deserializing the modules present in the graph.

            Args:
                serialized_modules: Dictionary containing graph modules.

            Returns:
                Dictionary of modules.
        """
        modules = {}
        for name, module_params in serialized_modules.items():
            # Check if module already exists.
            if self._app_state.modules.has(name):
                # Check if we can reuse the existing modules.
                if reuse_existing_modules:
                    modules[name] = self._app_state.modules[name]
                else:
                    raise KeyError("A module with name `{}` already exists!".format(name))
            else:
                # Ok, create a new module.
                modules[name] = NeuralModule.deserialize(module_params)
        # Ok, done.
        return modules

    def __deserialize_steps(self, serialized_steps):
        """ Private method deserializing the steps (order of module executions).

            Args:
                serialized_steps: Dictionary containing serialized steps.

            Returns:
                Odered dict with steps.
        """
        steps = OrderedDict()
        for i in range(len(serialized_steps)):
            steps[i] = serialized_steps[i]
        # Ok, done.
        return steps

    def __deserialize_connections(self, serialized_connections):
        """ Private method deserializing the connections in the graph.

            Args:
                serialized_steps: Dictionary containing serialized connections.

            Returns:
                List of connections, in a format enabling graph traversing.
        """
        connections = []
        # Deserialize connections one by one.
        for c in serialized_connections:
            # Deserialize!
            [producer, consumer] = c.split("->")
            [producer_name, producer_port_name] = producer.split(".")
            [consumer_name, consumer_port_name] = consumer.split(".")
            producer_mp = ModulePort(producer_name, producer_port_name)
            consumer_mp = ModulePort(consumer_name, consumer_port_name)
            # Add connection.
            connections.append(Connection(producer_mp, consumer_mp))
        # Ok, done.
        return connections

    def __execute_and_create_tensors(self, steps, modules, connections, inputs):
        """ Method creates (internal) tensors of the graph by executing it following the order and using 
            the provided connections and inputs.

            Args:
                steps: 
                modules
                connections
                inputs
        
        """
        # Activate this graph, so all the tensors will be added to this !
        self.activate()

        # We need to disable the binding of "defeault" ports on per module basis - we will "manually" produce
        # them only for ports that are already indicated as the "bound" ones in the inner graph.
        # self.default_output_binding = False

        # Now "copy" graph execution order and topology by actually executing each step of the nested graph.
        for _, module_name in steps.items():
            # Both module and step will be added by the modules' call().

            # Get the module.
            module = modules[module_name]

            # Produce list of arguments that will be passed to a given module.
            module_args = {}
            # Do it by:
            # - harvesing input port names of a given module,
            # - checking if the input was not bound (in the inner graph),
            # - checking if we have already tensors leading to that input (in outer graph).
            for input_port_name in module.input_ports.keys():
                # Check if this port was bound in the inner graph.
                key = inputs.has_binding(module_name, input_port_name)

                # import pdb;pdb.set_trace()
                # If so, then we must pass the binding!
                if key is not None:
                    # Copy the port "definition" (i.e. is NeuralType) using the same port name.
                    self.inputs[key] = inputs[key]

                    # Pass this object to module input argument.
                    module_args[input_port_name] = self.inputs[key]

                # Else: find a tensor that should be passed to the given module's input.
                else:
                    # Search for producer/port that we should use.
                    for connection in connections:
                        if (
                            connection.consumer.module_name == module_name
                            and connection.consumer.port_name == input_port_name
                        ):
                            # Got the connection!
                            producer_name = connection.producer.module_name
                            producer_port_name = connection.producer.port_name
                            break
                    # Now, the tensor is already produced in outer (i.e. this) graph!
                    module_args[input_port_name] = self.tensors[producer_name][producer_port_name]
                # End: for

            # Ok, now we have all keyword arguments. We can call() the module.
            # This will collect all the produced output tensors and add them to this graph.
            module(**module_args)

        # At that point we have all modules, steps and tensors added to outer (self) graph.
        # Now we have to prepare the outputs.

        # Deactivate graph.
        self.deactivate()

        # Ok, now we can turn automatic binding on.
        # self.default_output_binding = True

    def __str__(self):
        """ Prints a nice summary. """
        # TODO: a nice summary. ;)
        desc = "`{}` ({}):\n".format(self.name, len(self._steps))
        for op in self._steps:
            desc = desc + "  {}\n".format(type(op[0]).__name__)
        return desc

    def list_modules(self):
        """ Lists modules. """
        desc = "{} ({}):\n".format(self.name, len(self._modules))
        for key, value in self._modules.items():
            desc += " * `{}` ({})\n".format(key, value)
        return desc

    def show_inputs(self):
        print("bound input ports: ")
        # for key, value in self._bound_input_ports.items():
        #    print(" * `{}`: `{}` ({})".format(key, value, type(value)))

        print("bound input tensors: ")
        # for key, value in self._bound_input_tensors.items():
        #    print(" * `{}`: `{}` ({})".format(key, value, type(value)))

    def show_outputs(self):
        print("bound (default) output ports: ")
        # for key, value in self._bound_output_ports_default.items():
        #    print(" * `{}`: `{}` ({})".format(key, value, type(value)))

        print("bound (default) output tensors: ")
        # for key, value in self._bound_output_tensors_default.items():
        #    print(" * `{}`: `{}` ({})".format(key, value, type(value)))