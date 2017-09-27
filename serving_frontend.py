import json
from functools import partial

from service_manager import ServiceManager
from flask_handler import FlaskRequestHandler


class ServingFrontend(object):
    """ServingFrontend warps up all internal services including 
    model service manager, request handler. It provides all public 
    apis for users to extend and use our system.
    """
    def __init__(self, app_name):
        """
        Initialize handler for FlaskHandler and ServiceManager.

        Parameters
        ----------
        app_name : string
            App name to initialize request handler.
        """
        self.service_manager = ServiceManager()
        self.handler = FlaskRequestHandler(app_name)

    def start_model_serving(self, host, port):
        """
        Start model serving using given host and port.

        Parameters
        ----------
        host : string
            Host that server will use.
        port: int
            Port that server will use.
        """
        self.handler.start_handler(host, port)

    def load_models(self, models, ModelServiceClassDef):
        """
        Load models by using user passed Model Service Class Definitions.

        Parameters
        ----------
        models : List of model_name, model_path pairs
            List of model_name, model_path pairs that will be initialized.
        ModelServiceClassDef: python class
            Model Service Class Definition which can initialize a model service.
        """
        for model_name, model_path in models.iteritems():
            self.service_manager.load_model(model_name, model_path, ModelServiceClassDef)


    def register_module(self, user_defined_module_name):
        """
        Register a python module according to user_defined_module_name
        This module should contain a valid Model Service Class whose
        pre-process and post-process can be derived and customized.

        Parameters
        ----------
        user_defined_module_name : Python module name
            A python module will be loaded according to this name.
            

        Returns
        ----------
        List of model service class definitions.
            Those python class can be used to initialize model service.
        """
        model_class_definations = self.service_manager.parse_modelservices_from_module(user_defined_module_name)
        for ModelServiceClassDef in model_class_definations:
            self.service_manager.add_modelservice_to_registry(ModelServiceClassDef.__name__, ModelServiceClassDef)

        return model_class_definations

    def get_registered_modelservices(self, modelservice_names=None):
        """
        Get all registered Model Service Class Definitions into a dictionary 
        according to name or list of names. 
        If nothing is passed, all registered model services will be returned.

        Parameters
        ----------
        modelservice_names : string or List, optional
            Names to retrieve registered model services
            
        Returns
        ----------
        Dict of name, model service pairs
            Registered model services according to given names.
        """
        if not isinstance(modelservice_names, list) and modelservice_names is not None:
            modelservice_names = [modelservice_names]

        return self.service_manager.get_modelservices_registry(modelservice_names)

    def get_loaded_modelservices(self, modelservice_names=None):
        """
        Get all model services which are loaded in the system into a dictionary 
        according to name or list of names. 
        If nothing is passed, all loaded model services will be returned.

        Parameters
        ----------
        modelservice_names : string or List, optional
            Names to retrieve loaded model services
            
        Returns
        ----------
        Dict of name, model service pairs
            Loaded model services according to given names.
        """
        if not isinstance(modelservice_names, list) and modelservice_names is not None:
            modelservice_names = [modelservice_names]

        return self.service_manager.get_loaded_modelservices(modelservice_names)

    def get_query_string(self, field):
        """
        Get field data in the query string from request.

        Parameters
        ----------
        field : string
            Field in the query string from request.
            
        Returns
        ----------
        Object
            Field data in query string.
        """
        return self.handler.get_query_string(field)

    def add_endpoint(self, api_definition, callback, **kwargs):
        """
        Add an endpoint with OpenAPI compatible api definition and callback.

        Parameters
        ----------
        api_definition : dict(json)
            OpenAPI compatible api definition.

        callback: function
            Callback function in the endpoint.

        kwargs: dict
            Arguments for callback functions.
        """
        endpoint = api_definition.keys()[0]
        method = api_definition[endpoint].keys()[0]
        api_name = api_definition[endpoint][method]['operationId']

        self.handler.add_endpoint(api_name, endpoint, partial(callback, **kwargs), [method.upper()])

    def setup_openapi_endpoints(self, host, port):
        """
        Firstly, construct Openapi compatible api definition for 
        1. Predict
        2. Ping
        3. API description
        
        Then the api definition is used to setup web server endpoint.

        Parameters
        ----------
        host : string
            Host that server will use 

        port: int
            Host that server will use 
        """
        modelservices = self.service_manager.get_loaded_modelservices()
        # TODO: not hardcode host:port
        self.openapi_endpoints = {
            'swagger': '2.0',
            'info': {
                'version': '1.0.0',
                  'title': 'Model Serving Apis'
              },
              'host': host + ':' + str(port),
              'schemes': ['http'],
              'paths': {},
          }


        # 1. Predict endpoints
        for model_name, modelservice in modelservices.iteritems():
            input_type = modelservice.signature['input_type']
            inputs = modelservice.signature['inputs']
            output_type = modelservice.signature['output_type']
            
            # Contruct predict openapi specs
            endpoint = '/' + model_name + '/predict'
            predict_api = {
                endpoint: {
                    'post': {
                        'operationId': model_name + '_predict', 
                        'consumes': ['multipart/form-data'],
                        'produces': [output_type],
                        'parameters': [],
                        'responses': {
                            '200': {}
                        }
                    }
                }
            }
            input_names = ['input' + str(idx) for idx in range(len(inputs))]
            # Setup endpoint for each modelservice
            for idx in range(len(inputs)):
                # Check input content type to set up proper openapi consumes field
                if input_type == 'application/json':
                    parameter = {
                        'in': 'formData',
                        'name': input_names[idx],
                        'description': '%s should tensor with shape: %s' % 
                            (input_names[idx], inputs[idx]['data_shape'][1:]),
                        'required': 'true',
                        'schema': {
                            'type': 'string'
                        }
                    }
                elif input_type == 'image/jpeg':
                    parameter = {
                        'in': 'formData',
                        'name': input_names[idx],
                        'description': '%s should be image with shape: %s' % 
                            (input_names[idx], inputs[idx]['data_shape'][1:]),
                        'required': 'true',
                        'type': 'file'
                    }
                else:
                    raise Exception('%s is not supported for input content-type' % input_type)
                predict_api[endpoint]['post']['parameters'].append(parameter)

            # Contruct openapi response schema
            if output_type == 'application/json':
                responses = {
                    'description': 'OK',
                    'schema': {
                        'type': 'object',
                        'properties': {
                            'prediction': {
                                'type': 'string'
                            }
                        }
                    }
                }
            elif output_type == 'image/jpeg':
                responses = {
                    'description': 'OK',
                    'schema': {
                        'type': 'file'
                    }
                }
            else:
                raise Exception('%s is not supported for output content-type' % output_type)
            predict_api[endpoint]['post']['responses']['200'].update(responses) 

            self.openapi_endpoints['paths'].update(predict_api)

            # Setup Flask endpoint for predict api
            self.add_endpoint(predict_api, 
                              self.predict_callback, 
                              modelservice=modelservice,
                              input_names=input_names)


        # 2. Ping endpoints
        ping_api = {
            '/ping': {
                'get': {
                    'operationId': 'ping', 
                    'produces': ['application/json'],
                    'responses': {
                        '200': {
                            'description': 'OK',
                            'schema': {
                                'type': 'object',
                                'properties': {
                                    'health': {
                                        'type': 'string'
                                    }
                                }
                            }
                        }
                    }
                }
                
            }
        }
        self.openapi_endpoints['paths'].update(ping_api)
        self.add_endpoint(ping_api, self.ping_callback)


        # 3. Describe apis endpoints
        api_description_api = {
            '/api-description': {
                'get': {
                    'produces': ['application/json'],
                    'operationId': 'apiDescription', 
                    'responses': {
                        '200': {
                            'description': 'OK',
                            'schema': {
                                'type': 'object',
                                'properties': {
                                    'description': {
                                        'type': 'string'
                                    }
                                }
                            }
                        }
                    }
                }
                
            }
        }
        self.openapi_endpoints['paths'].update(api_description_api)
        self.add_endpoint(api_description_api, self.api_description)

        return self.openapi_endpoints
    
    def ping_callback(self, **kwargs):
        """
        Callback function for ping endpoint.
            
        Returns
        ----------
        Response
            Http response for ping endpiont.
        """
        try:
            for model in self.service_manager.get_loaded_modelservices().values():
                model.ping()
        except:
            return self.handler.jsonify({'health': 'unhealthy!'})

        return self.handler.jsonify({'health': 'healthy!'})

    def api_description(self, **kwargs):
        """
        Callback function for api description endpoint.

        Returns
        ----------
        Response
            Http response for api description endpiont.
        """
        return self.handler.jsonify({'description': self.openapi_endpoints})

    def predict_callback(self, **kwargs):
        """
        Callback for predict endpoint

        Parameters
        ----------
        modelservice : ModelService
            ModelService handler.

        input_names: list
            Input names in request form data.

        Returns
        ----------
        Response
            Http response for predict endpiont.
        """
        modelservice = kwargs['modelservice']
        input_names = kwargs['input_names']

        input_type = modelservice.signature['input_type']
        output_type = modelservice.signature['output_type']

        # Get data from request according to input type
        if input_type == 'application/json':
            data = map(self.handler.get_form_data, input_names)
        elif input_type == 'image/jpeg':
            data = [self.handler.get_file_data(name).read() for name in input_names]
        else:
            raise Exception('%s is not supported for input content-type' % input_type)

        # Construct response according to output type
        if output_type == 'application/json':
            return self.handler.jsonify({'prediction': modelservice.inference(data)})
        elif output_type == 'image/jpeg':
            return self.handler.send_file(modelservice.inference(data), output_type)
        else:
            raise Exception('%s is not supported for output content-type' % output_type)

