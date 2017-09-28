import argparse
from export_model import export_model

class StoreDictKeyPair(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, 'model', {kv.split("=", 1)[0]: kv.split("=", 1)[1] for kv in values})
    
class ArgParser(object):
    """Argument parser
    """

class ArgParser(object):
    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser(prog='mxnet-model-serving', description='MXNet Model Serving')

        parser.add_argument('--model',
                            required=True,
                            action=StoreDictKeyPair,
                            metavar='KEY1=VAL1,KEY2=VAL2...',
                            nargs="+",
                            help='Models to be deployed')

        parser.add_argument('--process', help='Using user defined model service')

        parser.add_argument('--gen-api', help='Generate API')

        parser.add_argument('--port', help='Port')

        return parser.parse_args()


