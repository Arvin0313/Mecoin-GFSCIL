import argparse
import json
import os
import uuid


class ArgumentParserWrapper(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_argument('--dataset', type=str, default="CoraFull",
                          help='Class Incremental learning dataset name')
        

    
    def parse(self):
        args = super().parse_args()
        for key in args.__dict__:
            value = args.__dict__[key]
            if value in ["true", "False", "True", "false"]:
                if value == "true" or value == "True":
                    args.__dict__[key] = True
                else:
                    args.__dict__[key] = False
        args.identifier = str(uuid.uuid4())
        args.root_dir = os.environ.get("PROJECT_ROOT_DIR", "./")
        return args