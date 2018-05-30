import importlib

def create(cls):
    '''expects a string that can be imported as with a module.class name'''
    module_name, class_name = cls.rsplit(".",1)

    print('importing '+module_name)
    somemodule = importlib.import_module(module_name)
    print('getattr '+class_name)
    cls_instance = getattr(somemodule, class_name)
    print(cls_instance)


    return cls_instance
