# 2018年10月15日
# About Class Definition


class MyClass:
    # class variable, belongs to the class, shared by all instances, like static class variable in c++
    CLASS_VARIABLE = 0

    # __xx__ is the function or variable defined by the system, you can rewrite it
    # the class constructor
    def __init__(self, parameter1):
        # define member variables and do initialization
        self.instance_variable = 0 # instance variable, belongs to the object
        return


    # if you want your class can use operators == and !=, rewite these two functions
    def __ne__(self, other):
        # not equal
        return self.__hash__() != other.__hash__()

    def __eq__(self, other):
        # eq
        return self.__hash__() == other.__hash__()


class BaseClass:
    def __init__(self, a):
        self.a = a
        return

    def change(self):
        self.a = 1

    def public_function(self):
        # do something
        return

    def _protected_function(self):
        # do something
        return

    def __private_function(self):
        # do something
        return


class DerivedClass(BaseClass):
    def __init__(self):
        return


def func(a):
    a = 2


def func2(ac:BaseClass):
    ac.change()


if __name__ == "__main__":
    # instantiate a class
    myClass = MyClass(1)

    # access the public variables
    print(myClass.CLASS_VARIABLE)
    print(myClass.instance_variable)

    a = 0
    func(a)
    print(a)

    ac = BaseClass(0)
    print(ac.a)
    func2(ac)
    print(ac.a)