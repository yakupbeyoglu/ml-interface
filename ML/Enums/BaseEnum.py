from abc import ABC, abstractmethod
from enum import Enum
import enum


class BaseEnum(Enum):

    # support enum, string, integer
    def IsExist(self, function):
        isexist = False
        if type(function.__class__) == enum.EnumMeta:
            isexist = function.value in self._value2member_map_
        elif not type(function) == str:
            isexist = function in self._value2member_map_
        else:
            function = function.lower()
            isexist = function in self._member_names_
        return isexist

    
    def GetName(self, function):

        if not self.IsExist(self, function):
            return None
        if type(function.__class__) == enum.EnumMeta:
            return function.name
        elif type(function) == str:
            return function.lower()
        else:
            return self(function).name
