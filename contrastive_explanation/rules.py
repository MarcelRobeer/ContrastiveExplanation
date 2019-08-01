from enum import Enum
from collections import namedtuple

ROUND = 3


class Operator(Enum):
    EQ = '='
    NOTEQ = '/='
    SEQ = '<='
    GEQ = '>='
    ST = '<'
    GT = '>'
    PLUSEQ = '+='
    MINUSEQ = '-='


class Literal:
    def __init__(self, feature, operator, value, categorical=False):
        self.feature = feature
        self.operator = operator
        self.value = value
        self.categorical = categorical

    def __str__(self):
        if not self.categorical and type(self.value) is list:
            if len(self.value) > 1:
                return '{} in [{}]'.format(self.feature,
                    ', '.join(str(round(v, ROUND)) for v in self.value))
            else:
                self.value = self.value[0]
        if self.categorical:
            return f'{self.feature} {self.operator.value} {self.value}'
        return f'{self.feature} {self.operator.value} {round(self.value, ROUND)}'

    def __repr__(self):
        return f'Literal(feature={self.feature}, operator={self.operator!r}, '\
               f'value={self.value!r}, categorical={self.categorical})'

    def __getitem__(self, index):
        return [self.feature, self.operator, self.value,
                self.categorical][index]

    def __setitem__(self, index, value):
        if index == 0:
            self.feature = value
        elif index == 1:
            self.operator = value
        elif index == 3:
            self.value = value
        elif index == 4:
            self.continuous = value

    def apply(self, sample):
        '''Apply literal to a sample observation.

        Args:
            sample: numpy array

        Returns:
            True if literal is true, False otherwise.
        '''
        if self.operator in [Operator.MINUSEQ, Operator.PLUSEQ]:
            raise Exception('Cannot compare to truth value')

        if type(self.feature) is not int:
            raise Exception('Feature needs to be int')

        return eval(f'sample[self.feature] {self.operator.value} self.value')
