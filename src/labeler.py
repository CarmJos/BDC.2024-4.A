import re

from src import config, utils

class Labeler:
    def label(self, filename, id_range=None) -> int:
        pass


class SedimentaryLabeler(Labeler):
    def __init__(self):
        self.label_pattern = re.compile(config.label_pattern)
        self.group_pattern = re.compile(config.group_pattern)
        self.id_pattern = re.compile(config.id_pattern)

    def label(self, filename, id_range=None) -> int:
        matches = self.label_pattern.findall(filename)
        if len(matches) < 2:
            return -1
        class_name = matches[0]
        stone_name = matches[1]

        matches = self.group_pattern.findall(filename)
        if not matches:
            return -1
        group_num = int(matches[0])

        matches = self.id_pattern.findall(filename)
        if not matches:
            return -1

        id_num = int(matches[0][2])

        if (id_range is not None and
                id_num not in id_range):
            return -1

        subclass_name, group_num = utils.sedimentary_type(group_num)

        return group_num


class StoneLabeler(Labeler):
    def __init__(self):
        self.name_pattern = re.compile(config.label_pattern)
        self.group_pattern = re.compile(config.group_pattern)
        self.id_pattern = re.compile(config.id_pattern)

    def label(self, filename, id_range=None) -> int:
        class_name = filename[0]

        matches = self.id_pattern.findall(filename)
        if not matches:
            return 0

        id_num = int(matches[0][2])

        if (id_range is not None and
                id_num not in id_range):
            return 0

        group_num = utils.stone_type(class_name)
        return group_num
