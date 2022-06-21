# coding: utf-8

import argparse
import json
import os
import time


def get_cur_time_stamp():
    """Get current time. """

    lt = time.localtime(time.time())

    if lt.tm_mon <= 9:
        tm_mon = '0' + str(lt.tm_mon)
    else:
        tm_mon = str(lt.tm_mon)

    if lt.tm_hour <= 9:
        tm_hour = '0' + str(lt.tm_hour)
    else:
        tm_hour = str(lt.tm_hour)

    if lt.tm_mday <= 9:
        tm_mday = '0' + str(lt.tm_mday)
    else:
        tm_mday = str(lt.tm_mday)

    if lt.tm_min <= 9:
        tm_min = '0' + str(lt.tm_min)
    else:
        tm_min = str(lt.tm_min)

    time_stamp = "{}{}{}-{}{}".format(lt.tm_year, tm_mon, tm_mday, tm_hour, tm_min)

    return time_stamp


class ConfigBase(object):

    def __init__(self, save_path=None):

        self._prepare_preserved_keys()

        self.suffix = ''
        self.save_dir = save_path if save_path else "save"

        self.save_txt_flag = True
        self.save_json_flag = True
        self._parse_flag = False

    def _prepare_preserved_keys(self):

        """Block some variables from being saved in 'config.txt'. """

        self._preserved_keys = ['_preserved_keys', '_ordered_keys', 'save_txt_flag', 'save_json_flag',
                                '_parse_flag', 'suffix', 'save_dir']
        self._ordered_keys = []

    def items(self):

        """Return nested-tuple in form of (key, value) for keys in '_ordered_keys'. """

        return [(k, self.__dict__[k]) for k in self._ordered_keys]

    def parse(self):

        """Use 'argparse' module to save parameters. """

        parser = argparse.ArgumentParser()

        for key, value in self.items():
            if value is None:
                raise ValueError('args value cannot be None type')
            elif isinstance(value, list) or isinstance(value, tuple):
                parser.add_argument('--%s' % key, dest=key, type=type(value[0]), default=value, nargs="+")
            elif isinstance(value, bool):
                parser.add_argument('--%s' % key, dest=key, type=lambda x: (str(x).lower() == 'true'), default=value)
            else:
                parser.add_argument('--%s' % key, dest=key, type=type(value), default=value)
        args = parser.parse_args()
        args = vars(args)

        # Overwrite the dict.
        for key, value in args.items():
            self.__dict__[key] = value

        self.save_config()

        print(self)

        self._parse_flag = True

    def save_config(self):

        """Save in 'txt' and 'json'. """

        if self.save_txt_flag:
            with open(os.path.join(self.real_save_dir, 'config.txt'), 'w') as f:
                for key, value in self.items():
                    line = '{} = {}\n'.format(key, value)
                    f.write(line)

        if self.save_json_flag:
            temp = [self.__dict__]
            with open(os.path.join(self.real_save_dir, 'config.json'), 'w') as f:
                json.dump(temp, f)

    def _prepare_save_dir(self):

        """Get the save path. """

        cur_time_stamp = get_cur_time_stamp()
        real_save_dir = os.path.join(self.save_dir, cur_time_stamp + self.suffix)
        if not os.path.exists(real_save_dir):
            os.makedirs(real_save_dir)
            print("mkdir", real_save_dir)
        return real_save_dir

    @property
    def real_save_dir(self):

        return self._prepare_save_dir()

    def load_from_txt(self, txt_path):

        """Load parameters from 'txt'. """

        if self._parse_flag:
            raise ValueError('Please call `load_from_txt` before `parse` function!')

        with open(txt_path) as f:
            f = f.readlines()
            for line in f:
                line = line.strip()
                # Skip empty lines.
                if len(line) == 0:
                    continue
                # Skip comment lines.
                if line.startswith('#') or line.startswith('/'):
                    continue

                key = line.split('=')[0].strip()
                value = line.split('=')[1].strip()

                if not value:
                    value = ''
                    self.__setattr__(key, value)
                    continue
                try:
                    _value = eval(value)
                except:
                    _value = value
                self.__setattr__(key, _value)

    def load_from_json(self, json_path):

        """Load parameters from 'json'. """

        if self._parse_flag:
            raise ValueError('Please call `load_from_json` before `parse` function!')
        if not os.path.exists(json_path):
            return
        json_dict = json.load(open(json_path, 'r'))
        for key, value in json_dict[0].items():
            if key not in self._ordered_keys:
                print("The var `{}` in {} file does not exist in your code, so skipped.".format(key, json_path))
            else:
                self.__setattr__(key, value)

    def __setattr__(self, name, value):

        self.__dict__[name] = value
        if name != '_preserved_keys':
            if name not in self._preserved_keys + self._ordered_keys:
                self._ordered_keys.append(name)

    def __str__(self):

        """Return designed content instead of object's name and ID when `print(object)`. """

        lines = '>>>>> Params Config: <<<<<\n'
        for key, value in self.items():
            lines += '{} = {}\n'.format(key, value)
        return lines

    # Usage of '__repr__' is similar to '__str__' in most situations apart from `print()`.
    __repr__ = __str__


if __name__ == '__main__':
    class MyConfig(ConfigBase):

        def __init__(self):
            super(MyConfig, self).__init__()

            self.a = 1
            self.b = True
            self.c = False
            self.d = [1, 2]
            self.e = ['1', '2']
            self.f = 'hello'

            self.suffix = ''


    args = MyConfig()

    # Method 1. Init with 'MyConfig()'.
    args.parse()

    # Method 2. Load parameters from 'txt'.
    # args.load_from_txt('./save/20200901-0944/config.txt')
    # args.parse()

    # Method 3. Load parameters from 'json'.
    # args.load_from_json('./save/20200901-1600/config.json')
    # args.parse()

    print(args.a)
    print(args.__dict__)
