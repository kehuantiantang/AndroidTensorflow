# coding=utf-8
import datetime
import os
import os.path
import traceback
from keras.models import model_from_json
import ast
import copy
import json
import numpy as np
import matplotlib.pyplot as plt
import warnings
from filelock import FileLock
from shutil import rmtree
from keras.callbacks import History
from keras.utils import plot_model
import pandas as pd

# plt.switch_backend('agg')
# plt.switch_backend('TKAgg')


class loadWriteModel(object):
    """
    the model to save all important training data to file
    """

    def __init__(self, mother_path):
        """
        :param mother_path: log mother path
        """

        self.mother_path = mother_path

        # the variable is a constance just for key in dict
        self._index_name = '1_index'
        # self.columns_name = ['index', 'struct', 'summary', 'json', 'weight',
        #                      'history', 'timestamp', 'name', 'epoch']
        self.columns_name = ['index', 'timestamp', 'name', 'epoch']

        self.csv = None

    def statistic_summary(self, columns = ['name', 'acc', 'val_acc'], show =
    True):
        """
        read csv file and return statistic outline
        :param columns: which column want to show
        :param show: print or not
        :return: statistic outline
        """

        self.__read_csv()

        df = self.csv[columns].copy()
        if show:
            print(df)

        return df


    def __check_create_dir(self, paths):
        """
        if directory is not exit then create it
        :param paths: dict
        :return:
        """
        for _, p in paths.items():
            d = os.path.split(p)[0]
            if not os.path.exists(d):
                os.makedirs(d, exist_ok=True)

    @staticmethod
    def str2history(history_str):
        """
        convert the json file into history file
        :param history_str: a string which is the format of history or history file
        :return:    the file name and history
        """

        # file
        if os.sep in history_str:
            # history_str is a path, the path include '/'
            if os.path.exists(history_str):
                with open(history_str, 'r') as file:
                    return os.path.split(history_str)[-1], ast.literal_eval(
                        file.read())
            else:
                raise FileExistsError('File %s is not exit!' % history_str)
        else:
            # just a string
            return None, ast.literal_eval(history_str)

    def __check_name_conflict(self, name):
        """
        confirm this name is conflict or not and return a new name
        :param name:
        :return:
        """

        # search in csv and avoid conflict
        if name is None:
            # file name is not exit
            file_name = self.__timestamp() + '_%02d' % (np.random.randint(100))
        else:
            if not self.csv[self.csv['name'] == name].empty:
                regx = name + '_rep*'
                search_row = self.csv[self.csv['name'].str.contains(regx)]
                if search_row.empty:
                    prefix = name
                    suffix = '_rep1'
                else:
                    name = search_row[
                        search_row['index'] == search_row['index'].max()][
                        'name'].values[0]
                    prefix, suffix = name.split('_rep')
                    prefix = prefix + '_rep'
                    suffix = str(int(float(suffix)) + 1)
                file_name = prefix + suffix
            else:
                file_name = name

        return str(file_name)

    def __timestamp(self):
        now = datetime.datetime.now()
        return now.strftime('%Y%m%d_%H_%M_%S')

    def save(self, model, history, name=None, save_weights=True):
        """
        save the model parameter which include weight, model, history and their statistic
        :param model:
        :param history:
        :param name:
        :param save_weights:
        :param tensorboard_path:
        :return:
        """

        if type(history) == History:
            # 原始keras history类,获得dict
            history = history.history
        elif type(history) == str:
            # 字符串类,是history写成的json文件
            re_name, history = self.str2history(history)
            if name is None:
                name = re_name
        elif type(history) == dict:
            history = history
        else:
            raise ValueError('History must be instance of '
                             'keras.callbacks.History() or str but type is %s' % (
                                 type(history)))

        dir = self.build_filepath(None)
        # thread lock to keep data consistency
        lock = FileLock(dir['csv'] + '.lock')
        with lock:
            file_exists = os.path.isfile(dir['csv'])
            pd.set_option('display.width', 300)
            if file_exists:
                self.__read_csv()
            else:
                columns_name = self.columns_name
                for index in history.keys():
                    columns_name.append(index)
                columns_name = sorted(columns_name)
                self.csv = pd.DataFrame(columns=columns_name, dtype='object')

            statistic_data = {}
            statistic_data['index'] = self.csv[
                                          'index'].max() + 1 if file_exists else 0

            # timestamp
            statistic_data['timestamp'] = self.__timestamp()

            name = self.__check_name_conflict(name)
            statistic_data['name'] = name

            dir = self.build_filepath(name)
            try:
                #  struct to image
                plot_model(model, to_file=dir['struct'], show_shapes=True)

                #  summary to image
                self.__data_summary2image(file_path=dir['summary'],
                                          history=history)
                # model json
                model_json = model.to_json()
                with open(dir['model'], "w") as json_file:
                    json_file.write(model_json)

                # weight
                if save_weights:
                    model.save_weights(dir['weight'])

                #write history to txt
                with open(dir['history'], 'wb') as his_file:
                    hist_data = {}
                    for index in history.keys():

                        statistic_d = history[index]

                        # TODO mean of last x epochs
                        num_epoch = 5
                        if index in ['val_acc', 'acc', 'loss', 'val_loss']:
                            statistic_d = np.mean(statistic_d[-num_epoch:])
                        else:
                            statistic_d = statistic_d[-1]

                        #               statistic history data
                        statistic_data[str(index)] = format(statistic_d,
                                                            '.8f')
                        # epoch
                        if 'epoch' not in statistic_data.keys():
                            statistic_data['epoch'] = int(len(history[index]))

                        # history txt file dict
                        hist_data[index] = history[index]
                    his_file.write(str(hist_data).encode())

                self.csv = self.csv.append(statistic_data, ignore_index=True)
                self.csv.to_csv(dir['csv'], index=False)
            except Exception:
                traceback.print_exc()
                warnings.warn('Data save failed ! System Rollback',
                              RuntimeWarning)
                lock.release()
                self.delete(name)

        return name

    def __data_summary2image(self, history, file_path):
        """
        Show history data in image and save it to data
        :param history:
        :param file_path: file path to save image
        :return:
        """
        # build sub-image
        nrows = (len(history.keys()) + 3) // 2
        plot_number = 1

        fig = plt.figure(figsize=(14, 15), facecolor='w', edgecolor='k')

        h = copy.deepcopy(history)
        h['train_acc|val_acc'] = [h['acc'], h['val_acc']]
        h['train_loss|val_loss'] = [h['loss'], h['val_loss']]

        for key, value in sorted(h.items()):
            plt.subplot(nrows, 2, plot_number)
            if type(value[0]) == list:
                plt.plot(value[0], label=key.split('|')[0])
                plt.plot(value[1], label=key.split('|')[1])
                plt.legend(framealpha=0.3, fancybox=True, loc='best',
                           prop={'size': 10})
            else:
                plt.plot(value)

            plt.subplots_adjust(hspace=.5, wspace=.001)
            plt.title(key)
            plt.grid(True)
            plot_number = plot_number + 1

        plt.tight_layout()
        fig.savefig(file_path, dpi=200, format='jpg')
        plt.show()
        plt.close()

    def __del_special_characters(self, series):
        """
        remove special character in DataFrame
        :param series: Series
        :return:
        """
        if type(series) == pd.Series:
            series.apply(self.__del_special_characters)
        else:
            if type(series) == str:
                return series.strip('\t')
            else:
                return series

    def __old2new(self, his_pandas: pd.DataFrame) -> pd.DataFrame:
        """
        the old csv data has some problem, it can change the old data format file to new
        :type his_pandas: pd.DataFrame input pandas' history data
        :return:
        """
        # old file index name is 1_index
        if '1_index' in his_pandas.columns.tolist():
            his_pandas.rename(columns={'1_index': 'name'}, inplace=True)
            his_pandas['name'] = his_pandas['name'].apply(
                func=self.__del_special_characters)
            # insert index
            his_pandas['index'] = his_pandas.index

            # when this row is changed
            columns = sorted(his_pandas.columns.tolist())
            his_pandas = his_pandas.reindex(columns=columns)
        return his_pandas

    def __index_convert(self, index):
        """
        if index is string which index by column ->name, it will return name
        else if index is int which index by index , it will return index by
        column name
        :param index:   index to search row, maybe name or true index
        :return:
        """
        try:
            # index is column index
            index = int(float(index))
        except:
            pass
        finally:
            if type(index) == str:
                row = (self.csv['name'] == index)
            elif type(index) == int:
                row = (self.csv['index'] == index)
            else:
                raise ValueError("Index input error !")

            if sum(row) == 0:
                raise ValueError('Index {0} is not find in csv'.format(index))
            else:
                return str(self.csv[row]['name'].values[0])

    def __read_csv(self):
        self.csv = None
        file_path = self.mother_path + '/statistic.csv'
        assert os.path.exists(file_path), "Statistic.csv file is not " \
                                          "exist !"
        h = pd.read_csv(file_path, dtype=str)
        pd.set_option('display.width', 300)
        h = self.__old2new(h)

        # keep index is int and name is object, which convenient to compare
        # and avoid type compare error
        h['index'] = h['index'].astype(float)
        h['name'] = h['name'].astype(str)
        self.csv = h

    def load(self, index, weight_load=True, model_load=True,
             history_load=False, show_model_summary = False):
        """
        load the model parameter from index

        :param index: index, key to search log
        :param weight_load: weight load or not
        :param model_load:  model load or not
        :param history_load:    history load or not
        :param show_model_summary: print model summary or not
        :return: model, history
        """
        model = None
        his = None

        self.__read_csv()
        index = self.__index_convert(index)
        row = self.csv[self.csv.name == index]
        if len(row) == 0:
            warnings.warn('Can not find index %s in csv file' % index)

        paths = self.build_filepath(index)

        if (model_load == False and weight_load == True):
            raise RuntimeWarning('You must load model before load weight')
        if model_load:
            if os.path.exists(paths['model']):
                with open(paths['model'], 'r') as file_json:
                    model = model_from_json(file_json.read())
                if show_model_summary:
                    model.summary()
            else:
                warnings.warn('Can not find model %s' % paths['model'])

        if weight_load:
            if os.path.exists(paths['weight']):
                model.load_weights(paths['weight'], by_name=True)
            else:
                warnings.warn('Can not find weight %s' % paths['weight'])

        if history_load:
            if os.path.exists(paths['history']):
                _, his = self.str2history(paths['history'])
                self.__data_summary2image(history=his,
                                          file_path=paths['history'])
            else:
                warnings.warn('Can not find history %s' % paths['history'])

        return model, his

    def build_filepath(self, index=None):
        """
        build relate file directory and path
        :param index: if None, return csv path dir, else return model,
        struct, summary, history, summary, tensorboard path
        :return:  path
        """

        d = {}
        if index is not None:
            f_model = os.path.join(self.mother_path, 'model')
            d['struct'] = os.path.join(f_model, 'struct_' + str(index) + '.jpg')
            d['model'] = os.path.join(f_model, str(index) + '.json')

            d['weight'] = os.path.join(self.mother_path, 'weight',
                                       str(index) + '.h5')

            f_history = os.path.join(self.mother_path, 'history')
            d['summary'] = os.path.join(f_history,
                                        'summary_' + str(index) + '.jpg')
            d['history'] = os.path.join(f_history, str(index) + '.txt')

            d['tensorboard'] = os.path.join(self.mother_path, 'tensorboard',
                                            str(index) + '/')

            d['checkpoint'] = os.path.join(self.mother_path, 'checkpoint',
                                           str(index) + '/')

        d['csv'] = os.path.join(self.mother_path, 'statistic.csv')
        self.__check_create_dir(d)
        return d

    def fuzzy_delete(self, keyword):
        self.__read_csv()
        df = self.csv[self.csv.name.str.contains(keyword)]
        for n in df.name:
            self.delete(n)

    def delete(self, index):
        """
        Delete data summary
        :param index:   delete file index which can search by name or index
        :return:
        """
        paths = self.build_filepath(index)

        log = {'Success': {}, 'Fail': {}}
        #         delete files
        for key, p in paths.items():
            # csv delete
            if key == 'csv':
                try:
                    self.__read_csv()
                    self.csv = self.__old2new(self.csv)
                    lock = FileLock(paths['csv'] + '.lock')
                    with lock:
                        index = self.__index_convert(index)
                        data_remain = self.csv[self.csv.name != index]
                        # print(paths['csv'])
                        # print(data_remain)
                        data_remain.to_csv(paths['csv'],
                                           index=False)  # print('csv delete ', index)
                    lock.release()
                except:
                    log['Fail'][key] = p
                    traceback.print_exc()
                else:
                    log['Success'][key] = p
            else:
                #  other file and directory
                try:
                    if os.path.isdir(p):
                        rmtree(p)
                    else:
                        os.remove(p)
                except:
                    log['Fail'][key] = p  # traceback.print_exc()
                else:
                    log['Success'][key] = p
        print(json.dumps(log, indent=1))

    def multi_summary(self, summary, save_path='./compare'):
        """
        get compare summary image from every index
        :param summary: can be json file path or csv name
        :return:
        """
        s_num = len(summary)
        table = {'name': [None for _ in range(s_num)]}

        key_names = ['name']

        for i, s in enumerate(summary):
            if os.sep in s:
                _, h = self.str2history(s)
                s = os.path.split(s)[-1]
                n = s[: s.rfind('.')]
            else:
                _, h = self.load(s, weight_load=False, model_load=False,
                                 history_load=True)
                n = s

            table['name'][i] = n
            for key, value in h.items():
                if key not in key_names:
                    table[key] = [None for _ in range(s_num)]
                    key_names.append(key)

                table[key][i] = value

        os.makedirs(save_path, exist_ok=True)
        for key, value in table.items():
            if key != 'name':
                plt.figure()
                for i, v in enumerate(value):
                    if v is not None:
                        plt.plot(v, label=table['name'][i])

                plt.legend(loc='best', framealpha=0.5, fancybox=True)
                plt.title(key)
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(save_path, key + '.png'), dpi=500)
                plt.close()
