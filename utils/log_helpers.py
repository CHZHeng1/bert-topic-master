import logging
import os
import sys
from datetime import datetime


def logger_init(log_file_name='monitor',
                log_level=logging.DEBUG,
                log_dir='./logs/',
                only_file=False):
    '''https://www.cnblogs.com/yyds/p/6901864.html logging详细说明'''
    # 指定路径
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_path = os.path.join(log_dir, log_file_name + '_' + str(datetime.now())[:10] + '.txt')
    formatter = '[%(asctime)s] - %(levelname)s: %(message)s'
    if only_file:
        logging.basicConfig(filename=log_path,
                            level=log_level,
                            format=formatter,
                            datefmt='%Y-%m-%d %H:%M:%S')
    else:
        logging.basicConfig(level=log_level,
                            format=formatter,
                            datefmt='%Y-%m-%d %H:%M:%S',
                            handlers=[logging.FileHandler(log_path), #将日志消息发送到磁盘文件，默认情况下文件大小会无限增长
                                      logging.StreamHandler(sys.stdout)] #将日志消息发送到输出到Stream
                            )