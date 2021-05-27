import logging
import sys
import os


def main():
    if not os.path.exists('logging'):
        os.makedirs('logging')
    # logging.basicConfig(level=logging.DEBUG,
    #                     filename='./logging/output_log.txt',
    #                     filemode='w',
    #                     format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    #                     datefmt='%H:%M:%S'
    #                     )
    # logger = logging.getLogger()
    # # logger.setLevel(logging.DEBUG)
    # # output_file_handler = logging.FileHandler("output.log")
    # stdout_handler = logging.StreamHandler(sys.stdout)
    #
    # # logger.addHandler(output_file_handler)
    # logger.addHandler(stdout_handler)
    # for i in range(1, 4):
    #     # logger.debug("This is line " + str(i))
    #     logger.debug('new new my test with the {}'.format(i))
    #

    # set up logging to file - see previous section for more details
    # including the console is built based on this the basicConfig level
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
                        datefmt='%m-%d %H:%M:%S',
                        filename='./logging/my_log.txt',
                        filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)s-%(levelname)s: %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logger = logging.getLogger()
    logger.addHandler(console)

    # Now, we can log to the root logger, or any other logger. First the root...
    logging.info('Jackdaws love my big sphinx of quartz.')
    logging.debug('This is a debug level msg.')
    logger.info('test logger will owrk?')

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.setFi

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)

    for i in range(1, 4):
        # logger.debug("This is line " + str(i))
        root.debug('new new my test with the {}'.format(i))
        if i == 2:
            root.info('pause here')




if __name__ == '__main__':
    main()
