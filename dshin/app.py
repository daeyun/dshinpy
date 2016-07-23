import sys

from dshin.third_party import gflags

FLAGS = gflags.FLAGS

gflags.DEFINE_boolean('debug', False, 'produces debugging output')


# Similar to https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/platform/app.py
def run(main=None):
    try:
        argv = FLAGS(sys.argv)  # parse flags
    except gflags.FlagsError as e:
        print('%s\nUsage: %s ARGS\n%s' % (e, sys.argv[0], FLAGS))
        sys.exit(1)
    if FLAGS.debug:
        print('non-flag arguments:', argv)
    main = main or sys.modules['__main__'].main

    if main.__code__.co_argcount == 0:
        status = main()
    else:
        status = main(sys.argv)

    # Exit code defaults to 0.
    sys.exit(status)
