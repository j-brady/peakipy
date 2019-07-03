#!/usr/bin/env python3
"""
Usage: peakipy <command> [<args>...]

Options:
   -h, --help

peakipy commands are:
   read     Read peaklist and generate initial peak clusters 
   edit     Interactively edit fit parameters
   fit      Fit peaks
   check    Check individual fits and generate plots

See 'peakipy help <command>' for more information on a specific command.

"""
from subprocess import call

from docopt import docopt


def main(argv):
    args = docopt(__doc__,
                  version='peakipy version 0.1.17',
                  options_first=True,
                  argv=argv[1:])

    argv = args['<args>']
    if args['<command>'] == 'read':
        import peakipy.commandline.read_peaklist as read_peaklist
        read_peaklist.main(argv)

    elif args['<command>'] == 'fit':
        import peakipy.commandline.fit_peaks as fit_peaks
        fit_peaks.main(argv)

    elif args['<command>'] == 'edit':
        import peakipy.commandline.edit_fits as edit_fits
        edit_fits.main(argv)

    elif args['<command>'] == 'check':
        import peakipy.commandline.check_fits as check_fits
        check_fits.main(argv)

    else:
        exit("%r is not a peakipy command. See 'peakipy help'." % args['<command>'])

if __name__ == '__main__':
    main()

