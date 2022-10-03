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
   spec     Plot spectra and make overlays

For help on specific <command> type peakipy <command> -h
E.g. peakipy read -h

"""
from docopt import docopt
from pkg_resources import get_distribution
import typer

def main(argv):
    version = get_distribution("peakipy").version
    args = docopt(
        __doc__, version=f"peakipy version {version}", options_first=True, argv=argv[1:]
    )

    argv = args["<args>"]
    if args["<command>"] == "read":
        import peakipy.commandline.read as read_peaklist

        #read_peaklist.main(argv)
        typer.run(read_peaklist.read)

    elif args["<command>"] == "fit":
        import peakipy.commandline.fit as fit

        fit.main(argv)

    elif args["<command>"] == "edit":
        import peakipy.commandline.edit as edit

        edit.main(argv)

    elif args["<command>"] == "check":
        import peakipy.commandline.check as check

        check.main(argv)

    elif args["<command>"] == "spec":
        import peakipy.commandline.spec as spec

        spec.main(argv)

    elif args["<command>"] == "help":
        print(__doc__)
        exit()

    else:
        print(__doc__)
        exit("%r is not a peakipy command. See 'peakipy help'." % args["<command>"])


if __name__ == "__main__":
    main()
