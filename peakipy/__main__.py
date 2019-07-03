__all__ = (
    'main',
)

def main():
    """ execute main peakipy script """
    import sys
    from peakipy.commandline.peakipy import main as _main
    _main(sys.argv)

if __name__ == "__main__":
    main()
