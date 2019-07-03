__all__ = (
    'main',
)

def main():
    """ execute main peakipy script """
    import sys
    from peakipy.scripts.peakipy import main as _main
    print("success")
    _main(sys.argv)

if __name__ == "__main__":
    main()
