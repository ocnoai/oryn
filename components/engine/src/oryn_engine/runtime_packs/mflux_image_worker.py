"""Compatibility entrypoint; implementation is owned by the MFLUX provider pack."""


def main():
    from oryn_mflux_provider.worker import main as provider_main

    provider_main()


if __name__ == "__main__":
    main()
