import baseAssignmentSimulations as bas


def main()-> None:

    """ Base assignment, using CLI interface """
    # bas.simulate_base_assignment_A()
    bas.simulate_base_assignment_B(epochs=1_000_000)


if __name__ == "__main__":
    main()
