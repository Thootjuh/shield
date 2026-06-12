def state2region(s):
    # Discretize each component
    d0 = min(4, max(-4, int(s[0] / 0.05)))   # 9 values
    d1 = min(6, max(-1, int(s[1] / 0.05)))    # 8 values
    d2 = min(4, max(-4, int(s[2] / 0.05)))    # 9 values
    d3 = min(4, max(-4, int(s[3] / 0.05)))    # 9 values
    d4 = min(4, max(-4, int(s[4] / 0.05)))    # 9 values
    d5 = min(3, max(-3, int(s[5] / 0.1)))    # 7 values
    d6 = int(s[6])                           # 2 values
    d7 = int(s[7])                           # 2 values

    # Shift ranges to start at 0
    d0 += 4
    d1 += 1
    d2 += 4
    d3 += 4
    d4 += 4
    d5 += 3

    # Encode into a single integer
    index = d0
    index = index * 8 + d1
    index = index * 9 + d2
    index = index * 9 + d3
    index = index * 9 + d4
    index = index * 7 + d5
    index = index * 2 + d6
    index = index * 2 + d7

    return index


def region2centre(index):
    """
    Convert an abstract state index back into the
    centre of its corresponding continuous region.
    """

    # Decode dimensions
    d7 = index % 2
    index //= 2

    d6 = index % 2
    index //= 2

    d5 = index % 7
    index //= 7

    d4 = index % 9
    index //= 7

    d3 = index % 9
    index //= 7

    d2 = index % 9
    index //= 7

    d1 = index % 8
    index //= 7

    d0 = index

    # Shift back to signed ranges
    d0 -= 4
    d1 -= 1
    d2 -= 4
    d3 -= 4
    d4 -= 4
    d5 -= 3

    # Return centre of each discretized bin
    return (
        (d0 + 0.5) * 0.05,   # x position
        (d1 + 0.5) * 0.05,    # y position
        (d2 + 0.5) * 0.05,    # x velocity
        (d3 + 0.5) * 0.05,    # y velocity
        (d4 + 0.5) * 0.05,   # angle
        (d5 + 0.5) * 0.1,    # angular velocity
        d6,                  # left leg contact
        d7                   # right leg contact
    )

def get_discretized_dataset(trajectories):
        discretized_trajectories = []
        print("trap region  = ", get_trap_region())
        print("goal region  = ", get_goal_region())
        for traj in trajectories:
            new_traj = []

            for (s, a, ns, r, term, trunc) in traj:
                s_region = state2region(s)
                if r == -100:
                    ns_region = get_trap_region()
                elif r == 100:
                    ns_region = get_goal_region()
                else:
                    ns_region = state2region(ns)

                new_traj.append([a, s_region, ns_region, r])

            discretized_trajectories.append(new_traj)

        return discretized_trajectories

def get_nb_states():
    return 9*8*9*9*9*7*2*2+2

def get_goal_region():
    return get_nb_states()-1

def get_trap_region():
    return get_nb_states()-2

