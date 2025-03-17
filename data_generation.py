import torch

def generate_data(x1range, x2range, numICs, mu, lam, T, dt, seed):
   # Set random seed for reproducibility
    torch.manual_seed(seed)

    # Generate random initial conditions for x1 and x2
    x1 = (x1range[1] - x1range[0]) * torch.rand(numICs) + x1range[0]
    x2 = (x2range[1] - x2range[0]) * torch.rand(numICs) + x2range[0]
    u = torch.rand(numICs, T) - 0.5

    dt_lam = dt * lam

    # Preallocate xu with shape [numICs, lenT, 3]
    xuk = torch.zeros(numICs, T, 3, dtype=torch.float32)

    xuk[:, :, 2] = u

    for t in range(T):

        xuk[:, t, 0] = x1
        xuk[:, t, 1] = x2

        dx1 = dt * mu * x1 + dt*u[:, t-1]
        dx2 = dt_lam * (x2 - x1**2)

        x1 += dx1
        x2 += dx2

    return xuk
   
def generate_two_link_data(q1_range, q2_range, dq1_range, dq2_range, numICs, T, dt, seed,
                           tau_max=1.0,
                           m1=1.0, m2=1.0,
                           l1=1.0, l2=1.0,
                           g=9.81):
    """
    Generate simulation data for a two-link planar manipulator.

    Parameters:
        q1_range, q2_range : tuple (min, max)
            Ranges for initial joint angles (in radians).
        dq1_range, dq2_range : tuple (min, max)
            Ranges for initial joint angular velocities.
        tau_max : float
            Maximum absolute torque applied at each joint.
        m1, m2 : float
            Masses of link 1 and link 2.
        l1, l2 : float
            Lengths of link 1 and link 2.
        g : float
            Acceleration due to gravity.

    Returns:
        data : torch.Tensor of shape [numICs, T, 6]
            For each trajectory and each time step, the first four entries are
            [q1, q2, dq1, dq2] and the last two are the applied torques [tau1, tau2].
    """
    torch.manual_seed(seed)

    # Compute the moments of inertia dynamically
    lc1, lc2 = l1 / 2, l2 / 2  # Assuming center of mass at middle of each link
    I1 = m1 * lc1**2  # Moment of inertia of link 1
    I2 = m2 * lc2**2  # Moment of inertia of link 2

    # Print computed values for debugging
    print(f"Computed Inertia: I1 = {I1:.4f}, I2 = {I2:.4f}")

    # Generate initial conditions
    q1 = (q1_range[1] - q1_range[0]) * torch.rand(numICs) + q1_range[0]
    q2 = (q2_range[1] - q2_range[0]) * torch.rand(numICs) + q2_range[0]
    dq1 = (dq1_range[1] - dq1_range[0]) * torch.rand(numICs) + dq1_range[0]
    dq2 = (dq2_range[1] - dq2_range[0]) * torch.rand(numICs) + dq2_range[0]

    # Generate random control torques for each time step
    tau = (torch.rand(numICs, T, 2) - 0.5) * 2 * tau_max

    # Preallocate data tensor
    data = torch.zeros(numICs, T, 6, dtype=torch.float32)
    data[:, :, 4:] = tau

    for t in range(T):
        # Save current state
        data[:, t, 0] = q1
        data[:, t, 1] = q2
        data[:, t, 2] = dq1
        data[:, t, 3] = dq2

        # Precompute trigonometric functions
        cos_q2 = torch.cos(q2)
        sin_q2 = torch.sin(q2)
        cos_q1q2 = torch.cos(q1 + q2)

        # Compute elements of the inertia matrix M(q)
        M11 = I1 + I2 + m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * cos_q2)
        M12 = I2 + m2 * (lc2**2 + l1 * lc2 * cos_q2)
        M21 = M12
        M22 = I2 + m2 * lc2**2

        # Compute the inverse of the inertia matrix
        detM = M11 * M22 - M12 * M21
        invM11 = M22 / detM
        invM12 = -M12 / detM
        invM21 = -M21 / detM
        invM22 = M11 / detM

        # Compute Coriolis terms
        h = -m2 * l1 * lc2 * sin_q2
        C1 = h * dq2 * (2 * dq1 + dq2)
        C2 = h * dq1**2

        # Compute gravity terms
        G1 = (m1 * lc1 + m2 * l1) * g * torch.cos(q1) + m2 * lc2 * g * cos_q1q2
        G2 = m2 * lc2 * g * cos_q1q2

        # Get current torques
        tau_t = tau[:, t, :]

        # Compute joint accelerations
        rhs1 = tau_t[:, 0] - C1 - G1
        rhs2 = tau_t[:, 1] - C2 - G2

        ddq1 = invM11 * rhs1 + invM12 * rhs2
        ddq2 = invM21 * rhs1 + invM22 * rhs2

        # Euler integration
        dq1 = dq1 + ddq1 * dt
        dq2 = dq2 + ddq2 * dt
        q1 = q1 + dq1 * dt
        q2 = q2 + dq2 * dt

    return data


def generate_data_unforced(x1range, x2range, numICs, mu, lam, T_step, dt, seed):
   # Set random seed for reproducibility
    torch.manual_seed(seed)

    # Generate random initial conditions for x1 and x2
    x1 = (x1range[1] - x1range[0]) * torch.rand(numICs) + x1range[0]
    x2 = (x2range[1] - x2range[0]) * torch.rand(numICs) + x2range[0]
    u = torch.rand(numICs, T_step)*0 # Sets input to be 0

    dt_lam = dt * lam

    # Preallocate xu with shape [numICs, lenT, 3]
    xuk = torch.zeros(numICs, T_step, 3, dtype=torch.float32)

    xuk[:, :, 2] = u

    for t in range(T_step):

        xuk[:, t, 0] = x1
        xuk[:, t, 1] = x2

        dx1 = dt * mu * x1 + dt*u[:, t-1]
        dx2 = dt_lam * (x2 - x1**2)

        x1 += dx1
        x2 += dx2

    return xuk

def DataGenerator(x1range, x2range, numICs, mu, lam, T, dt):

    # Create test, validation, and training tensors with different percentages of numICs
    seed = 1
    test_tensor = generate_data(x1range, x2range, round(0.1 * numICs), mu, lam, T, dt, seed)

    seed = 2
    val_tensor = generate_data(x1range, x2range, round(0.2 * numICs), mu, lam, T, dt, seed)

    seed = 3
    train_tensor = generate_data(x1range, x2range, round(0.7 * numICs), mu, lam, T, dt, seed)

    return train_tensor, test_tensor, val_tensor
   
def TwoLinkRobotDataGenerator(q1_range, q2_range, dq1_range, dq2_range, numICs, T, dt, tau_max):

    # Create test, validation, and training tensors with different percentages of numICs
    seed = 1
    test_tensor = generate_two_link_data(q1_range, q2_range, dq1_range, dq2_range, round(0.1 * numICs), T, dt, seed, tau_max)

    seed = 2
    val_tensor = generate_two_link_data(q1_range, q2_range, dq1_range, dq2_range, round(0.2 * numICs), T, dt, seed, tau_max)

    seed = 3
    train_tensor = generate_two_link_data(q1_range, q2_range, dq1_range, dq2_range, round(0.7 * numICs), T, dt, seed, tau_max)

    return train_tensor, test_tensor, val_tensor

def DataGenerator_mixed(x1range, x2range, numICs, mu, lam, T, dt):

    # Create test, validation, and training tensors with different percentages of numICs
    seed = 1
    test_tensor_unforced = generate_data_unforced(x1range, x2range, round(0.05 * numICs), mu, lam, T, dt, seed)

    seed = 2
    test_tensor_forced = generate_data(x1range, x2range, round(0.05 * numICs), mu, lam, T, dt, seed)

    seed = 3
    val_tensor = generate_data(x1range, x2range, round(0.1 * numICs), mu, lam, T, dt, seed)

    seed = 4
    train_tensor_unforced = generate_data_unforced(x1range, x2range, round(0.4 * numICs), mu, lam, T, dt, seed)

    seed = 5
    train_tensor_forced = generate_data(x1range, x2range, round(0.4 * numICs), mu, lam, T, dt, seed)

    return train_tensor_unforced, train_tensor_forced, test_tensor_unforced, test_tensor_forced, val_tensor
