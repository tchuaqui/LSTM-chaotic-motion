import torch

class energy_conservation_loss():
    """
    Custom loss function using standard MSE and
    an additional penalty term incorporating energy conservation, such that:

    J(output, target) = (1 - alpha) * MSE(output, target) + alpha * P(output, target)

    where alpha is a user-defined hyperparameter.
    The penalty term P(output, target) is computed from the mean squared error in internal energy
    between the output (predicted) state and the target state such that:

    P(output, target) = mean( (Internal_energy(output) - Internal_energy(target))**2 ),
    with
    Internal_energy = 1/2 * m1 * l1**2 * omega1**2 + 1/2 * m2 * l2**2 * omega2**2 +\
                      m1 * g * (l1 + l2 - l1*cos(theta1)) + m2 * g * (l1 + l2 - l1*cos(theta1) - l2*cos(theta2))

    Note that MSE is computed on standardised output/targets whereas P requires unstandardised values.
    Alpha must be adjusted accordingly.
    """

    def __init__(self, M, L, g, alpha=0):
        self.alpha = alpha
        self.A = 0.5*M[0]*L[0]**2
        self.B = 0.5*M[1]*L[1]**2
        self.C = (M[0]+M[1])*g*L[0]
        self.D = M[1]*g*L[1]
        # Compute total and compute ratio of scalars instead
        self.T = self.A + self.B + self.C + self.D
        self.A = self.A/self.T
        self.B = self.B/self.T
        self.C = self.C/self.T
        self.D = self.D/self.T

    def loss(self, output, target, scaler):
        # Compute MSE
        mse = torch.mean((output - target)**2)
        # Inverse transformation to recover unstandardised outputs and targets
        scaler.inverse_transform(output)
        scaler.inverse_transform(target)
        # Compute penalty term
        P = torch.mean((self.A * (torch.square(output[:, :, 1]) - torch.square(target[:, :, 1])) + \
            self.B * (torch.square(output[:, :, 3]) - torch.square(target[:, :, 3])) - \
            self.C * (torch.cos(output[:, :, 0]) - torch.cos(target[:, :, 0])) - \
            self.D * (torch.cos(output[:, :, 2]) - torch.cos(target[:, :, 2])))**2)

        J = (1 - self.alpha)*mse + self.alpha*P
        return J

def accuracy_metric(output, target):
    """
    Model accuracy metric. Corresponds to the MSE of the standardised positions (angles).
    """
    return torch.mean((output[:,:,[0,2]] - target[:,:,[0,2]])**2)