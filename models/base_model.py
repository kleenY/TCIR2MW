import os
import torch

class BaseModel():
    """Class BaseModel.

    Notes:
        Auto-generated documentation. Please refine as needed.
    """
    def name(self):
        """Perform the name operation.

        Args:
            None

        Returns:
            Any: Result.
        """
        return 'BaseModel'

    def initialize(self, opt):
        """Initialize model networks and optimizers.

        Args:
            opt (Any): Description.

        Returns:
            Any: Result.
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    def set_input(self, input):
        """Set input data for the current iteration.

        Args:
            input (Tensor): Description.

        Returns:
            Any: Result.
        """
        self.input = input

    def forward(self):
        """Run the forward pass of the network.

        Args:
            None

        Returns:
            Tensor: Result.
        """
        pass

    # used in test time, no backprop
    def test(self):
        """Run inference (evaluation mode).

        Args:
            None

        Returns:
            Any: Result.
        """
        pass

    def get_image_paths(self):
        """Return paths to the current images.

        Args:
            None

        Returns:
            Any: Result.
        """
        pass

    def optimize_parameters(self):
        """Perform the optimize_parameters operation.

        Args:
            None

        Returns:
            Any: Result.
        """
        pass

    def get_current_visuals(self):
        """Perform the get_current_visuals operation.

        Args:
            None

        Returns:
            Any: Result.
        """
        return self.input

    def get_current_errors(self):
        """Perform the get_current_errors operation.

        Args:
            None

        Returns:
            Any: Result.
        """
        return {}

    def save(self, label):
        """Perform the save operation.

        Args:
            label (Any): Description.

        Returns:
            Any: Result.
        """
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        """Save network weights to disk.

        Args:
            network (Any): Description.
            network_label (Any): Description.
            epoch_label (int): Description.
            gpu_ids (Any): Description.

        Returns:
            Any: Result.
        """
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(gpu_ids[0])

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        """Load network weights from disk.

        Args:
            network (Any): Description.
            network_label (Any): Description.
            epoch_label (int): Description.

        Returns:
            Any: Result.
        """
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        """Update the learning rate for all schedulers.

        Args:
            None

        Returns:
            Any: Result.
        """
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)
