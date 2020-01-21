from test_stylegan import *
import abc

# kwargs is the central data store
# everything writes and has access to it.
import easydict
args = easydict.EasyDict()

# interface
class TTTExperiment(abc.ABC):
    @abc.abstractmethod
    def setup(self, **kwargs):
        pass
    @abc.abstractmethod
    def save_results(self, **kwargs):
        pass
    @abc.abstractmethod
    def sample_z(self, **kwargs):
        pass
    @abc.abstractmethod
    def sample_w(self, **kwargs):
        pass
    @abc.abstractmethod
    def sample_n_stylegan_images_without_tt(self, **kwargs):
        pass
    @abc.abstractmethod
    def sample_n_stylegan_images_with_w_ttl(self, **kwargs):
        pass
    @abc.abstractmethod
    def sample_n_stylegan_images_with_z_tt(self, **kwargs):
        pass
    @abc.abstractmethod
    def sample_n_stylegan_images_with_coachgan(self, **kwargs):
        pass
    @abc.abstractmethod
    def sample_n_stylegan_images_with_prenetwork_ttt(self, **kwargs):
        pass
    @abc.abstractmethod
    def sample_n_stylegan_images_with_intranetwork_ttt(self, **kwargs):
        pass
    @abc.abstractmethod
    def sample_n_stylegan_images_with_pre_and_intranetwork_ttt(self, **kwargs):
        pass
    @abc.abstractmethod
    def setup_prenetwork_ttt(self, **kwargs):
        pass
    @abc.abstractmethod
    def setup_intranetwork_ttt(self, **kwargs):
        pass
    @abc.abstractmethod
    def train_prenetwork_ttt(self, **kwargs):
        pass
    @abc.abstractmethod
    def train_intranetwork_ttt(self, **kwargs):
        pass
    @abc.abstractmethod
    def train_prenetwork_and_intranetwork_ttt(self, **kwargs):
        pass
    @abc.abstractmethod
    def save_models(self, **kwargs):
        pass
    @abc.abstractmethod
    def coach_z(self, **kwargs):
        pass
    @abc.abstractmethod
    def coach_w(self, **kwargs):
        pass
