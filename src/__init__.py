# src/__init__.py

from src.physics                  import PhysicalModels
from src.physics                  import Plasma
from src.physics.Plasma.magn      import BFields

from src.numerics.Simulations     import Simulation
from src.numerics                 import NumericalFluxes
from src.numerics                 import Propagators
from src.numerics                 import WeakForms

from src.utils                    import save_load
from src.utils.save_load          import save_xdmf
from src.utils                    import mesh_reader
from src.utils                    import mesh_creator
from src.utils                    import Elliptic_UFL
from src.utils                    import config_utils
from src.utils                    import MathAux

from src.postpro                  import extract_mn_2d
from src.postpro.Plotting         import plot_class
