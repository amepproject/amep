import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field 
import numpy as np
from typing import Collection, Iterable, Sequence, Optional, Union, Any, Dict, List, Tuple
from pathlib import Path


class ReaderInterface(ABC):
    """Abstract interface for readers - defines required methods."""
    
    @property
    @abstractmethod
    def steps(self) -> np.ndarray:
        """Array of simulation steps."""
        pass
    
    @property
    @abstractmethod
    def times(self) -> np.ndarray:
        """Array of physical times."""
        pass
    
    @property
    @abstractmethod
    def savedir(self) -> str:
        """Directory containing the HDF5 file."""
        pass
    
    @property
    @abstractmethod
    def filename(self) -> Union[str, Path]:
        """Name of the HDF5 file."""
        pass


class FrameInterface(ABC):
    """Abstract interface for frames - defines required methods."""
    
    @property
    @abstractmethod
    def step(self) -> int:
        """Time step of the frame."""
        pass
    
    @property
    @abstractmethod
    def time(self) -> float:
        """Physical time of the frame."""
        pass
    
    @property
    @abstractmethod
    def center(self) -> np.ndarray:
        """Center of simulation box."""
        pass
    
    @property
    @abstractmethod
    def dim(self) -> int:
        """Spatial dimension."""
        pass
    
    @property
    @abstractmethod
    def box(self) -> np.ndarray:
        """Box boundaries."""
        pass


class TrajectoryInterface(ABC):
    """Abstract interface for trajectories."""
    
    @abstractmethod
    def __getitem__(self, item):
        """Get frame by index."""
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """Number of frames."""
        pass
    
    @property
    @abstractmethod
    def type(self) -> str:
        """Type of trajectory."""
        pass


class EvaluationInterface(ABC):
    """Abstract interface for evaluations."""
    
    @abstractmethod
    def keys(self) -> List[str]:
        """Available data keys."""
        pass
    
    @abstractmethod
    def save(self, path: str, **kwargs) -> None:
        """Save evaluation results."""
        pass








